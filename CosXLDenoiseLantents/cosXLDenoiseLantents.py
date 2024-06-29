from invokeai.app.invocations.baseinvocation import invocation, InvocationContext
from invokeai.app.invocations.primitives import InputField, LatentsOutput
from invokeai.app.invocations.fields import UIComponent
from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.app.invocations.latent import DenoiseLatentsInvocation
from invokeai.backend.util.silence_warnings import SilenceWarnings
from invokeai.backend.stable_diffusion import PipelineIntermediateState, set_seamless
from invokeai.backend.lora import LoRAModelRaw
from invokeai.backend.model_patcher import ModelPatcher
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.stable_diffusion.schedulers import SCHEDULER_MAP

from contextlib import ExitStack



from diffusers import SchedulerMixin as Scheduler, ConfigMixin 
from diffusers import EDMEulerScheduler
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.schedulers.scheduling_tcd import TCDScheduler


from typing import Literal, Iterator, Tuple, Union, List, Dict, Any
import inspect


import torch
from torch import mps

def get_scheduler(
    context: InvocationContext,
    scheduler_info: ModelIdentifierField,
    scheduler_name: str,
    seed: int,
) -> Scheduler:
    scheduler_class, scheduler_extra_config = SCHEDULER_MAP.get('EDMEulerScheduler', SCHEDULER_MAP["ddim"])
    orig_scheduler_info = context.models.load(scheduler_info)
    with orig_scheduler_info as orig_scheduler:
        scheduler_config = orig_scheduler.config

    if "_backup" in scheduler_config:
        scheduler_config = scheduler_config["_backup"]
    scheduler_config = {
        **scheduler_config,
        **scheduler_extra_config,  # FIXME
        "_backup": scheduler_config,
    }

    scheduler = EDMEulerScheduler(sigma_min=0.002, sigma_max=120.0, sigma_data=1.0, prediction_type="v_prediction")

    # hack copied over from generate.py
    if not hasattr(scheduler, "uses_inpainting_model"):
        scheduler.uses_inpainting_model = lambda: False
    assert isinstance(scheduler, Scheduler)
    return scheduler

@invocation('denoise_latents_for_CosXL', version="1.0.0")
class CosXLDenoiseLatentsInvocation(DenoiseLatentsInvocation):
    '''Denoise latents for CosXL, which need a schedukler not support by InvokeAI ATOC'''

    scheduler: Literal['EDM Euler'] = InputField(default="EDM Euler", description="Style", ui_component=UIComponent.Textarea)



    def init_scheduler(
        self,
        scheduler: Union[Scheduler, ConfigMixin],
        device: torch.device,
        steps: int,
        denoising_start: float,
        denoising_end: float,
        seed: int,
    ) -> Tuple[int, List[int], int, Dict[str, Any]]:
        assert isinstance(scheduler, ConfigMixin)
        if scheduler.config.get("cpu_only", False):
            scheduler.set_timesteps(steps, device="cpu")
            timesteps = scheduler.timesteps.to(device=device)
        else:
            scheduler.set_timesteps(steps, device=device)
            timesteps = scheduler.timesteps

        print("xxxxxxxCALLEDxxxxxxxx")

        # skip greater order timesteps
        _timesteps = timesteps[:: scheduler.order]

        # get start timestep index
        t_start_idx = 0

        # get end timestep index
        t_end_idx = len(timesteps)

        # apply order to indexes
        t_start_idx *= scheduler.order
        t_end_idx *= scheduler.order

        init_timestep = timesteps[t_start_idx : t_start_idx + 1]
        timesteps = timesteps[t_start_idx : t_start_idx + t_end_idx]
        num_inference_steps = len(timesteps) // scheduler.order

        scheduler_step_kwargs: Dict[str, Any] = {}
        scheduler_step_signature = inspect.signature(scheduler.step)
        if "generator" in scheduler_step_signature.parameters:
            # At some point, someone decided that schedulers that accept a generator should use the original seed with
            # all bits flipped. I don't know the original rationale for this, but now we must keep it like this for
            # reproducibility.
            scheduler_step_kwargs.update({"generator": torch.Generator(device=device).manual_seed(seed ^ 0xFFFFFFFF)})
        if isinstance(scheduler, TCDScheduler):
            scheduler_step_kwargs.update({"eta": 1.0})

        return num_inference_steps, timesteps, init_timestep, scheduler_step_kwargs


    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        with SilenceWarnings():  # this quenches NSFW nag from diffusers
            seed = None
            noise = None
            if self.noise is not None:
                noise = context.tensors.load(self.noise.latents_name)
                seed = self.noise.seed

            if self.latents is not None:
                latents = context.tensors.load(self.latents.latents_name)
                if seed is None:
                    seed = self.latents.seed

                if noise is not None and noise.shape[1:] != latents.shape[1:]:
                    raise Exception(f"Incompatable 'noise' and 'latents' shapes: {latents.shape=} {noise.shape=}")

            elif noise is not None:
                latents = torch.zeros_like(noise)
            else:
                raise Exception("'latents' or 'noise' must be provided!")

            if seed is None:
                seed = 0

            mask, masked_latents, gradient_mask = self.prep_inpaint_mask(context, latents)

            # TODO(ryand): I have hard-coded `do_classifier_free_guidance=True` to mirror the behaviour of ControlNets,
            # below. Investigate whether this is appropriate.
            t2i_adapter_data = self.run_t2i_adapters(
                context,
                self.t2i_adapter,
                latents.shape,
                do_classifier_free_guidance=True,
            )

            ip_adapters: List[IPAdapterField] = []
            if self.ip_adapter is not None:
                # ip_adapter could be a list or a single IPAdapterField. Normalize to a list here.
                if isinstance(self.ip_adapter, list):
                    ip_adapters = self.ip_adapter
                else:
                    ip_adapters = [self.ip_adapter]

            # If there are IP adapters, the following line runs the adapters' CLIPVision image encoders to return
            # a series of image conditioning embeddings. This is being done here rather than in the
            # big model context below in order to use less VRAM on low-VRAM systems.
            # The image prompts are then passed to prep_ip_adapter_data().
            image_prompts = self.prep_ip_adapter_image_prompts(context=context, ip_adapters=ip_adapters)

            # get the unet's config so that we can pass the base to dispatch_progress()
            unet_config = context.models.get_config(self.unet.unet.key)

            def step_callback(state: PipelineIntermediateState) -> None:
                context.util.sd_step_callback(state, unet_config.base)

            def _lora_loader() -> Iterator[Tuple[LoRAModelRaw, float]]:
                for lora in self.unet.loras:
                    lora_info = context.models.load(lora.lora)
                    assert isinstance(lora_info.model, LoRAModelRaw)
                    yield (lora_info.model, lora.weight)
                    del lora_info
                return

            unet_info = context.models.load(self.unet.unet)
            assert isinstance(unet_info.model, UNet2DConditionModel)
            with (
                ExitStack() as exit_stack,
                unet_info as unet,
                ModelPatcher.apply_freeu(unet, self.unet.freeu_config),
                set_seamless(unet, self.unet.seamless_axes),  # FIXME
                # Apply the LoRA after unet has been moved to its target device for faster patching.
                ModelPatcher.apply_lora_unet(unet, _lora_loader()),
            ):
                assert isinstance(unet, UNet2DConditionModel)
                latents = latents.to(device=unet.device, dtype=unet.dtype)
                if noise is not None:
                    noise = noise.to(device=unet.device, dtype=unet.dtype)
                if mask is not None:
                    mask = mask.to(device=unet.device, dtype=unet.dtype)
                if masked_latents is not None:
                    masked_latents = masked_latents.to(device=unet.device, dtype=unet.dtype)

                scheduler = get_scheduler(
                    context=context,
                    scheduler_info=self.unet.scheduler,
                    scheduler_name=self.scheduler,
                    seed=seed,
                )

                pipeline = self.create_pipeline(unet, scheduler)

                _, _, latent_height, latent_width = latents.shape
                conditioning_data = self.get_conditioning_data(
                    context=context, unet=unet, latent_height=latent_height, latent_width=latent_width
                )
                
                controlnet_data = self.prep_control_data(
                    context=context,
                    control_input=self.control,
                    latents_shape=latents.shape,
                    # do_classifier_free_guidance=(self.cfg_scale >= 1.0))
                    do_classifier_free_guidance=True,
                    exit_stack=exit_stack,
                )

                ip_adapter_data = self.prep_ip_adapter_data(
                    context=context,
                    ip_adapters=ip_adapters,
                    image_prompts=image_prompts,
                    exit_stack=exit_stack,
                    latent_height=latent_height,
                    latent_width=latent_width,
                    dtype=unet.dtype,
                )
                
                num_inference_steps, timesteps, init_timestep, scheduler_step_kwargs = self.init_scheduler(
                    scheduler,
                    device=unet.device,
                    steps=self.steps,
                    denoising_start=self.denoising_start,
                    denoising_end=self.denoising_end,
                    seed=seed,
                )

                result_latents = pipeline.latents_from_embeddings(
                    latents=latents,
                    timesteps=timesteps,
                    init_timestep=init_timestep,
                    noise=noise,
                    seed=seed,
                    mask=mask,
                    masked_latents=masked_latents,
                    gradient_mask=gradient_mask,
                    num_inference_steps=num_inference_steps,
                    scheduler_step_kwargs=scheduler_step_kwargs,
                    conditioning_data=conditioning_data,
                    control_data=controlnet_data,
                    ip_adapter_data=ip_adapter_data,
                    t2i_adapter_data=t2i_adapter_data,
                    callback=step_callback,
                )

            # https://discuss.huggingface.co/t/memory-usage-by-later-pipeline-stages/23699
            result_latents = result_latents.to("cpu")
            TorchDevice.empty_cache()

            name = context.tensors.save(tensor=result_latents)
        return LatentsOutput.build(latents_name=name, latents=result_latents, seed=None)


'''
    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        with SilenceWarnings():  # This quenches the NSFW nag from diffusers.
            seed = None
            noise = None
            if self.noise is not None:
                noise = context.tensors.load(self.noise.latents_name)
                seed = self.noise.seed

            if self.latents is not None:
                latents = context.tensors.load(self.latents.latents_name)
                if seed is None:
                    seed = self.latents.seed

                if noise is not None and noise.shape[1:] != latents.shape[1:]:
                    raise Exception(f"Incompatable 'noise' and 'latents' shapes: {latents.shape=} {noise.shape=}")

            elif noise is not None:
                latents = torch.zeros_like(noise)
            else:
                raise Exception("'latents' or 'noise' must be provided!")

            if seed is None:
                seed = 0

            mask, masked_latents, gradient_mask = self.prep_inpaint_mask(context, latents)

            # TODO(ryand): I have hard-coded `do_classifier_free_guidance=True` to mirror the behaviour of ControlNets,
            # below. Investigate whether this is appropriate.
            t2i_adapter_data = self.run_t2i_adapters(
                context,
                self.t2i_adapter,
                latents.shape,
                do_classifier_free_guidance=True,
            )

            ip_adapters: List[IPAdapterField] = []
            if self.ip_adapter is not None:
                # ip_adapter could be a list or a single IPAdapterField. Normalize to a list here.
                if isinstance(self.ip_adapter, list):
                    ip_adapters = self.ip_adapter
                else:
                    ip_adapters = [self.ip_adapter]

            # If there are IP adapters, the following line runs the adapters' CLIPVision image encoders to return
            # a series of image conditioning embeddings. This is being done here rather than in the
            # big model context below in order to use less VRAM on low-VRAM systems.
            # The image prompts are then passed to prep_ip_adapter_data().
            image_prompts = self.prep_ip_adapter_image_prompts(context=context, ip_adapters=ip_adapters)

            # get the unet's config so that we can pass the base to dispatch_progress()
            unet_config = context.models.get_config(self.unet.unet.key)

            def step_callback(state: PipelineIntermediateState) -> None:
                context.util.sd_step_callback(state, unet_config.base)

            def _lora_loader() -> Iterator[Tuple[LoRAModelRaw, float]]:
                for lora in self.unet.loras:
                    lora_info = context.models.load(lora.lora)
                    assert isinstance(lora_info.model, LoRAModelRaw)
                    yield (lora_info.model, lora.weight)
                    del lora_info
                return

            unet_info = context.models.load(self.unet.unet)
            assert isinstance(unet_info.model, UNet2DConditionModel)
            with (
                ExitStack() as exit_stack,
                unet_info.model_on_device() as (model_state_dict, unet),
                ModelPatcher.apply_freeu(unet, self.unet.freeu_config),
                set_seamless(unet, self.unet.seamless_axes),  # FIXME
                # Apply the LoRA after unet has been moved to its target device for faster patching.
                ModelPatcher.apply_lora_unet(
                    unet,
                    loras=_lora_loader(),
                    model_state_dict=model_state_dict,
                ),
            ):
                assert isinstance(unet, UNet2DConditionModel)
                latents = latents.to(device=unet.device, dtype=unet.dtype)
                if noise is not None:
                    noise = noise.to(device=unet.device, dtype=unet.dtype)
                if mask is not None:
                    mask = mask.to(device=unet.device, dtype=unet.dtype)
                if masked_latents is not None:
                    masked_latents = masked_latents.to(device=unet.device, dtype=unet.dtype)

                scheduler = get_scheduler(
                    context=context,
                    scheduler_info=self.unet.scheduler,
                    scheduler_name=self.scheduler,
                    seed=seed,
                )

                pipeline = self.create_pipeline(unet, scheduler)

                _, _, latent_height, latent_width = latents.shape
                conditioning_data = self.get_conditioning_data(
                    context=context, unet=unet, latent_height=latent_height, latent_width=latent_width
                )

                controlnet_data = self.prep_control_data(
                    context=context,
                    control_input=self.control,
                    latents_shape=latents.shape,
                    # do_classifier_free_guidance=(self.cfg_scale >= 1.0))
                    do_classifier_free_guidance=True,
                    exit_stack=exit_stack,
                )

                ip_adapter_data = self.prep_ip_adapter_data(
                    context=context,
                    ip_adapters=ip_adapters,
                    image_prompts=image_prompts,
                    exit_stack=exit_stack,
                    latent_height=latent_height,
                    latent_width=latent_width,
                    dtype=unet.dtype,
                )

                num_inference_steps, timesteps, init_timestep, scheduler_step_kwargs = self.init_scheduler(
                    scheduler,
                    device=unet.device,
                    steps=self.steps,
                    denoising_start=self.denoising_start,
                    denoising_end=self.denoising_end,
                    seed=seed,
                )

                result_latents = pipeline.latents_from_embeddings(
                    latents=latents,
                    timesteps=timesteps,
                    init_timestep=init_timestep,
                    noise=noise,
                    seed=seed,
                    mask=mask,
                    masked_latents=masked_latents,
                    gradient_mask=gradient_mask,
                    num_inference_steps=num_inference_steps,
                    scheduler_step_kwargs=scheduler_step_kwargs,
                    conditioning_data=conditioning_data,
                    control_data=controlnet_data,
                    ip_adapter_data=ip_adapter_data,
                    t2i_adapter_data=t2i_adapter_data,
                    callback=step_callback,
                )

            # https://discuss.huggingface.co/t/memory-usage-by-later-pipeline-stages/23699
            result_latents = result_latents.to("cpu")
            TorchDevice.empty_cache()

            name = context.tensors.save(tensor=result_latents)
            return LatentsOutput.build(latents_name=name, latents=result_latents, seed=None)
'''



