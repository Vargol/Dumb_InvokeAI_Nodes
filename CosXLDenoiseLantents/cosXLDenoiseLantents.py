import inspect
from contextlib import ExitStack
from typing import Any, Dict, Iterator, List, Tuple, Union

import torch
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from diffusers.schedulers.scheduling_dpmsolver_sde import DPMSolverSDEScheduler
from diffusers.schedulers.scheduling_dpmsolver_singlestep import DPMSolverSinglestepScheduler
from diffusers.schedulers.scheduling_tcd import TCDScheduler
from diffusers.schedulers.scheduling_utils import SchedulerMixin as Scheduler

from invokeai.app.invocations.baseinvocation import  invocation
from invokeai.app.invocations.fields import (
    InputField,
    UIComponent,
)
from invokeai.app.invocations.denoise_latents import DenoiseLatentsInvocation
from invokeai.app.invocations.ip_adapter import IPAdapterField
from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.config import AnyModelConfig
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelVariantType
from invokeai.backend.model_patcher import ModelPatcher
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.stable_diffusion import PipelineIntermediateState
from invokeai.backend.stable_diffusion.denoise_context import DenoiseContext, DenoiseInputs
from invokeai.backend.stable_diffusion.diffusion.custom_atttention import CustomAttnProcessor2_0
from invokeai.backend.stable_diffusion.diffusion_backend import StableDiffusionBackend
from invokeai.backend.stable_diffusion.extension_callback_type import ExtensionCallbackType
from invokeai.backend.stable_diffusion.extensions.freeu import FreeUExt
from invokeai.backend.stable_diffusion.extensions.inpaint import InpaintExt
from invokeai.backend.stable_diffusion.extensions.inpaint_model import InpaintModelExt
from invokeai.backend.stable_diffusion.extensions.lora import LoRAExt
from invokeai.backend.stable_diffusion.extensions.preview import PreviewExt
from invokeai.backend.stable_diffusion.extensions.rescale_cfg import RescaleCFGExt
from invokeai.backend.stable_diffusion.extensions.seamless import SeamlessExt
from invokeai.backend.stable_diffusion.extensions_manager import ExtensionsManager
from invokeai.backend.stable_diffusion.schedulers import SCHEDULER_MAP
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.silence_warnings import SilenceWarnings

from diffusers import EDMEulerScheduler


from typing import Literal, Iterator, Tuple, Union, List, Dict, Any
import inspect


import torch

def get_scheduler_old(
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



def get_scheduler(
    context: InvocationContext,
    scheduler_info: ModelIdentifierField,
    scheduler_name: str,
    seed: int,
    unet_config: AnyModelConfig,
) -> Scheduler:
    """Load a scheduler and apply some scheduler-specific overrides."""
    # TODO(ryand): Silently falling back to ddim seems like a bad idea. Look into why this was added and remove if
    # possible.
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

    if hasattr(unet_config, "prediction_type"):
        scheduler_config["prediction_type"] = unet_config.prediction_type

    # make dpmpp_sde reproducable(seed can be passed only in initializer)
    if scheduler_class is DPMSolverSDEScheduler:
        scheduler_config["noise_sampler_seed"] = seed

    if scheduler_class is DPMSolverMultistepScheduler or scheduler_class is DPMSolverSinglestepScheduler:
        if scheduler_config["_class_name"] == "DEISMultistepScheduler" and scheduler_config["algorithm_type"] == "deis":
            scheduler_config["algorithm_type"] = "dpmsolver++"

    scheduler = EDMEulerScheduler(sigma_min=0.002, sigma_max=120.0, sigma_data=1.0, prediction_type="v_prediction")
    #scheduler = scheduler_class.from_config(scheduler_config)

    # hack copied over from generate.py
    if not hasattr(scheduler, "uses_inpainting_model"):
        scheduler.uses_inpainting_model = lambda: False
    assert isinstance(scheduler, Scheduler)
    return scheduler

@invocation('denoise_latents_for_CosXL', version="1.0.0")
class CosXLDenoiseLatentsInvocation(DenoiseLatentsInvocation):
    '''Denoise latents for CosXL, which need a schedukler not support by InvokeAI ATOC'''

    scheduler: Literal['EDM Euler'] = InputField(default="EDM Euler", description="Style", ui_component=UIComponent.Textarea)


    @staticmethod
    def init_scheduler(
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

        return timesteps, init_timestep, scheduler_step_kwargs

    def iai_init_scheduler(
        scheduler: Union[Scheduler, ConfigMixin],
        device: torch.device,
        steps: int,
        denoising_start: float,
        denoising_end: float,
        seed: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        assert isinstance(scheduler, ConfigMixin)
        if scheduler.config.get("cpu_only", False):
            scheduler.set_timesteps(steps, device="cpu")
            timesteps = scheduler.timesteps.to(device=device)
        else:
            scheduler.set_timesteps(steps, device=device)
            timesteps = scheduler.timesteps

        # skip greater order timesteps
        _timesteps = timesteps[:: scheduler.order]

        # get start timestep index
        t_start_val = int(round(scheduler.config["num_train_timesteps"] * (1 - denoising_start)))
        t_start_idx = len(list(filter(lambda ts: ts >= t_start_val, _timesteps)))

        # get end timestep index
        t_end_val = int(round(scheduler.config["num_train_timesteps"] * (1 - denoising_end)))
        t_end_idx = len(list(filter(lambda ts: ts >= t_end_val, _timesteps[t_start_idx:])))

        # apply order to indexes
        t_start_idx *= scheduler.order
        t_end_idx *= scheduler.order

        init_timestep = timesteps[t_start_idx : t_start_idx + 1]
        timesteps = timesteps[t_start_idx : t_start_idx + t_end_idx]

        scheduler_step_kwargs: Dict[str, Any] = {}
        scheduler_step_signature = inspect.signature(scheduler.step)
        if "generator" in scheduler_step_signature.parameters:
            # At some point, someone decided that schedulers that accept a generator should use the original seed with
            # all bits flipped. I don't know the original rationale for this, but now we must keep it like this for
            # reproducibility.
            #
            # These Invoke-supported schedulers accept a generator as of 2024-06-04:
            #   - DDIMScheduler
            #   - DDPMScheduler
            #   - DPMSolverMultistepScheduler
            #   - EulerAncestralDiscreteScheduler
            #   - EulerDiscreteScheduler
            #   - KDPM2AncestralDiscreteScheduler
            #   - LCMScheduler
            #   - TCDScheduler
            scheduler_step_kwargs.update({"generator": torch.Generator(device=device).manual_seed(seed ^ 0xFFFFFFFF)})
        if isinstance(scheduler, TCDScheduler):
            scheduler_step_kwargs.update({"eta": 1.0})

        return timesteps, init_timestep, scheduler_step_kwargs


    @torch.no_grad()
    @SilenceWarnings()  # This quenches the NSFW nag from diffusers.
    def _new_invoke(self, context: InvocationContext) -> LatentsOutput:
        ext_manager = ExtensionsManager(is_canceled=context.util.is_canceled)

        device = TorchDevice.choose_torch_device()
        dtype = TorchDevice.choose_torch_dtype()

        seed, noise, latents = self.prepare_noise_and_latents(context, self.noise, self.latents)
        _, _, latent_height, latent_width = latents.shape

        # get the unet's config so that we can pass the base to sd_step_callback()
        unet_config = context.models.get_config(self.unet.unet.key)

        conditioning_data = self.get_conditioning_data(
            context=context,
            positive_conditioning_field=self.positive_conditioning,
            negative_conditioning_field=self.negative_conditioning,
            cfg_scale=self.cfg_scale,
            steps=self.steps,
            latent_height=latent_height,
            latent_width=latent_width,
            device=device,
            dtype=dtype,
            # TODO: old backend, remove
            cfg_rescale_multiplier=self.cfg_rescale_multiplier,
        )

        scheduler = get_scheduler(
            context=context,
            scheduler_info=self.unet.scheduler,
            scheduler_name=self.scheduler,
            seed=seed,
            unet_config=unet_config,
        )

        timesteps, init_timestep, scheduler_step_kwargs = self.init_scheduler(
            scheduler,
            seed=seed,
            device=device,
            steps=self.steps,
            denoising_start=self.denoising_start,
            denoising_end=self.denoising_end,
        )

        ### preview
        def step_callback(state: PipelineIntermediateState) -> None:
            context.util.sd_step_callback(state, unet_config.base)

        ext_manager.add_extension(PreviewExt(step_callback))

        ### cfg rescale
        if self.cfg_rescale_multiplier > 0:
            ext_manager.add_extension(RescaleCFGExt(self.cfg_rescale_multiplier))

        ### freeu
        if self.unet.freeu_config:
            ext_manager.add_extension(FreeUExt(self.unet.freeu_config))

        ### lora
        if self.unet.loras:
            for lora_field in self.unet.loras:
                ext_manager.add_extension(
                    LoRAExt(
                        node_context=context,
                        model_id=lora_field.lora,
                        weight=lora_field.weight,
                    )
                )
        ### seamless
        if self.unet.seamless_axes:
            ext_manager.add_extension(SeamlessExt(self.unet.seamless_axes))

        ### inpaint
        mask, masked_latents, is_gradient_mask = self.prep_inpaint_mask(context, latents)
        # NOTE: We used to identify inpainting models by inspecting the shape of the loaded UNet model weights. Now we
        # use the ModelVariantType config. During testing, there was a report of a user with models that had an
        # incorrect ModelVariantType value. Re-installing the model fixed the issue. If this issue turns out to be
        # prevalent, we will have to revisit how we initialize the inpainting extensions.
        if unet_config.variant == ModelVariantType.Inpaint:
            ext_manager.add_extension(InpaintModelExt(mask, masked_latents, is_gradient_mask))
        elif mask is not None:
            ext_manager.add_extension(InpaintExt(mask, is_gradient_mask))

        # Initialize context for modular denoise
        latents = latents.to(device=device, dtype=dtype)
        if noise is not None:
            noise = noise.to(device=device, dtype=dtype)
        denoise_ctx = DenoiseContext(
            inputs=DenoiseInputs(
                orig_latents=latents,
                timesteps=timesteps,
                init_timestep=init_timestep,
                noise=noise,
                seed=seed,
                scheduler_step_kwargs=scheduler_step_kwargs,
                conditioning_data=conditioning_data,
                attention_processor_cls=CustomAttnProcessor2_0,
            ),
            unet=None,
            scheduler=scheduler,
        )

        # context for loading additional models
        with ExitStack() as exit_stack:
            # later should be smth like:
            # for extension_field in self.extensions:
            #    ext = extension_field.to_extension(exit_stack, context, ext_manager)
            #    ext_manager.add_extension(ext)
            self.parse_controlnet_field(exit_stack, context, self.control, ext_manager)
            bgr_mode = self.unet.unet.base == BaseModelType.StableDiffusionXL
            self.parse_t2i_adapter_field(exit_stack, context, self.t2i_adapter, ext_manager, bgr_mode)

            # ext: t2i/ip adapter
            ext_manager.run_callback(ExtensionCallbackType.SETUP, denoise_ctx)

            with (
                context.models.load(self.unet.unet).model_on_device() as (cached_weights, unet),
                ModelPatcher.patch_unet_attention_processor(unet, denoise_ctx.inputs.attention_processor_cls),
                # ext: controlnet
                ext_manager.patch_extensions(denoise_ctx),
                # ext: freeu, seamless, ip adapter, lora
                ext_manager.patch_unet(unet, cached_weights),
            ):
                sd_backend = StableDiffusionBackend(unet, scheduler)
                denoise_ctx.unet = unet
                result_latents = sd_backend.latents_from_embeddings(denoise_ctx, ext_manager)

        # https://discuss.huggingface.co/t/memory-usage-by-later-pipeline-stages/23699
        result_latents = result_latents.detach().to("cpu")
        TorchDevice.empty_cache()

        name = context.tensors.save(tensor=result_latents)
        return LatentsOutput.build(latents_name=name, latents=result_latents, seed=None)

    @torch.no_grad()
    @SilenceWarnings()  # This quenches the NSFW nag from diffusers.
    def _old_invoke(self, context: InvocationContext) -> LatentsOutput:
        device = TorchDevice.choose_torch_device()
        seed, noise, latents = self.prepare_noise_and_latents(context, self.noise, self.latents)

        mask, masked_latents, gradient_mask = self.prep_inpaint_mask(context, latents)
        # At this point, the mask ranges from 0 (leave unchanged) to 1 (inpaint).
        # We invert the mask here for compatibility with the old backend implementation.
        if mask is not None:
            mask = 1 - mask

        # TODO(ryand): I have hard-coded `do_classifier_free_guidance=True` to mirror the behaviour of ControlNets,
        # below. Investigate whether this is appropriate.
        t2i_adapter_data = self.run_t2i_adapters(
            context,
            self.t2i_adapter,
            latents.shape,
            device=device,
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

        # get the unet's config so that we can pass the base to sd_step_callback()
        unet_config = context.models.get_config(self.unet.unet.key)

        def step_callback(state: PipelineIntermediateState) -> None:
            context.util.sd_step_callback(state, unet_config.base)

        def _lora_loader() -> Iterator[Tuple[ModelPatchRaw, float]]:
            for lora in self.unet.loras:
                lora_info = context.models.load(lora.lora)
                assert isinstance(lora_info.model, ModelPatchRaw)
                yield (lora_info.model, lora.weight)
                del lora_info
            return

        with (
            ExitStack() as exit_stack,
            context.models.load(self.unet.unet).model_on_device() as (cached_weights, unet),
            ModelPatcher.apply_freeu(unet, self.unet.freeu_config),
            SeamlessExt.static_patch_model(unet, self.unet.seamless_axes),  # FIXME
            # Apply the LoRA after unet has been moved to its target device for faster patching.
            LayerPatcher.apply_smart_model_patches(
                model=unet,
                patches=_lora_loader(),
                prefix="lora_unet_",
                dtype=unet.dtype,
                cached_weights=cached_weights,
            ),
        ):
            assert isinstance(unet, UNet2DConditionModel)
            latents = latents.to(device=device, dtype=unet.dtype)
            if noise is not None:
                noise = noise.to(device=device, dtype=unet.dtype)
            if mask is not None:
                mask = mask.to(device=device, dtype=unet.dtype)
            if masked_latents is not None:
                masked_latents = masked_latents.to(device=device, dtype=unet.dtype)

            scheduler = get_scheduler(
                context=context,
                scheduler_info=self.unet.scheduler,
                scheduler_name=self.scheduler,
                seed=seed,
                unet_config=unet_config,
            )

            pipeline = self.create_pipeline(unet, scheduler)

            _, _, latent_height, latent_width = latents.shape
            conditioning_data = self.get_conditioning_data(
                context=context,
                positive_conditioning_field=self.positive_conditioning,
                negative_conditioning_field=self.negative_conditioning,
                device=device,
                dtype=unet.dtype,
                latent_height=latent_height,
                latent_width=latent_width,
                cfg_scale=self.cfg_scale,
                steps=self.steps,
                cfg_rescale_multiplier=self.cfg_rescale_multiplier,
            )

            controlnet_data = self.prep_control_data(
                context=context,
                control_input=self.control,
                latents_shape=latents.shape,
                device=device,
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

            timesteps, init_timestep, scheduler_step_kwargs = self.init_scheduler(
                scheduler,
                device=device,
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
                is_gradient_mask=gradient_mask,
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


