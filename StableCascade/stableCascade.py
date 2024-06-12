from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation, InvocationContext
from invokeai.app.invocations.fields import UIComponent, FieldDescriptions, InputField, WithBoard
from invokeai.app.util.misc import SEED_MAX
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.backend.util.devices import torch_dtype, choose_torch_device
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState
from invokeai.backend.util.util import image_to_dataURL
from invokeai.app.services.session_processor.session_processor_common import ProgressImage

from typing import Literal, Union, List
import torch
from diffusers import (
    StableCascadeDecoderPipeline,
    StableCascadePriorPipeline,
    StableCascadeUNet,
)

from PIL import Image


def interrupt_callback(pipeline, step, timestep, callback_kwargs):
    if pipeline.invokeai_context.util.is_canceled():
        pipeline._interrupt = True

    context_data=pipeline.invokeai_context._data
    events = pipeline.invokeai_context._services.events    

    intermediate_state = PipelineIntermediateState(step + pipeline.invokeai_steps["stage_c_steps"],     
                                                   1,
                                                   pipeline.invokeai_steps["total_steps"],
                                                   step,
                                                   None)

    events.emit_invocation_denoise_progress(
         context_data.queue_item,
         context_data.invocation,
         intermediate_state,
         ProgressImage(dataURL=pipeline.dataURL, width=8, height=8))
    
    return callback_kwargs

@invocation('stable_cascade', version="1.2.0")
class StableCascadeInvocation(BaseInvocation, WithBoard):
    '''Runs Stable Cascade Inference'''

    Prompt: str = InputField(default="", description="Prompt", ui_component=UIComponent.Textarea)
    NegativePrompt: str = InputField(default="", description="Prompt", ui_component=UIComponent.Textarea)

    StageC: Literal['Full Model', 'Lite Model', 'Invictus Redmond']= InputField(default="Full Model", description="Stage C (Prior) Model", title="Stage C (Prior) Model", ui_component=UIComponent.Textarea)
    StageCSteps: int = InputField(default=10, gt=0, description=FieldDescriptions.steps,  title="Stage C Steps")
    StageCScale: Union[float, List[float]] = InputField(
        default=8.0, ge=1, description=FieldDescriptions.cfg_scale, title="Stage C CFG Scale"
    )
    
    StageB: Literal['Full Model', 'Lite Model']= InputField(default="Full Model", description="Stage B (Decoder) Model", title="Stage B (Decoder) Model", ui_component=UIComponent.Textarea)
    StageBSteps: int = InputField(default=10, gt=0, description=FieldDescriptions.steps,  title="Stage B Steps")
    StageBScale: Union[float, List[float]] = InputField(
        default=0.0, ge=0, description=FieldDescriptions.cfg_scale, title="Stage B CFG Scale"
    )
    
    StageA: Literal['Original Model', 'Ollin Model']= InputField(default="Original Model", description="Stage A (VQGAN) Model", title="Stage A (VQGAN) Model", ui_component=UIComponent.Textarea)

    Seed: int = InputField(
        default=0,
        ge=0,
        le=SEED_MAX,
        description=FieldDescriptions.seed,
    )


    def invoke(self, context: InvocationContext) -> ImageOutput:

        temp_image = Image.new('RGB', (8, 8))
        dataURL = image_to_dataURL(temp_image, image_format="JPEG")

        invictus_c = self.StageC == 'Invictus Redmond'
        lite_c = self.StageC == 'Lite Model'
        lite_b = self.StageB == 'Lite Model'
        ollin_a = self.StageA == 'Ollin Model'

        device = choose_torch_device()
        dtype = torch_dtype(device)

        print(f"DEVICE: {device}")
        print(f"DTYPE: {dtype}")

        generator =  torch.Generator(device).manual_seed(self.Seed);

        prior_kwargs = { }
        decoder_kwargs = { }

        prior_unet = None
        stage_a_ft_hq = None
        decoder_unet = None

        if context.util.is_canceled():
            return None


        if lite_c:
            prior_unet = StableCascadeUNet.from_pretrained("stabilityai/stable-cascade-prior", 
                                                            subfolder="prior_lite", 
                                                            variant="bf16",
                                                            torch_dtype=(torch.float32 if dtype == torch.float16 else dtype) ) 
            prior_kwargs['prior'] = prior_unet
        

        if invictus_c:
            prior_unet = StableCascadeUNet.from_single_file("https://huggingface.co/artificialguybr/Invictus-Redmond/blob/main/InvictusRedmond-StageC-Updated.safetensors", 
                                                            torch_dtype=(torch.float32 if dtype == torch.float16 else dtype) ) 
            prior_kwargs['prior'] = prior_unet

        prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", 
                                                            torch_dtype=(torch.float32 if dtype == torch.float16 else dtype) ,
                                                            variant="bf16",
                                                            **prior_kwargs).to(device)
        
        prior.invokeai_context = context
        prior.invokeai_steps = {"stage_c_steps": 0, "total_steps": self.StageCSteps + self.StageBSteps}
        prior.dataURL = dataURL;

        prior_output = prior(
            prompt=self.Prompt,
            height=1024,
            width=1024,
            negative_prompt=self.NegativePrompt,
            guidance_scale=self.StageCScale,
            num_images_per_prompt=1,
            num_inference_steps=self.StageCSteps,
            generator=generator,
            callback_on_step_end=interrupt_callback
        )

        del prior
        del prior_unet

        if context.util.is_canceled():
            return None

        if lite_b:
            decoder_unet = StableCascadeUNet.from_pretrained("stabilityai/stable-cascade", 
                                                                subfolder="decoder_lite",
                                                                variant="bf16",
                                                                torch_dtype=dtype)
            decoder_kwargs['decoder'] = decoder_unet

        if ollin_a:
            from diffusers.pipelines.wuerstchen import PaellaVQModel
            stage_a_ft_hq = PaellaVQModel.from_pretrained("madebyollin/stage-a-ft-hq", torch_dtype=dtype).to(device)


        decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade",
                                                                text_encoder=None,
                                                                variant="bf16",
                                                                torch_dtype=dtype, 
                                                                **decoder_kwargs).to(device)
        if ollin_a:
            print("using ollin vqgan")
            decoder.vqgan = stage_a_ft_hq   
       

        decoder.invokeai_context = context
        decoder.invokeai_steps = {"stage_c_steps": self.StageCSteps, "total_steps": self.StageCSteps + self.StageBSteps}
        decoder.dataURL = dataURL;

        decoder_output = decoder(
            image_embeddings=prior_output.image_embeddings,
            prompt_embeds=prior_output.prompt_embeds,
            prompt_embeds_pooled=prior_output.prompt_embeds_pooled,
            negative_prompt_embeds=prior_output.negative_prompt_embeds,
            negative_prompt_embeds_pooled=prior_output.negative_prompt_embeds_pooled,
            guidance_scale=self.StageBScale,
            output_type="pil",
            num_inference_steps=self.StageBSteps,
            generator=generator,
            callback_on_step_end=interrupt_callback
        ).images[0]

        del stage_a_ft_hq
        del decoder_unet

        image_dto = context.images.save(image=decoder_output)
        return ImageOutput.build(image_dto)        
    
