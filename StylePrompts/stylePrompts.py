from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation, InvocationContext, invocation_output, BaseInvocationOutput
from invokeai.app.invocations.primitives import InputField
from invokeai.app.invocations.strings import String2Output
from invokeai.app.invocations.fields import UIComponent, OutputField

from typing import Literal

@invocation_output("pos_neg_prompt_output")
class PosNegPromptOutput(BaseInvocationOutput):
    """Base class for invocations that output two strings for the prompt and negative prompt"""

    positive_prompt: str = OutputField(description="Positive Prompt")
    negative_prompt: str = OutputField(description="Negative Prompt")

@invocation('style', version="1.0.0")
class StylePromptInvocation(BaseInvocation):
    '''wraps a style around prompt text'''

    Prompt: str = InputField(default="", description="Prompt", ui_component=UIComponent.Textarea)
    NegativePrompt: str = InputField(default="", description="Negative Prompt", ui_component=UIComponent.Textarea)
    Style: Literal['Enhance', 'Anime', 'Photographic', 'Digital art', 'Comic book', 'Fantasy art', 'Analog film', 'Neonpunk', 'Isometric', 'Lowpoly', 'Origami', 'Line art', 'Craft clay', 'Cinematic', '3d-model', 'Pixel art', 'Texture'] = InputField(default="Enhance", description="Style", ui_component=UIComponent.Textarea)



    
    def invoke(self, context: InvocationContext) -> PosNegPromptOutput:

        styles = {
            "Enhance" : {
            "Positive" : "breathtaking {prompt} . award-winning, professional, highly detailed",
            "Negative" : "ugly, deformed, noisy, blurry, distorted, grainy",
            },
            "Anime" : {
            "Positive" : "anime artwork {prompt} . anime style, key visual, vibrant, studio anime,  highly detailed",
            "Negative" : "photo, deformed, black and white, realism, disfigured, low contrast",
            },
            "Photographic" : {
            "Positive" : "cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed",
            "Negative" : "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly",
            },
            "Digital art" : {
            "Positive" : "concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed",
            "Negative" : "photo, photorealistic, realism, ugly",
            },
            "Comic book" : {
            "Positive" : "comic {prompt} . graphic illustration, comic art, graphic novel art, vibrant, highly detailed",
            "Negative" : "photograph, deformed, glitch, noisy, realistic, stock photo",
            },
            "Fantasy art" : {
            "Positive" : "ethereal fantasy concept art of  {prompt} . magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy",
            "Negative" : "photographic, realistic, realism, 35mm film, dslr, cropped, frame, text, deformed, glitch, noise, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, sloppy, duplicate, mutated, black and white",
            },
            "Analog film" : {
            "Positive" : "analog film photo {prompt} . faded film, desaturated, 35mm photo, grainy, vignette, vintage, Kodachrome, Lomography, stained, highly detailed, found footage",
            "Negative" : "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured",
            },
            "Neonpunk" : {
            "Positive" : "neonpunk style {prompt} . cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, ultra detailed, intricate, professional",
            "Negative" : "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured",
            },
            "Isometric" : {
            "Positive" : "isometric style {prompt} . vibrant, beautiful, crisp, detailed, ultra detailed, intricate",
            "Negative" : "deformed, mutated, ugly, disfigured, blur, blurry, noise, noisy, realistic, photographic",
            },
            "Lowpoly" : {
            "Positive" : "low-poly style {prompt} . low-poly game art, polygon mesh, jagged, blocky, wireframe edges, centered composition",
            "Negative" : "noisy, sloppy, messy, grainy, highly detailed, ultra textured, photo",
            },
            "Origami" : {
            "Positive" : "origami style {prompt} . paper art, pleated paper, folded, origami art, pleats, cut and fold, centered composition",
            "Negative" : "noisy, sloppy, messy, grainy, highly detailed, ultra textured, photo",
            },
            "Line art" : {
            "Positive" : "line art drawing {prompt} . professional, sleek, modern, minimalist, graphic, line art, vector graphics",
            "Negative" : "anime, photorealistic, 35mm film, deformed, glitch, blurry, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, mutated, realism, realistic, impressionism, expressionism, oil, acrylic",
            },
            "Craft clay" : {
            "Positive" : "play-doh style {prompt} . sculpture, clay art, centered composition, Claymation",
            "Negative" : "sloppy, messy, grainy, highly detailed, ultra textured, photo",
            },
            "Cinematic" : {
            "Positive" : "cinematic film still {prompt} . shallow depth of field, vignette, highly detailed, high budget Hollywood movie, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
            "Negative" : "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
            },
            "3d-model" : {
            "Positive" : "professional 3d model {prompt} . octane render, highly detailed, volumetric, dramatic lighting",
            "Negative" : "ugly, deformed, noisy, low poly, blurry, painting",
            },
            "Pixel art" : {
            "Positive" : "pixel-art {prompt} . low-res, blocky, pixel art style, 8-bit graphics",
            "Negative" : "sloppy, messy, blurry, noisy, highly detailed, ultra textured, photo, realistic",
            },
            "Texture" : {
            "Positive" : "texture {prompt} top down close-up",
            "Negative" : "ugly, deformed, noisy, blurry",
            }
        }

        prompt = styles[self.Style]["Positive"].replace("{prompt}", self.Prompt)
        negative_prompt = self.NegativePrompt + "," + styles[self.Style]["Negative"]

        return PosNegPromptOutput(positive_prompt=prompt, negative_prompt=negative_prompt)
