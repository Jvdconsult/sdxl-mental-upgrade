from diffusers import StableDiffusionXLPipeline
from cog import BasePredictor, Input
import torch
from PIL import Image

class Predictor(BasePredictor):
    def setup(self):
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16"
        ).to("cuda")
        self.pipe.load_lora_weights("./mental_upgrade.safetensors")
        self.pipe.fuse_lora()

    def predict(
        self,
        prompt: str = Input(description="Prompt"),
        steps: int = Input(default=30),
        guidance_scale: float = Input(default=7.5)
    ) -> Image.Image:
        return self.pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale
        ).images[0]
