import litserve as ls
import torch
from diffusers import StableDiffusion3Pipeline
from PIL import Image
import io
from huggingface_hub import login
import os
from fastapi.responses import Response
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StableDiffusionAPI(ls.LitAPI):
    def setup(self, device):
        # Determine device
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"  # Apple Silicon GPU
        else:
            device = "cpu"
        print(f"Using device: {device}")
        
        # Login to Hugging Face
        login(token=os.environ.get('HF_TOKEN'))
        
        # Initialize the pipeline with optimizations for M1
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-medium", 
            torch_dtype=torch.float16,
            variant="fp16"
        ).to(device)

        self.pipe.load_lora_weights(
            "tensorart/stable-diffusion-3.5-medium-turbo",
            weight_name="lora_sd3.5m_turbo_8steps.safetensors",
        )

        self.pipe.fuse_lora()

        # Enable memory efficient attention
        self.pipe.enable_attention_slicing()
        
        # Enable torch compile for faster inference
        if hasattr(torch, 'compile') and device != "mps":  # MPS doesn't support torch.compile yet
            self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=True)
        
        self.device = device
        
        # CLIP token limit
        self.max_token_limit = 77
        # Get tokenizer from text encoder for proper token counting
        self.tokenizer = self.pipe.tokenizer

    def truncate_prompt(self, prompt):
        """Properly truncate prompt to stay within CLIP token limit using actual tokenizer"""
        if not prompt:
            return prompt

        print(prompt)
        
        # Tokenize the prompt
        tokens = self.tokenizer.encode(prompt)
        
        # Check if truncation is needed
        if len(tokens) <= self.max_token_limit:
            return prompt
            
        # Truncate tokens and decode back to text
        truncated_tokens = tokens[:self.max_token_limit]
        truncated_prompt = self.tokenizer.decode(truncated_tokens)
        
        logger.warning(f"Prompt truncated from {len(tokens)} tokens to {len(truncated_tokens)} tokens")
        return truncated_prompt

    def decode_request(self, request):
        # Extract prompt and optional parameters from request
        prompt = request.get("prompt", "")
        negative_prompt = request.get("negative_prompt", 
            "low quality, bad anatomy, worst quality, low resolution")
        
        # Truncate prompts to fit within CLIP token limits
        truncated_prompt = self.truncate_prompt(prompt)
        truncated_negative_prompt = self.truncate_prompt(negative_prompt)
        
        if truncated_prompt != prompt:
            logger.warning(f"Prompt was truncated due to token limit. Original length: {len(prompt)}")
            
        return {
            "prompt": truncated_prompt,
            "num_inference_steps": request.get("num_inference_steps", 28),  # Changed default to match SD3 example
            "guidance_scale": request.get("guidance_scale", 3.5),  # Changed default to match SD3 example
            "negative_prompt": truncated_negative_prompt
        }

    def predict(self, params):
        try:
            # Generate image using the pipeline
            output = self.pipe(
                prompt=params["prompt"],
                num_inference_steps=params["num_inference_steps"],
                guidance_scale=params["guidance_scale"],
                negative_prompt=params["negative_prompt"]
            )
            
            return output.images[0]  # Return first image from pipeline output
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            # Create a fallback image
            from PIL import Image as PILImage
            fallback_img = PILImage.new('RGB', (512, 512), color=(255, 255, 255))
            from PIL import ImageDraw
            draw = ImageDraw.Draw(fallback_img)
            draw.text((10, 10), f"Error: {str(e)}", fill=(0, 0, 0))
            draw.text((10, 30), f"Prompt: {params['prompt'][:50]}...", fill=(0, 0, 0))
            return fallback_img

    def encode_response(self, image):
        # Convert PIL image to bytes for response
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        return Response(content=img_byte_arr, media_type="image/png")

if __name__ == "__main__":
    api = StableDiffusionAPI()
    server = ls.LitServer(api, accelerator="auto")
    server.run(port=8000) 