from fastapi import FastAPI, HTTPException
from transformers import pipeline
from pydantic import BaseModel
import torch
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
from PIL import Image
import io
import base64
import random
from fastapi.responses import JSONResponse

class TextRequest(BaseModel):
    prompt: str
    max_length: int = 100

class ImageRequest(BaseModel):
    prompt: str
    negative_prompt: str = "low quality, blurry, distorted"
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    seed: int = -1

app = FastAPI()
generator = pipeline('text-generation', model='mlabonne/gemma-3-12b-it-qat-abliterated', device=-1)  # CPU mode

# Initialize image generation pipeline
device = "cuda" if torch.cuda.is_available() else "cpu"
image_pipe = None

def load_image_pipeline():
    global image_pipe
    if image_pipe is None:
        try:
            image_pipe = StableDiffusionXLPipeline.from_pretrained(
                "Heartsync/NSFW-Uncensored",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                use_safetensors=True
            )
            image_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(image_pipe.scheduler.config)
            image_pipe = image_pipe.to(device)
            if device == "cuda":
                image_pipe.enable_model_cpu_offload()
        except Exception as e:
            print(f"Failed to load image pipeline: {e}")
    return image_pipe

@app.post("/generate-text")
async def generate_text(request: TextRequest):
    try:
        result = generator(request.prompt, max_length=request.max_length, num_return_sequences=1)
        return {"generated_text": result[0]['generated_text']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-image")
async def generate_image(request: ImageRequest):
    try:
        pipe = load_image_pipeline()
        if pipe is None:
            raise HTTPException(status_code=500, detail="Image generation pipeline not available")
        
        # Set random seed if not provided
        if request.seed == -1:
            request.seed = random.randint(0, 2**32 - 1)
        
        # Generate image
        generator_obj = torch.Generator(device=device).manual_seed(request.seed)
        
        image = pipe(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            generator=generator_obj
        ).images[0]
        
        # Convert image to base64
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            "image": f"data:image/png;base64,{img_str}",
            "seed": request.seed,
            "prompt": request.prompt
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))