#!/usr/bin/env python3
"""
Kaggle Notebook Setup Script for NSFW Playground
Run this in a Kaggle notebook with GPU enabled
"""

import os
import subprocess
import sys

def run_command(cmd):
    """Run shell command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_gpu():
    """Check GPU availability"""
    print("🔍 Checking GPU availability...")
    success, stdout, stderr = run_command("nvidia-smi")
    if success:
        print("✅ GPU detected!")
        print(stdout)
        return True
    else:
        print("❌ No GPU detected. Please enable GPU in Kaggle notebook settings.")
        return False

def install_dependencies():
    """Install required packages"""
    print("📦 Installing dependencies...")
    
    # Core dependencies
    packages = [
        "fastapi",
        "uvicorn[standard]",
        "transformers",
        "torch",
        "diffusers",
        "accelerate",
        "xformers",
        "Pillow",
        "pydantic",
        "python-multipart",
        "safetensors",
        "invisible_watermark",
        "sentencepiece",
        "cohere"
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        success, stdout, stderr = run_command(f"pip install {package}")
        if not success:
            print(f"⚠️ Warning: Failed to install {package}")
            print(stderr)
    
    print("✅ Dependencies installation completed!")

def check_torch_cuda():
    """Check PyTorch CUDA setup"""
    print("🔥 Checking PyTorch CUDA setup...")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"GPU {i}: {props.name} ({props.total_memory/1e9:.1f}GB)")
        return torch.cuda.is_available()
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def create_kaggle_api_file():
    """Create optimized API file for Kaggle"""
    print("📝 Creating Kaggle-optimized API file...")
    
    api_content = '''
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import io
import base64
import random
import gc
import os

app = FastAPI(title="NSFW Playground - Kaggle Edition")

# Global variables for models
text_pipeline = None
image_pipeline = None

class TextRequest(BaseModel):
    prompt: str
    max_length: int = 100
    api_key: str = "demo"

class ImageRequest(BaseModel):
    prompt: str
    negative_prompt: str = "blurry, low quality, distorted"
    width: int = 512
    height: int = 512
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    seed: int = -1
    api_key: str = "demo"

def load_text_model():
    """Load text generation model with memory optimization"""
    global text_pipeline
    if text_pipeline is None:
        print("Loading text generation model...")
        try:
            # Use a smaller model for Kaggle
            model_name = "microsoft/DialoGPT-medium"
            text_pipeline = pipeline(
                "text-generation",
                model=model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device=0 if torch.cuda.is_available() else -1
            )
            print("✅ Text model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading text model: {e}")
            raise
    return text_pipeline

def load_image_model():
    """Load image generation model with memory optimization"""
    global image_pipeline
    if image_pipeline is None:
        print("Loading image generation model...")
        try:
            model_id = "stabilityai/stable-diffusion-xl-base-1.0"
            image_pipeline = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                use_safetensors=True
            )
            if torch.cuda.is_available():
                image_pipeline = image_pipeline.to("cuda")
                # Memory optimizations
                image_pipeline.enable_model_cpu_offload()
                image_pipeline.enable_xformers_memory_efficient_attention()
            print("✅ Image model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading image model: {e}")
            raise
    return image_pipeline

@app.get("/")
async def root():
    return {"message": "NSFW Playground API - Kaggle Edition", "status": "running"}

@app.post("/generate-text")
async def generate_text(request: TextRequest):
    try:
        pipeline = load_text_model()
        
        # Generate text
        generated = pipeline(
            request.prompt,
            max_length=min(request.max_length, 200),  # Limit for Kaggle
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=pipeline.tokenizer.eos_token_id
        )
        
        generated_text = generated[0]["generated_text"]
        
        # Clean up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            "generated_text": generated_text,
            "prompt": request.prompt,
            "model": "DialoGPT-medium (Kaggle optimized)"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text generation failed: {str(e)}")

@app.post("/generate-image")
async def generate_image(request: ImageRequest):
    try:
        pipeline = load_image_model()
        
        # Set seed for reproducibility
        if request.seed == -1:
            seed = random.randint(0, 2**32 - 1)
        else:
            seed = request.seed
        
        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
        generator.manual_seed(seed)
        
        # Generate image with Kaggle-friendly settings
        image = pipeline(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=min(request.width, 768),  # Limit for Kaggle
            height=min(request.height, 768),
            num_inference_steps=min(request.num_inference_steps, 25),
            guidance_scale=request.guidance_scale,
            generator=generator
        ).images[0]
        
        # Convert to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Clean up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return {
            "image": img_base64,
            "seed": seed,
            "prompt": request.prompt,
            "model": "SDXL (Kaggle optimized)"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting NSFW Playground - Kaggle Edition")
    print("📊 GPU Status:", "Available" if torch.cuda.is_available() else "Not Available")
    
    # Start server
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )
'''
    
    with open("kaggle_api.py", "w") as f:
        f.write(api_content)
    
    print("✅ Kaggle API file created!")

def main():
    """Main setup function"""
    print("🎯 NSFW Playground - Kaggle Setup")
    print("=" * 40)
    
    # Check GPU
    if not check_gpu():
        print("\n⚠️ Please enable GPU in Kaggle notebook settings and restart.")
        return
    
    # Install dependencies
    install_dependencies()
    
    # Check PyTorch CUDA
    if not check_torch_cuda():
        print("\n❌ PyTorch CUDA setup failed.")
        return
    
    # Create optimized API file
    create_kaggle_api_file()
    
    print("\n" + "=" * 40)
    print("✅ Setup completed successfully!")
    print("\n🚀 To start the application:")
    print("   python kaggle_api.py")
    print("\n🌐 The API will be available at:")
    print("   http://localhost:8000")
    print("\n📝 API Endpoints:")
    print("   POST /generate-text")
    print("   POST /generate-image")
    print("\n💡 Tip: Use smaller image sizes (512x512) for faster generation on Kaggle.")

if __name__ == "__main__":
    main()