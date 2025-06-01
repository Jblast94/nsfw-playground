# 🚀 Intel Jupyter Notebook Setup Guide

## High-Performance Testing Environment

**Your Setup**: Intel Jupyter Notebook + PyTorch + Unlimited GPU/VRAM

**Advantages**: Perfect for testing, model optimization, and development without resource constraints

---

## 📊 Environment Assessment

### Check Your Current Setup

```python
# Run this in a Jupyter cell to assess your environment
import torch
import sys
import platform
import psutil
import subprocess

print("=== System Information ===")
print(f"Platform: {platform.platform()}")
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Count: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}")
        print(f"  Memory: {props.total_memory/1e9:.1f}GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")

print(f"\nRAM: {psutil.virtual_memory().total/1e9:.1f}GB")
print(f"CPU Cores: {psutil.cpu_count()}")
```

---

## 🔧 Jupyter Notebook Optimization

### 1. Install Required Packages

```bash
# In a terminal or notebook cell with !
pip install fastapi uvicorn transformers diffusers accelerate xformers
pip install Pillow pydantic python-multipart python-dotenv requests
pip install safetensors invisible_watermark sentencepiece cohere
pip install jupyter-server-proxy  # For running FastAPI in Jupyter
```

### 2. Memory Management Configuration

```python
# Add this to the top of your notebook
import torch
import gc
import os

# Optimize memory usage
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
torch.backends.cudnn.benchmark = True

# Memory cleanup function
def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB / {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
```

---

## 🎯 Optimized API Setup for Jupyter

### Create Jupyter-Optimized API

Create a new cell with this optimized version:

```python
# Jupyter-Optimized NSFW Playground API
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from PIL import Image
import io
import base64
import random
import gc
import asyncio
import uvicorn
from threading import Thread
import nest_asyncio

# Enable nested event loops for Jupyter
nest_asyncio.apply()

app = FastAPI(title="NSFW Playground - Jupyter Edition")

# Global model variables
text_model = None
image_pipeline = None

class TextRequest(BaseModel):
    prompt: str
    max_length: int = 200
    temperature: float = 0.8
    top_p: float = 0.9
    api_key: str = "demo"

class ImageRequest(BaseModel):
    prompt: str
    negative_prompt: str = "blurry, low quality, distorted, ugly, bad anatomy"
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    seed: int = -1
    api_key: str = "demo"

def load_text_model():
    """Load optimized text generation model"""
    global text_model
    if text_model is None:
        print("Loading text generation model...")
        # Use the full model since we have unlimited resources
        model_name = "mlabonne/gemma-3-12b-it-qat-abliterated"
        
        try:
            text_model = pipeline(
                "text-generation",
                model=model_name,
                torch_dtype=torch.float16,
                device_map="auto",  # Automatic device mapping
                trust_remote_code=True
            )
            print("✅ Text model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading full model, falling back to smaller model: {e}")
            # Fallback to smaller model
            text_model = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-large",
                torch_dtype=torch.float16,
                device=0 if torch.cuda.is_available() else -1
            )
    return text_model

def load_image_model():
    """Load optimized image generation model"""
    global image_pipeline
    if image_pipeline is None:
        print("Loading image generation model...")
        try:
            model_id = "stabilityai/stable-diffusion-xl-base-1.0"
            image_pipeline = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16"
            )
            
            # Optimize for unlimited VRAM
            image_pipeline = image_pipeline.to("cuda")
            
            # Use faster scheduler
            image_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                image_pipeline.scheduler.config
            )
            
            # Enable optimizations
            image_pipeline.enable_xformers_memory_efficient_attention()
            image_pipeline.enable_model_cpu_offload()
            
            print("✅ Image model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading image model: {e}")
            raise
    return image_pipeline

@app.get("/")
async def root():
    return {"message": "NSFW Playground - Jupyter Edition", "status": "running"}

@app.post("/generate-text")
async def generate_text(request: TextRequest):
    try:
        model = load_text_model()
        
        # Generate with optimized settings
        generated = model(
            request.prompt,
            max_length=min(request.max_length, 500),  # Increased limit
            num_return_sequences=1,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=True,
            pad_token_id=model.tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
        
        generated_text = generated[0]["generated_text"]
        
        # Memory cleanup
        cleanup_memory()
        
        return {
            "generated_text": generated_text,
            "prompt": request.prompt,
            "model": "Gemma-3-12B (Jupyter Optimized)",
            "gpu_memory_used": f"{torch.cuda.memory_allocated()/1e9:.2f}GB"
        }
        
    except Exception as e:
        cleanup_memory()
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
        
        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        # Generate with high-quality settings (unlimited resources)
        image = pipeline(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=min(request.width, 1536),  # Higher resolution allowed
            height=min(request.height, 1536),
            num_inference_steps=min(request.num_inference_steps, 50),
            guidance_scale=request.guidance_scale,
            generator=generator
        ).images[0]
        
        # Convert to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG", quality=95)
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Memory cleanup
        cleanup_memory()
        
        return {
            "image": img_base64,
            "seed": seed,
            "prompt": request.prompt,
            "model": "SDXL (Jupyter Optimized)",
            "gpu_memory_used": f"{torch.cuda.memory_allocated()/1e9:.2f}GB"
        }
        
    except Exception as e:
        cleanup_memory()
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

@app.get("/memory-status")
async def memory_status():
    """Check current memory usage"""
    if torch.cuda.is_available():
        return {
            "gpu_memory_allocated": f"{torch.cuda.memory_allocated()/1e9:.2f}GB",
            "gpu_memory_reserved": f"{torch.cuda.memory_reserved()/1e9:.2f}GB",
            "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory/1e9:.2f}GB",
            "ram_usage": f"{psutil.virtual_memory().percent}%"
        }
    else:
        return {"error": "CUDA not available"}

@app.post("/cleanup")
async def cleanup():
    """Manual memory cleanup"""
    cleanup_memory()
    return {"status": "Memory cleaned up"}

# Function to run the server
def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

print("✅ Jupyter API setup complete!")
print("To start the server, run: start_server()")
```

### Start the Server

```python
# Start the FastAPI server in a separate thread
def start_server():
    server_thread = Thread(target=run_server, daemon=True)
    server_thread.start()
    print("🚀 Server starting on http://localhost:8000")
    print("📊 Monitor at http://localhost:8000/memory-status")
    return server_thread

# Start the server
server = start_server()
```

---

## 🧪 Testing and Development

### 1. Model Testing

```python
# Test text generation
import requests
import json

def test_text_generation(prompt, max_length=100):
    response = requests.post(
        "http://localhost:8000/generate-text",
        json={
            "prompt": prompt,
            "max_length": max_length,
            "temperature": 0.8
        }
    )
    return response.json()

# Test image generation
def test_image_generation(prompt, width=1024, height=1024):
    response = requests.post(
        "http://localhost:8000/generate-image",
        json={
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": 30
        }
    )
    return response.json()

# Example usage
text_result = test_text_generation("Write a creative story about")
print(text_result["generated_text"])

image_result = test_image_generation("a beautiful sunset over mountains")
print(f"Image generated with seed: {image_result['seed']}")
```

### 2. Performance Monitoring

```python
# Monitor performance during generation
import time
import matplotlib.pyplot as plt

def benchmark_generation(prompts, image_sizes):
    results = []
    
    for prompt in prompts:
        for size in image_sizes:
            start_time = time.time()
            
            # Generate image
            result = test_image_generation(prompt, size, size)
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            results.append({
                "prompt": prompt[:30] + "...",
                "size": f"{size}x{size}",
                "time": generation_time,
                "gpu_memory": result.get("gpu_memory_used", "N/A")
            })
            
            print(f"Generated {size}x{size} in {generation_time:.2f}s")
    
    return results

# Run benchmark
prompts = [
    "a photorealistic portrait of a woman",
    "abstract digital art with vibrant colors",
    "a futuristic cityscape at night"
]
sizes = [512, 768, 1024]

benchmark_results = benchmark_generation(prompts, sizes)
```

### 3. Model Comparison

```python
# Compare different models or settings
def compare_models():
    test_prompt = "a beautiful landscape painting"
    
    # Test different inference steps
    steps = [10, 20, 30, 50]
    
    for step in steps:
        start_time = time.time()
        result = requests.post(
            "http://localhost:8000/generate-image",
            json={
                "prompt": test_prompt,
                "num_inference_steps": step,
                "width": 768,
                "height": 768
            }
        ).json()
        
        end_time = time.time()
        print(f"Steps: {step}, Time: {end_time - start_time:.2f}s, Memory: {result.get('gpu_memory_used')}")

compare_models()
```

---

## 🎨 Advanced Features for Testing

### 1. Batch Generation

```python
# Generate multiple images in batch
def batch_generate_images(prompts, batch_size=4):
    results = []
    
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        batch_results = []
        
        for prompt in batch:
            result = test_image_generation(prompt)
            batch_results.append(result)
        
        results.extend(batch_results)
        
        # Cleanup between batches
        requests.post("http://localhost:8000/cleanup")
    
    return results

# Example usage
prompt_list = [
    "a serene lake at dawn",
    "cyberpunk street scene",
    "abstract geometric patterns",
    "vintage car in a garage"
]

batch_results = batch_generate_images(prompt_list)
```

### 2. Interactive UI in Jupyter

```python
# Create interactive widgets for testing
import ipywidgets as widgets
from IPython.display import display, HTML, Image as IPImage

def create_interactive_ui():
    # Text input
    prompt_input = widgets.Textarea(
        value="a beautiful sunset",
        placeholder="Enter your prompt here",
        description="Prompt:",
        layout=widgets.Layout(width='500px', height='100px')
    )
    
    # Size slider
    size_slider = widgets.IntSlider(
        value=768,
        min=512,
        max=1536,
        step=256,
        description="Size:"
    )
    
    # Steps slider
    steps_slider = widgets.IntSlider(
        value=30,
        min=10,
        max=50,
        step=5,
        description="Steps:"
    )
    
    # Generate button
    generate_button = widgets.Button(
        description="Generate Image",
        button_style='success'
    )
    
    # Output area
    output = widgets.Output()
    
    def on_generate_click(b):
        with output:
            output.clear_output()
            print("Generating image...")
            
            result = test_image_generation(
                prompt_input.value,
                size_slider.value,
                size_slider.value
            )
            
            if 'image' in result:
                # Decode and display image
                import base64
                from io import BytesIO
                
                image_data = base64.b64decode(result['image'])
                display(IPImage(data=image_data))
                print(f"Generated with seed: {result['seed']}")
                print(f"GPU Memory: {result.get('gpu_memory_used', 'N/A')}")
            else:
                print(f"Error: {result}")
    
    generate_button.on_click(on_generate_click)
    
    # Layout
    ui = widgets.VBox([
        prompt_input,
        size_slider,
        steps_slider,
        generate_button,
        output
    ])
    
    display(ui)

# Create the interactive UI
create_interactive_ui()
```

---

## 📊 Performance Optimization Tips

### 1. Memory Management

```python
# Optimal memory settings for unlimited VRAM
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Enable compilation for faster inference (PyTorch 2.0+)
if hasattr(torch, 'compile'):
    # Compile models for faster inference
    pass  # Add compilation when models are loaded
```

### 2. Model Caching

```python
# Pre-load models to avoid loading delays
def preload_all_models():
    print("Pre-loading all models...")
    load_text_model()
    load_image_model()
    print("✅ All models loaded and ready!")

# Run this once at startup
preload_all_models()
```

### 3. Quality Settings

```python
# High-quality generation settings for testing
HIGH_QUALITY_SETTINGS = {
    "num_inference_steps": 50,
    "guidance_scale": 8.0,
    "width": 1024,
    "height": 1024
}

FAST_SETTINGS = {
    "num_inference_steps": 15,
    "guidance_scale": 7.0,
    "width": 512,
    "height": 512
}
```

---

## 🚀 Next Steps

1. **Run the setup code** in your Jupyter notebook
2. **Start the server** with `start_server()`
3. **Test the API** using the provided functions
4. **Use the interactive UI** for easy testing
5. **Monitor performance** with the benchmark tools

Your unlimited GPU/VRAM setup is perfect for:
- **Model experimentation**: Try different models and settings
- **Quality testing**: Generate high-resolution images
- **Performance benchmarking**: Compare different configurations
- **Batch processing**: Generate multiple items efficiently

Enjoy your high-performance testing environment! 🎯