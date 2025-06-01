# ☁️ Cloud Setup Guide: EC2 + RunPod Hybrid

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Browser  │───▶│   EC2 API Server │───▶│ RunPod Serverless│
│                 │    │  (Text + Web UI) │    │ (Image Generation)│
│   index.html    │    │                  │    │                  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
     FREE                    FREE RDT              Pay-per-use
```

## Benefits of This Setup
- **Cost Effective**: Free EC2 + pay-per-use GPU
- **Scalable**: RunPod auto-scales based on demand
- **Reliable**: EC2 for stable API, RunPod for GPU bursts
- **Development Friendly**: Easy to test and iterate

---

## Step 1: Amazon EC2 Setup

### Launch EC2 Instance

1. **Go to AWS Console** → EC2 → Launch Instance

2. **Choose Configuration**:
   ```
   AMI: Ubuntu Server 22.04 LTS
   Instance Type: t3.medium (Free tier eligible)
   Storage: 20GB gp3 (Free tier)
   Security Group: Allow HTTP (80), HTTPS (443), Custom (8000)
   ```

3. **Connect via SSH**:
   ```bash
   ssh -i your-key.pem ubuntu@your-ec2-ip
   ```

### Install Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and Git
sudo apt install python3-pip python3-venv git nginx -y

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Clone your repository
git clone https://github.com/your-username/nsfw-playground.git
cd nsfw-playground

# Install Python dependencies
pip install -r requirements.txt
```

### Configure Environment

```bash
# Create environment file
cat > .env << EOF
RUNPOD_API_KEY=your_runpod_api_key
RUNPOD_ENDPOINT_ID=your_endpoint_id
API_KEY=your_api_key
EOF

# Make it secure
chmod 600 .env
```

---

## Step 2: RunPod Serverless Setup

### Create RunPod Account

1. **Sign up** at [runpod.io](https://runpod.io)
2. **Add credits** (minimum $10 recommended)
3. **Go to Serverless** → Create Endpoint

### Configure Serverless Endpoint

```yaml
Endpoint Configuration:
  Name: nsfw-image-generation
  GPU: RTX 4090 (24GB VRAM)
  Container Image: runpod/pytorch:2.1.0-py3.10-cuda12.1.1-devel-ubuntu22.04
  Container Disk: 20GB
  Environment Variables:
    - MODEL_NAME=stabilityai/stable-diffusion-xl-base-1.0
    - TORCH_DTYPE=float16
```

### Deploy Handler Code

Create `runpod_handler.py`:

```python
import runpod
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import io
import base64
import os

# Global pipeline variable
pipeline = None

def load_pipeline():
    global pipeline
    if pipeline is None:
        model_id = os.getenv("MODEL_NAME", "stabilityai/stable-diffusion-xl-base-1.0")
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        pipeline = pipeline.to("cuda")
        pipeline.enable_model_cpu_offload()
        pipeline.enable_xformers_memory_efficient_attention()
    return pipeline

def generate_image(job):
    try:
        # Get job input
        job_input = job["input"]
        prompt = job_input.get("prompt", "")
        negative_prompt = job_input.get("negative_prompt", "blurry, low quality")
        width = min(job_input.get("width", 512), 1024)
        height = min(job_input.get("height", 512), 1024)
        steps = min(job_input.get("num_inference_steps", 20), 50)
        guidance = job_input.get("guidance_scale", 7.5)
        seed = job_input.get("seed", -1)
        
        # Load pipeline
        pipe = load_pipeline()
        
        # Set seed
        if seed == -1:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        
        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        # Generate image
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator
        ).images[0]
        
        # Convert to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Clean up
        torch.cuda.empty_cache()
        
        return {
            "image": img_base64,
            "seed": seed,
            "prompt": prompt,
            "status": "success"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "status": "error"
        }

# Start the serverless function
runpod.serverless.start({"handler": generate_image})
```

---

## Step 3: Modify API for Hybrid Setup

Update `text_generation_api.py`:

```python
import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import pipeline
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="NSFW Playground - Hybrid Cloud")

# Serve static files
app.mount("/static", StaticFiles(directory="."), name="static")

# Global text pipeline
text_pipeline = None

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

def load_text_pipeline():
    global text_pipeline
    if text_pipeline is None:
        # Use CPU-friendly model for EC2
        model_name = "microsoft/DialoGPT-medium"
        text_pipeline = pipeline(
            "text-generation",
            model=model_name,
            device=-1  # CPU only
        )
    return text_pipeline

@app.get("/")
async def root():
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/generate-text")
async def generate_text(request: TextRequest):
    try:
        pipe = load_text_pipeline()
        
        generated = pipe(
            request.prompt,
            max_length=request.max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=pipe.tokenizer.eos_token_id
        )
        
        return {
            "generated_text": generated[0]["generated_text"],
            "prompt": request.prompt,
            "model": "DialoGPT-medium (CPU)"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-image")
async def generate_image(request: ImageRequest):
    try:
        # Call RunPod serverless endpoint
        runpod_api_key = os.getenv("RUNPOD_API_KEY")
        endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID")
        
        if not runpod_api_key or not endpoint_id:
            raise HTTPException(status_code=500, detail="RunPod credentials not configured")
        
        # Prepare request
        payload = {
            "input": {
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "width": request.width,
                "height": request.height,
                "num_inference_steps": request.num_inference_steps,
                "guidance_scale": request.guidance_scale,
                "seed": request.seed
            }
        }
        
        # Call RunPod API
        response = requests.post(
            f"https://api.runpod.ai/v2/{endpoint_id}/runsync",
            headers={
                "Authorization": f"Bearer {runpod_api_key}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=300  # 5 minutes timeout
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"RunPod API error: {response.text}")
        
        result = response.json()
        
        if result.get("status") == "COMPLETED":
            output = result.get("output", {})
            if output.get("status") == "success":
                return {
                    "image": output["image"],
                    "seed": output["seed"],
                    "prompt": output["prompt"],
                    "model": "SDXL (RunPod Serverless)"
                }
            else:
                raise HTTPException(status_code=500, detail=output.get("error", "Unknown error"))
        else:
            raise HTTPException(status_code=500, detail=f"RunPod job failed: {result.get('status')}")
            
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Image generation timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## Step 4: Configure Nginx (Optional)

```bash
# Create Nginx config
sudo tee /etc/nginx/sites-available/nsfw-playground << EOF
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

# Enable site
sudo ln -s /etc/nginx/sites-available/nsfw-playground /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

---

## Step 5: Create Systemd Service

```bash
# Create service file
sudo tee /etc/systemd/system/nsfw-playground.service << EOF
[Unit]
Description=NSFW Playground API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/nsfw-playground
Environment=PATH=/home/ubuntu/nsfw-playground/venv/bin
ExecStart=/home/ubuntu/nsfw-playground/venv/bin/python text_generation_api.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable nsfw-playground
sudo systemctl start nsfw-playground

# Check status
sudo systemctl status nsfw-playground
```

---

## Cost Monitoring

### AWS CloudWatch
```bash
# Install AWS CLI
sudo apt install awscli -y

# Configure billing alerts
aws cloudwatch put-metric-alarm \
  --alarm-name "EC2-HighCPU" \
  --alarm-description "EC2 High CPU Usage" \
  --metric-name CPUUtilization \
  --namespace AWS/EC2 \
  --statistic Average \
  --period 300 \
  --threshold 80 \
  --comparison-operator GreaterThanThreshold
```

### RunPod Monitoring
- Check dashboard at [runpod.io/console](https://runpod.io/console)
- Set up spending alerts
- Monitor request patterns

---

## Testing the Setup

```bash
# Test text generation
curl -X POST "http://your-ec2-ip:8000/generate-text" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello world", "max_length": 50}'

# Test image generation
curl -X POST "http://your-ec2-ip:8000/generate-image" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a beautiful sunset", "width": 512, "height": 512}'
```

---

## Troubleshooting

### Common Issues

1. **RunPod API Key Issues**:
   ```bash
   # Check environment variables
   echo $RUNPOD_API_KEY
   echo $RUNPOD_ENDPOINT_ID
   ```

2. **EC2 Connection Issues**:
   ```bash
   # Check security groups
   # Ensure port 8000 is open
   sudo ufw allow 8000
   ```

3. **Memory Issues**:
   ```bash
   # Monitor memory usage
   htop
   # Add swap if needed
   sudo fallocate -l 2G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

### Logs
```bash
# Check application logs
sudo journalctl -u nsfw-playground -f

# Check Nginx logs
sudo tail -f /var/log/nginx/error.log
```

---

## Expected Costs

- **EC2 t3.medium**: FREE (with RDT credits)
- **RunPod Serverless**: ~$0.01-0.05 per image
- **Data Transfer**: Minimal for API calls
- **Total Monthly**: $10-50 for moderate usage

This setup gives you a production-ready, cost-effective solution that scales with your usage!