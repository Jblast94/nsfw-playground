# 🚀 NSFW Playground Deployment Guide

## Current Situation Analysis

**Your Setup**: Amazon Free RDT + RunPod Serverless Options

**Recommended Strategy**: Cloud-first development with serverless GPU endpoints for cost optimization

---

## 📊 Instance Recommendations by Use Case

### 🆓 Free Tier Testing (Start Here)

#### 1. Kaggle Notebooks (RECOMMENDED FIRST)
- **Cost**: FREE (30 hours/week GPU quota)
- **GPU**: Tesla T4 (16GB VRAM)
- **RAM**: 13GB
- **Storage**: 20GB
- **Perfect for**: Testing both text and image generation
- **Limitations**: Session timeout, no persistent deployment

**Setup Steps**:
1. Go to [kaggle.com/code](https://kaggle.com/code)
2. Create new notebook
3. Enable GPU accelerator
4. Upload your code files
5. Install dependencies: `!pip install -r requirements.txt`
6. Run: `!python text_generation_api.py`

#### 2. Google Colab (Backup Option)
- **Cost**: FREE (limited GPU hours)
- **GPU**: T4 (when available)
- **RAM**: 12GB
- **Good for**: Quick testing, prototyping

---

### 🎯 Recommended Cloud Solutions

#### 1. RunPod Serverless (BEST FOR COST CONTROL)
- **Pay per request**: $0.0002-0.0008 per second
- **Auto-scaling**: Spins up/down automatically
- **GPU Options**: RTX 4090, A100, H100
- **Perfect for**: Development with intermittent usage
- **Setup**: Pre-built containers available

**RunPod Serverless Setup**:
```python
# Create serverless endpoint
# Use RunPod's FastAPI template
# Deploy your text_generation_api.py
# Access via HTTPS endpoint
```

#### 2. Amazon EC2 (FREE RDT)
- **t3.medium**: Free tier eligible (CPU only)
- **g4dn.xlarge**: $0.526/hr (T4 16GB) - for GPU workloads
- **g5.xlarge**: $0.50/hr (A10G 24GB) - better performance
- **Perfect for**: Stable development environment

**EC2 Setup**:
```bash
# Launch Ubuntu 22.04 LTS
# Install CUDA toolkit
sudo apt update
sudo apt install python3-pip git nvidia-driver-535 -y
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run
sudo sh cuda_12.2.0_535.54.03_linux.run
```

#### 3. Vast.ai Spot Instances (BUDGET OPTION)
- **RTX 3090**: $0.15-0.25/hr (24GB VRAM)
- **RTX 4090**: $0.25-0.40/hr (24GB VRAM)
- **Risk**: Can be preempted
- **Good for**: Testing and development

---

### 🔄 Serverless vs Traditional Compute

#### RunPod Serverless Benefits:
- **Cost**: Only pay when generating
- **Scaling**: Automatic based on demand
- **Management**: No server maintenance
- **Cold starts**: 10-30 seconds

#### Traditional Compute Benefits:
- **Latency**: Always warm, instant response
- **Control**: Full server access
- **Debugging**: Easier development workflow
- **Cost**: Predictable hourly rates

---

## 🎯 Recommended Development Strategy

### Phase 1: Initial Development (Free Tier)
1. **Amazon EC2 t3.medium**: Use free RDT for API development
2. **Kaggle Notebooks**: Test models with free GPU quota
3. **Local testing**: Validate API endpoints and UI

### Phase 2: GPU Development
1. **RunPod Serverless**: Deploy image generation endpoint
2. **EC2 g4dn.xlarge**: For integrated testing ($0.526/hr)
3. **Hybrid approach**: CPU API + Serverless GPU

### Phase 3: Production Optimization
1. **RunPod Serverless**: For production image generation
2. **EC2 Auto Scaling**: For API layer
3. **CloudFront CDN**: For static assets

---

## ☁️ Cloud Setup Instructions

### Amazon EC2 Setup (Free RDT)

1. **Launch Instance**:
   ```bash
   # Choose Ubuntu 22.04 LTS
   # t3.medium for CPU-only development
   # g4dn.xlarge for GPU workloads
   ```

2. **Install Dependencies**:
   ```bash
   sudo apt update
   sudo apt install python3-pip git -y
   
   # For GPU instances, install CUDA
   sudo apt install nvidia-driver-535 -y
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
   sudo dpkg -i cuda-keyring_1.0-1_all.deb
   sudo apt update
   sudo apt install cuda-toolkit-12-2 -y
   ```

3. **Deploy Application**:
   ```bash
   git clone <your-repo>
   cd nsfw-playground
   pip3 install -r requirements.txt
   python3 text_generation_api.py
   ```

### RunPod Serverless Setup

1. **Create Serverless Endpoint**:
   - Use RunPod's FastAPI template
   - Upload your `text_generation_api.py`
   - Configure environment variables

2. **API Integration**:
   ```python
   # Call RunPod serverless endpoint
   import requests
   
   response = requests.post(
       "https://api.runpod.ai/v2/your-endpoint-id/runsync",
       headers={"Authorization": f"Bearer {api_key}"},
       json={"input": {"prompt": "your prompt"}}
   )
   ```

---

## 🔧 Configuration for Different Scenarios

### Low VRAM Setup (8-12GB)
```python
# In text_generation_api.py
# Use smaller models or CPU fallback
model_name = "microsoft/DialoGPT-medium"  # Smaller alternative

# Reduce image generation settings
default_width = 512
default_height = 512
default_steps = 15
```

### High VRAM Setup (24GB+)
```python
# Full models with optimizations
pipeline.enable_xformers_memory_efficient_attention()
pipeline.enable_model_cpu_offload()
```

---

## 💰 Cost Optimization Strategies

### 1. Hybrid Architecture
- **API Layer**: EC2 t3.medium (free tier)
- **Text Generation**: CPU-based models on EC2
- **Image Generation**: RunPod serverless (pay per use)

### 2. Smart Resource Usage
- **Development**: Use free tiers (EC2, Kaggle)
- **Testing**: RunPod serverless for GPU tasks
- **Production**: Auto-scaling based on demand

### 3. Model Optimization
- **Quantized models**: Reduce VRAM requirements
- **Model caching**: Keep frequently used models warm
- **Batch processing**: Process multiple requests together

### 4. Monitoring and Alerts
- **AWS CloudWatch**: Monitor EC2 usage
- **RunPod Dashboard**: Track serverless costs
- **Billing alerts**: Set spending limits

---

## 🚨 Troubleshooting Common Issues

### Out of Memory Errors
```python
# Reduce batch size
torch.cuda.empty_cache()

# Use gradient checkpointing
model.gradient_checkpointing_enable()

# Reduce precision
pipeline.to(torch_dtype=torch.float16)
```

### Model Download Issues
```bash
# Pre-download models
huggingface-cli download mlabonne/gemma-3-12b-it-qat-abliterated
huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0
```

### Network/Firewall Issues
```bash
# Open port 8000
# Windows Firewall
netsh advfirewall firewall add rule name="NSFW Playground" dir=in action=allow protocol=TCP localport=8000

# For cloud instances, configure security groups
```

---

## 🚀 Immediate Next Steps

### Quick Start (Today)
1. **Launch EC2 t3.medium**: Use your free RDT credits
2. **Deploy API layer**: Text generation on CPU
3. **Test basic functionality**: Validate endpoints

### GPU Integration (This Week)
1. **Set up RunPod account**: Create serverless endpoint
2. **Deploy image generation**: Use SDXL model
3. **Integrate endpoints**: Connect EC2 API to RunPod GPU

### Production Ready (Next Week)
1. **Optimize performance**: Fine-tune model parameters
2. **Set up monitoring**: CloudWatch + RunPod dashboards
3. **Configure auto-scaling**: Based on usage patterns

**Cost Estimate**: 
- EC2 t3.medium: FREE (with RDT)
- RunPod Serverless: ~$0.01-0.05 per image
- Total monthly: $10-50 for moderate usage