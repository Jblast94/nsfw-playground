# NSFW Playground

A modern web application for generating NSFW content using AI models.

## Features

- **Text Generation**: Uses `mlabonne/gemma-3-12b-it-qat-abliterated` (uncensored model)
- **Image Generation**: NSFW-capable Stable Diffusion XL pipeline
- **Modern UI**: Tabbed interface with real-time generation status
- **Download Support**: Save generated images locally
- **Flexible Deployment**: Docker or native Python setup

## 🚀 Quick Start Options

### Option 1: Free Testing on Kaggle (RECOMMENDED)
1. **Upload to Kaggle Notebooks**:
   - Go to [kaggle.com/code](https://kaggle.com/code)
   - Create new notebook with GPU enabled
   - Upload `kaggle_setup.py`
   - Run: `python kaggle_setup.py`
   - Start: `python kaggle_api.py`

### Option 2: Windows Server (Local)
1. **Prerequisites**: Python 3.9+, CUDA-compatible GPU (12GB+ VRAM)
2. **Quick Setup**:
   ```cmd
   setup_windows.bat
   python text_generation_api.py
   ```
3. **Open**: `http://localhost:8000`

### Option 3: Cloud Deployment
See **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** for detailed cloud options:
- 🆓 **Kaggle**: Free GPU (30hrs/week)
- 💰 **Vast.ai**: $0.15-0.25/hr (RTX 3090/4090)
- 🏢 **AWS/GCP**: Enterprise options

## 💻 System Requirements

### Minimum Requirements:
- **RAM**: 16GB+ (32GB recommended)
- **VRAM**: 12GB+ for image generation
- **Storage**: 50GB+ free space
- **OS**: Windows 10/11, Linux, macOS

### Model Requirements:
- **Text Model**: `mlabonne/gemma-3-12b-it-qat-abliterated` (~24GB)
- **Image Model**: Stable Diffusion XL (~13GB)

## ☁️ Cloud Deployment Recommendations

### For Text + Image Generation:
- **RunPod**: RTX 4090 (24GB VRAM) - $0.34/hr
- **Vast.ai**: RTX 3090/4090 instances - $0.20-0.40/hr
- **AWS**: g5.2xlarge (A10G 24GB) - $1.01/hr
- **Google Cloud**: n1-standard-4 + T4 - $0.35/hr

### For Text Only:
- **Kaggle**: Free 30hrs/week (T4 16GB) - Perfect for testing
- **Google Colab Pro**: $10/month (A100 access)
- **Paperspace**: Gradient instances starting $0.07/hr

### Budget Options:
- **Kaggle Notebooks**: Free tier with GPU quota
- **Google Colab**: Free tier (limited GPU time)
- **Vast.ai**: Spot instances as low as $0.10/hr

## 🔧 Configuration

### Environment Variables:
```bash
# Optional: Set custom model paths
TEXT_MODEL_NAME=mlabonne/gemma-3-12b-it-qat-abliterated
IMAGE_MODEL_NAME=stabilityai/stable-diffusion-xl-base-1.0

# GPU settings
CUDA_VISIBLE_DEVICES=0
```

### Model Customization:
Edit `text_generation_api.py` to change models:
```python
# Text generation model
model_name = "your-preferred-model"

# Image generation model  
pipeline = StableDiffusionXLPipeline.from_pretrained("your-image-model")
```

## 📋 Dependencies

**Core Requirements**:
- `fastapi` - Web framework
- `transformers` - Text generation
- `diffusers` - Image generation
- `torch` - PyTorch backend
- `Pillow` - Image processing

**Full list**: See `requirements.txt`

## 🎯 Usage

1. **Text Generation Tab**:
   - Enter your prompt
   - Adjust max length (10-500 tokens)
   - Click "Generate Text"

2. **Image Generation Tab**:
   - Enter image description
   - Set negative prompts (optional)
   - Configure dimensions, steps, guidance
   - Set seed for reproducibility
   - Click "Generate Image"
   - Download generated images

## 🔒 Security Notes

- This application generates NSFW content
- Ensure compliance with local laws and ToS
- Use appropriate content filtering if deploying publicly
- Keep API endpoints secured

## 🐛 Troubleshooting

**Out of Memory Errors**:
- Reduce image dimensions (512x512 instead of 1024x1024)
- Lower inference steps (10-15 instead of 20+)
- Use CPU fallback for text generation

**Model Loading Issues**:
- Ensure sufficient disk space
- Check internet connection for model downloads
- Verify CUDA installation for GPU acceleration

**Performance Optimization**:
- Use `torch.compile()` for faster inference
- Enable `xformers` for memory efficiency
- Consider model quantization for lower VRAM usage