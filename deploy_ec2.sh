#!/bin/bash

# NSFW Playground - EC2 Deployment Script
# Run this script on a fresh Ubuntu 22.04 EC2 instance

set -e  # Exit on any error

echo "🚀 NSFW Playground - EC2 Deployment Starting..."
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    print_error "Please don't run this script as root. Run as ubuntu user."
    exit 1
fi

# Update system
print_status "Updating system packages..."
sudo apt update && sudo apt upgrade -y
print_success "System updated successfully"

# Install essential packages
print_status "Installing essential packages..."
sudo apt install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    git \
    nginx \
    htop \
    curl \
    wget \
    unzip \
    build-essential
print_success "Essential packages installed"

# Create project directory
print_status "Setting up project directory..."
cd ~
if [ -d "nsfw-playground" ]; then
    print_warning "Project directory already exists. Backing up..."
    mv nsfw-playground nsfw-playground-backup-$(date +%Y%m%d-%H%M%S)
fi

# Clone repository (you'll need to replace with your actual repo URL)
print_status "Cloning repository..."
if [ -z "$REPO_URL" ]; then
    print_warning "REPO_URL not set. Please clone manually:"
    echo "git clone https://github.com/your-username/nsfw-playground.git"
    mkdir -p nsfw-playground
    cd nsfw-playground
else
    git clone "$REPO_URL" nsfw-playground
    cd nsfw-playground
fi

# Create virtual environment
print_status "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate
print_success "Virtual environment created"

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
print_status "Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    print_success "Python dependencies installed"
else
    print_warning "requirements.txt not found. Installing basic dependencies..."
    pip install fastapi uvicorn transformers torch torchvision torchaudio \
                diffusers accelerate xformers Pillow pydantic python-multipart \
                python-dotenv requests safetensors invisible_watermark sentencepiece
fi

# Create environment file template
print_status "Creating environment configuration..."
cat > .env << EOF
# RunPod Configuration
RUNPOD_API_KEY=your_runpod_api_key_here
RUNPOD_ENDPOINT_ID=your_runpod_endpoint_id_here

# API Configuration
API_KEY=demo

# Model Configuration
TEXT_MODEL=microsoft/DialoGPT-medium
IMAGE_MODEL=stabilityai/stable-diffusion-xl-base-1.0

# Server Configuration
HOST=0.0.0.0
PORT=8000
EOF

chmod 600 .env
print_success "Environment file created (.env)"

# Create systemd service
print_status "Creating systemd service..."
sudo tee /etc/systemd/system/nsfw-playground.service > /dev/null << EOF
[Unit]
Description=NSFW Playground API Server
After=network.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/nsfw-playground
Environment=PATH=/home/ubuntu/nsfw-playground/venv/bin
EnvironmentFile=/home/ubuntu/nsfw-playground/.env
ExecStart=/home/ubuntu/nsfw-playground/venv/bin/python text_generation_api.py
Restart=always
RestartSec=3
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

print_success "Systemd service created"

# Configure Nginx
print_status "Configuring Nginx..."
sudo tee /etc/nginx/sites-available/nsfw-playground > /dev/null << EOF
server {
    listen 80;
    server_name _;
    
    client_max_body_size 50M;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 300;
        proxy_connect_timeout 300;
        proxy_send_timeout 300;
    }
    
    # Health check endpoint
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
EOF

# Enable Nginx site
sudo ln -sf /etc/nginx/sites-available/nsfw-playground /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Test Nginx configuration
if sudo nginx -t; then
    print_success "Nginx configuration is valid"
    sudo systemctl restart nginx
    sudo systemctl enable nginx
else
    print_error "Nginx configuration is invalid"
    exit 1
fi

# Configure firewall
print_status "Configuring firewall..."
sudo ufw allow ssh
sudo ufw allow 80
sudo ufw allow 443
sudo ufw allow 8000
echo "y" | sudo ufw enable
print_success "Firewall configured"

# Create swap file (helpful for memory-intensive operations)
print_status "Creating swap file..."
if [ ! -f /swapfile ]; then
    sudo fallocate -l 2G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    print_success "2GB swap file created"
else
    print_warning "Swap file already exists"
fi

# Create useful scripts
print_status "Creating utility scripts..."

# Start script
cat > start.sh << 'EOF'
#!/bin/bash
echo "Starting NSFW Playground..."
sudo systemctl start nsfw-playground
sudo systemctl status nsfw-playground --no-pager
EOF
chmod +x start.sh

# Stop script
cat > stop.sh << 'EOF'
#!/bin/bash
echo "Stopping NSFW Playground..."
sudo systemctl stop nsfw-playground
EOF
chmod +x stop.sh

# Status script
cat > status.sh << 'EOF'
#!/bin/bash
echo "=== Service Status ==="
sudo systemctl status nsfw-playground --no-pager
echo ""
echo "=== Recent Logs ==="
sudo journalctl -u nsfw-playground --no-pager -n 20
echo ""
echo "=== System Resources ==="
free -h
df -h /
EOF
chmod +x status.sh

# Logs script
cat > logs.sh << 'EOF'
#!/bin/bash
echo "Following NSFW Playground logs (Ctrl+C to exit)..."
sudo journalctl -u nsfw-playground -f
EOF
chmod +x logs.sh

# Test script
cat > test.sh << 'EOF'
#!/bin/bash
echo "Testing NSFW Playground API..."
echo ""
echo "Testing health endpoint..."
curl -s http://localhost/health
echo ""
echo ""
echo "Testing text generation..."
curl -X POST "http://localhost:8000/generate-text" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, this is a test", "max_length": 50}' \
  | python3 -m json.tool
EOF
chmod +x test.sh

print_success "Utility scripts created"

# Enable and start the service
print_status "Starting NSFW Playground service..."
sudo systemctl daemon-reload
sudo systemctl enable nsfw-playground

# Check if the main API file exists
if [ -f "text_generation_api.py" ]; then
    sudo systemctl start nsfw-playground
    sleep 5
    
    if sudo systemctl is-active --quiet nsfw-playground; then
        print_success "NSFW Playground service started successfully"
    else
        print_error "Service failed to start. Check logs with: sudo journalctl -u nsfw-playground"
    fi
else
    print_warning "text_generation_api.py not found. Service not started."
fi

# Final instructions
echo ""
echo "================================================"
print_success "EC2 Deployment Complete!"
echo "================================================"
echo ""
echo "📋 Next Steps:"
echo "1. Edit .env file with your RunPod credentials:"
echo "   nano .env"
echo ""
echo "2. If you haven't cloned your repo yet:"
echo "   git clone https://github.com/your-username/nsfw-playground.git ."
echo ""
echo "3. Start the service:"
echo "   ./start.sh"
echo ""
echo "4. Test the API:"
echo "   ./test.sh"
echo ""
echo "5. View logs:"
echo "   ./logs.sh"
echo ""
echo "6. Check status:"
echo "   ./status.sh"
echo ""
echo "🌐 Access your application:"
echo "   http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)"
echo ""
echo "📊 Monitor resources:"
echo "   htop"
echo "   df -h"
echo "   free -h"
echo ""
print_warning "Remember to configure your RunPod credentials in .env file!"
echo ""