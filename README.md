🛰️ GeoChat: ISRO Integrated Pipeline
Grounded Large Vision-Language Model for Remote Sensing

Original Authors: Kartik Kuckreja*, Muhammad Sohail Danish*, Muzammal Naseer, Abhijit Das, Salman Khan and Fahad Khan (MBZUAI, BITS Pilani, ANU, Linkoping University)

🌍 Overview
GeoChat is the first grounded Large Vision Language Model specifically tailored to Remote Sensing (RS) scenarios. Unlike general-domain models, GeoChat excels in handling high-resolution RS imagery (ISRO/Satellite data), employing region-level reasoning for comprehensive scene interpretation.

Key Capabilities:

Image & Region Captioning

Visual Question Answering (VQA)

Scene Classification

Visually Grounded Conversations (Identifying coordinates of objects)

🚀 Installation & Setup Guide
This project has been containerized for secure, offline, and reproducible deployment. Follow these steps strictly to run the pipeline on Windows/Linux systems.

📋 Prerequisites
Docker Desktop installed.

NVIDIA GPU (Minimum 6GB VRAM for 4-bit quantization).

Git LFS installed (git lfs install).

1️⃣ Step 1: Windows Memory Optimization (CRITICAL)
If you are on Linux, skip to Step 2.

By default, Docker on Windows (WSL2) does not allocate enough Swap memory to load the 13GB model, causing "Killed" errors. You must run this fix once.

Open PowerShell as Administrator.

Copy and paste the following block to configure 8GB RAM + 20GB Disk Swap:

PowerShell

# Define path to user config
$ConfigPath = "$env:USERPROFILE\.wslconfig"

# Settings: 8GB RAM for Docker, 20GB Swap (Disk)
# Note: If you have >32GB RAM, you can increase 'memory' to 16GB
$Content = "[wsl2]`nmemory=8GB`nswap=20GB`nlocalhostForwarding=true"

# Force create file
Set-Content -Path $ConfigPath -Value $Content -Force

Write-Host "Success! Config created at: $ConfigPath" -ForegroundColor Green
Write-Host "ACTION REQUIRED: Restart Docker Desktop now." -ForegroundColor Yellow
Restart Docker Desktop:

Right-click the Docker Whale icon in the system tray -> Quit Docker Desktop.

Open PowerShell and run: wsl --shutdown

Wait 10 seconds, then open Docker Desktop again.

2️⃣ Step 2: Download Model Weights
The Docker image contains the code, but not the model weights (13GB). You must download them into the repository folder.

Navigate to the GeoChat directory:

Bash

cd geo-intellix/language_engine/GeoChat
Download the weights from Hugging Face:

Bash

git lfs install
git clone https://huggingface.co/MBZUAI/geochat-7B
Ensure you see a folder named geochat-7B containing .bin files.

3️⃣ Step 3: Build the Environment
We use a custom Dockerfile that fixes numpy conflicts, cv2 errors, and installs all dependencies automatically.

Run this command from geo-intellix/language_engine/GeoChat:

Bash

docker build -t geochat-prod .
(This may take 5-10 minutes the first time)

4️⃣ Step 4: Run the Inference Server
This command launches the secure container, mounts your local weights, and starts the web interface.

For Windows (PowerShell):

PowerShell

docker run --gpus all `
  -p 7860:7860 `
  -v ${PWD}/geochat-7B:/app/GeoChat/geochat-7B `
  --name geochat_runner `
  --rm -it `
  geochat-prod
For Linux (Bash):

Bash

docker run --gpus all \
  -p 7860:7860 \
  -v $(pwd)/geochat-7B:/app/GeoChat/geochat-7B \
  --name geochat_runner \
  --rm -it \
  geochat-prod
Note on Low VRAM (RTX 3050/4050): The Dockerfile is configured to automatically load the model in 4-bit mode to fit on 6GB GPUs. If you have A100s, you can edit the CMD in the Dockerfile to remove --load-4bit.

5️⃣ Usage
Wait for the terminal to show: Running on local URL: http://0.0.0.0:7860 (The first load may take 1-2 minutes as it moves data to Swap memory).

Open your browser and go to: 👉 http://localhost:7860

To Stop: Press Ctrl+C in the terminal.

### 📂 Project Structure

```text
geo-intellix/
├── backend/                # FastAPI Server (Integration Logic)
├── language_engine/
│   └── GeoChat/            # The core AI Model
│       ├── Dockerfile      # Production Build File
│       ├── geochat-7B/     # Model Weights (Git Ignored)
│       └── geochat_demo.py # Inference Script
├── preprocessing/          # Satellite Tiling Scripts
└── infrastructure/         # Deployment Configs