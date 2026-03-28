# Ubuntu Server Setup Guide

Complete setup for running Executive Helper, Home Assistant, and Ollama on a single Ubuntu machine using Docker containers. This is the recommended path for a dedicated server (headless box, NUC, or repurposed desktop with a GPU).

## Prerequisites

| Component | Requirement |
|---|---|
| **OS** | Ubuntu 22.04+ (Server or Desktop) |
| **GPU** | NVIDIA with 8GB+ VRAM (RTX 3060 8GB minimum, RTX 4060/4090 recommended) |
| **RAM** | 16GB+ system RAM |
| **Disk** | 50GB+ free (models, HA config, containers) |
| **Network** | Ethernet recommended for reliability; Wi-Fi works |

## Step 1: Install NVIDIA GPU Drivers

The container toolkit (Step 3) is just a bridge — you need the actual NVIDIA drivers on the host first.

```bash
# Check that your GPU is detected
lspci | grep -i nvidia
# You should see your GPU listed (e.g., "NVIDIA Corporation GA106 [GeForce RTX 3060]")

# Check what driver versions are available
apt-cache search nvidia-driver | grep -E '^nvidia-driver-[0-9]+\s' | sort -t- -k3 -n

# Install the latest non-server driver (check the list above for the newest one)
# As of early 2026, nvidia-driver-570 is a good choice for RTX 30/40 series
sudo apt-get update
sudo apt-get install -y nvidia-driver-570

# REBOOT REQUIRED — the kernel module won't load until you reboot
sudo reboot
```

After reboot, verify the driver is loaded:

```bash
nvidia-smi
# You should see your GPU name, driver version, and CUDA version
```

> **If `nvidia-smi` fails after reboot**: Check `dkms status` to see if the kernel module built correctly. Secure Boot can also block unsigned kernel modules — if you have Secure Boot enabled, you may need to enroll the MOK key or disable Secure Boot in BIOS.

## Step 2: Install Docker Engine

> **Important**: Docker Desktop is not supported by Home Assistant. You need Docker Engine.

```bash
# Remove any old Docker packages
sudo apt-get remove -y docker docker-engine docker.io containerd runc 2>/dev/null

# Install prerequisites
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

# Add Docker's official GPG key and repository
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine + Compose plugin
# ⚠️  You MUST run apt-get update here — the Docker repo was just added above
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Allow your user to run Docker without sudo
sudo usermod -aG docker $USER
newgrp docker

# Verify
docker --version
docker compose version
```

## Step 3: Install NVIDIA Container Toolkit

This lets Docker containers access your GPU (required for Ollama and the EF model). The host GPU drivers must already be installed (Step 1).

```bash
# Add NVIDIA container toolkit repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use the NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU is accessible from Docker
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

You should see your GPU listed. If not, check that your NVIDIA drivers are installed (Step 1) and that you rebooted after installing them.

## Step 4: Create Project Directory

```bash
mkdir -p ~/executive-helper-server
cd ~/executive-helper-server

# Create data directories
mkdir -p ha-config ollama-data eh-data
```

## Step 5: Set Up Docker Compose

```bash
cd ~/executive-helper-server

# Download compose file and env template from the repo
curl -fLO https://raw.githubusercontent.com/sylvainsf/executive-helper/main/deploy/compose.yaml
curl -fLo .env https://raw.githubusercontent.com/sylvainsf/executive-helper/main/deploy/.env.example

# Edit .env with your timezone
nano .env
```

## Step 6: Start Home Assistant and Ollama

Start the infrastructure first (skip Executive Helper for now — it needs the fine-tuned model and an HA token).

```bash
# Start HA and Ollama only
docker compose up -d homeassistant ollama

# Watch the logs to make sure everything boots
docker compose logs -f
```

Wait until both containers are healthy:

```bash
docker compose ps
# Both should show "Up" / "healthy"
```

Home Assistant will be available at `http://<your-server-ip>:8123`. Complete the onboarding wizard (create account, set location, etc.).

## Step 7: Pull Models into Ollama

```bash
# Pull the Home LLM model (for HA device control via voice)
docker exec ollama ollama pull hf.co/acon96/Home-Llama-3.2-3B

# Pull the Phi-4-mini base model (will be replaced by your fine-tune later)
docker exec ollama ollama pull phi4-mini:latest

# Verify both are loaded
docker exec ollama ollama list
```

## Step 8: Install Home LLM in Home Assistant

1. Open Home Assistant at `http://<your-server-ip>:8123`
2. Install HACS if you haven't already:
   - Follow the [HACS installation guide](https://hacs.xyz/docs/use/download/download/)
3. Install Home LLM via HACS:
   - Go to **HACS → Integrations → search "Local LLM"**
   - Or use this button: [![Open HACS](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?category=Integration&repository=home-llm&owner=acon96)
4. **Restart Home Assistant** (Settings → System → Restart)
5. After restart, add the integration:
   - Go to **Settings → Devices & Services → Add Integration → Local LLM**
   - Select **Ollama API** as the backend
   - IP Address: `localhost` (or your server IP if HA is on a different host)
   - Port: `11434`
   - Use HTTPS: unchecked
   - API Key: leave blank
   - Click **Submit**
6. Add a conversation agent:
   - Under the Ollama service, click **Add conversation agent**
   - Model: `hf.co/acon96/Home-Llama-3.2-3B`
   - Select the **Assist** API
   - Enable **in-context learning (ICL) examples**
   - Click **Submit**

## Step 9: Create HA Access Token for Executive Helper

1. In Home Assistant, click your **profile** (bottom-left of sidebar)
2. Scroll to **Long-Lived Access Tokens**
3. Click **Create Token**, name it `executive-helper`
4. Copy the token
5. Paste it into your `.env` file:

```bash
cd ~/executive-helper-server
nano .env
# Replace paste_your_token_here with your actual token
```

## Step 10: Deploy Executive Helper

```bash
cd ~/executive-helper-server

# Clone the project (needed for the Dockerfile and prompts)
git clone https://github.com/sylvainsf/executive-helper.git executive-helper

# If you have a fine-tuned model GGUF, load it into Ollama:
# docker exec ollama ollama create executive-helper-ef -f- < executive-helper/models/Modelfile.executive-helper-ef

# If you don't have a fine-tuned model yet, update .env to use the base model:
# sed -i 's/EH_EF_MODEL=.*/EH_EF_MODEL=phi4-mini:latest/' .env

# Start everything
docker compose up -d

# Verify all three services are running
docker compose ps
```

## Step 11: Install Executive Helper Integration in HA

1. In HACS, go to **Integrations → ⋮ menu → Custom Repositories**
2. Add the Executive Helper repo URL, category **Integration**
3. Find "Executive Helper" and install
4. **Restart Home Assistant**
5. Go to **Settings → Devices & Services → Add Integration → Executive Helper**
6. Enter:
   - Host: `localhost`
   - Port: `8000`
7. Click **Submit** — it will verify connectivity to the backend

## Verifying the Setup

```bash
# Check all containers are running
docker compose ps

# Check Ollama health and loaded models
docker exec ollama ollama list

# Check Executive Helper backend
curl http://localhost:8000/health
# Expected: {"status": "ok", "ollama": true}

# Check Home Assistant can reach Ollama
# (in HA, go to Settings → Devices & Services → Local LLM → the Ollama service
#  should show as connected)
```

In Home Assistant, you should see:
- **EH Backend Connected** binary sensor → `on`
- **Active Intents** sensor → `0`
- **Local LLM** integration → connected with the Home-Llama model

## Common Operations

### Updating containers

```bash
cd ~/executive-helper-server

# Pull latest images
docker compose pull

# Recreate with new images
docker compose up -d
```

### Viewing logs

```bash
# All services
docker compose logs -f

# Just one service
docker compose logs -f executive-helper
docker compose logs -f ollama
docker compose logs -f homeassistant
```

### Loading a new fine-tuned model

After fine-tuning on your training machine (see the main [README](../README.md#fine-tuning--deployment)):

```bash
# Copy the GGUF and Modelfile to the server
scp models/executive-helper-ef.Q4_K_M.gguf user@server:~/executive-helper-server/executive-helper/models/
scp models/Modelfile.executive-helper-ef user@server:~/executive-helper-server/executive-helper/models/

# Load into Ollama
docker exec ollama ollama create executive-helper-ef \
  -f /path/to/Modelfile.executive-helper-ef

# Restart Executive Helper to pick up the new model
docker compose restart executive-helper
```

### Backing up

```bash
# Stop services first for a clean backup
docker compose down

# Back up all data
tar czf executive-helper-backup-$(date +%Y%m%d).tar.gz \
  ha-config/ eh-data/ ollama-data/ .env compose.yaml

# Restart
docker compose up -d
```

### Resetting

```bash
# Stop everything
docker compose down

# Remove all data (DESTRUCTIVE — removes HA config, models, journal)
# rm -rf ha-config/ eh-data/ ollama-data/

# Start fresh
docker compose up -d
```

## Network Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│  Ubuntu Server                                                   │
│                                                                   │
│  ┌─────────────────────┐  ┌──────────────────────────────────┐  │
│  │ Docker: Ollama      │  │ Docker: Home Assistant            │  │
│  │ :11434 (GPU)        │  │ :8123                             │  │
│  │                     │  │                                    │  │
│  │ • Home-Llama-3.2-3B │  │ • Home LLM integration            │  │
│  │ • executive-helper- │  │ • Executive Helper integration    │  │
│  │   ef (fine-tuned)   │  │ • Automations, entities, UI       │  │
│  └────────┬────────────┘  └──────────┬───────────────────────┘  │
│           │                          │                           │
│  ┌────────┴──────────────────────────┴───────────────────────┐  │
│  │ Docker: Executive Helper Backend                           │  │
│  │ :8000                                                       │  │
│  │                                                              │  │
│  │ • Audio pipeline (WebSocket ← ReSpeaker nodes)              │  │
│  │ • Whisper ASR + speaker ID                                  │  │
│  │ • Intent tracker                                             │  │
│  │ • Decision journal (SQLite)                                 │  │
│  │ • Kokoro TTS                                                 │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              ▲                                    │
└──────────────────────────────┼────────────────────────────────────┘
                               │ WebSocket (16kHz PCM)
                    ┌──────────┴──────────┐
                    │ Per-Room Nodes       │
                    │ (ReSpeaker Lite +    │
                    │  speaker)            │
                    └─────────────────────┘
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `apt-get install docker-ce` says "package not available" | You missed the `apt-get update` after adding the Docker repo. The repo was added but apt doesn't know about it yet. Run `sudo apt-get update` then retry the install |
| `nvidia-smi` works on host but `docker run --gpus all` fails with `libnvidia-ml.so.1: cannot open shared object file` | NVIDIA drivers aren't installed on the host. Docker GPU passthrough requires host drivers. Install them (Step 1), reboot, then retry |
| `nvidia-smi` says "command not found" after installing nvidia-driver | Reboot required. The kernel module only loads after a reboot: `sudo reboot` |
| `nvidia-smi` fails after reboot with "driver/library mismatch" | Kernel was updated but the NVIDIA module wasn't rebuilt. Run `sudo apt-get install --reinstall nvidia-driver-570` and reboot again |
| Home Assistant can't connect to Ollama | Both use `network_mode: host`, so `localhost:11434` should work. Check `docker compose logs ollama` for errors |
| Executive Helper "backend disconnected" in HA | Check `curl http://localhost:8000/health`. If it fails, check `docker compose logs executive-helper` |
| Ollama OOM (out of memory) on 8GB GPU | Only one model loads at a time. If both Home LLM and EF model are loaded, Ollama swaps. This is normal but slow. Consider 12GB+ GPU |
| Home Assistant boot loop | Check `docker compose logs homeassistant`. Common cause: corrupt config. Try `docker compose down` then `docker compose up -d homeassistant` |
| Port 8123 already in use | Another HA instance or service is using it. Check with `sudo lsof -i :8123` |
| HACS not appearing after install | Clear browser cache and hard-refresh. HACS requires a restart after initial download |
| `dpkg` lock held by another process | Another apt/dpkg operation is running (often unattended-upgrades). Wait for it: `while sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do sleep 5; done` |
