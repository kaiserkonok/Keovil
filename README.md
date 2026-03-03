# Keovil - Private AI Data Agent

A high-performance interface for document query, structured data analysis, and real-time file engineering.

---

## What is Keovil?

Keovil is a private AI Agent that understands everything you own. From PDFs and text files to complex Excel sheets and databases, just ask a question in plain English and get an answer.

### Features
- **Instant Learning:** Upload files and start chatting immediately.
- **Cross-File Reasoning:** Ask questions that span across multiple documents at once.
- **Total Privacy:** Everything stays on your machine. No manual sorting, no one-by-one uploads—just instant reports and analysis.

### Why Use Keovil?
- **Privacy:** Your data never leaves your machine.
- **No Monthly Fees:** Use your own hardware instead of paying for subscriptions.
- **Permanent Knowledge Base:** Builds a long-term "Second Brain" from your data that stays ready even after you close the app.

---

## Step 1: Prepare Your Machine

### System Requirements

Keovil is optimized for modern NVIDIA hardware to ensure speed and accuracy.

- **GPU:** NVIDIA RTX Series (30, 40, or 50 series) with **8GB+ VRAM** recommended.
- **Driver:** **Version 550 or higher** (Required for CUDA 12.4 native acceleration).
- **Software:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows) or Docker Engine (Linux).

### The GPU Bridge (Linux Only)

If you are on Linux, Docker needs the **NVIDIA Container Toolkit** to access your hardware. If you haven't installed it yet, run this:

```bash
# 1. Setup the repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

# 2. Install and Restart
```bash
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## Step 2: Install Ollama & Model

Keovil connects to Ollama running on your host machine. This keeps the image small and lets you manage models yourself.

### Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Pull the Required Model

```bash
ollama pull qwen2.5-coder:7b-instruct
```

### Start Ollama

In a separate terminal, run:
```bash
ollama serve
```

---

## Step 3: Install Keovil

Run this command to install Keovil:

```bash
bash <(curl -s https://kevil.io/Keovil/install/)
```

This will:
1. Download the latest Keovil image
2. Create the necessary directories
3. Register the `keovil` command
4. Launch Keovil automatically

---

## Step 4: Start Using It

1. **Ignition:** Watch the terminal for the live status dashboard.
2. **Access:** Once the dashboard shows **[✔] ALL SYSTEMS OPERATIONAL**, open your browser to: **[http://localhost:5000](http://localhost:5000)**
3. **Adding master key**: When you try to use `Keovil` for the first time, it will ask you for a master key. For getting the master key, you need to **[Register here](https://kevil.io/sync/)** and get the master key from the top right of the home page. You just need to click the code and it would be copied to your clipboard.
4. **Control:** To stop the engine and free up your GPU, press `Ctrl + C`.

### Running Keovil Again

After the first run, simply type:
```bash
keovil
```

---

## Architecture

- **Ollama:** Runs on your host machine (not in Docker)
- **Keovil:** Runs in Docker, connects to host's Ollama via host network
- **Data:** Stored in `~/.keovil_storage` (persists across restarts)
- **GPU:** Shared with host for fast ColBERT embeddings

---

## Troubleshooting

### Ollama Not Found
If you see "Ollama is required but not found", make sure:
1. Ollama is installed: `curl -fsSL https://ollama.com/install.sh | sh`
2. Ollama is running: `ollama serve`
3. Model is pulled: `ollama pull qwen2.5-coder:7b-instruct`

### GPU Not Detected
Ensure NVIDIA Container Toolkit is installed and Docker can access your GPU:
```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```
