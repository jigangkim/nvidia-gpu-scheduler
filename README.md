# Manage multiple NVIDIA GPU compute tasks

Supports per GPU compute limits (number of processes, utilization rate, memory usage) on a per-(UNIX)user/worker basis, load-balancing, multiple nodes(machines) and more.

Tested on tensorflow-gpu tasks.

<p align="center">
  <img src="screenshot.png"><br>
</p>

Installation (virtual python environment such as venv/conda is recommended)
```bash
cd /path/to/install
git clone https://github.com/jigangkim/nvidia-gpu-scheduler.git
cd /path/to/install/nvidia-gpu-scheduler

pip install . # standard installation
pip install -e . # editable (develop mode) installation
```

Usage (dummy example)
```bash
cd /path/to/install/nvidia-gpu-scheduler
# Run job server
python example.py --identity scheduler
```
```bash
# Run worker
python example.py --identity worker
```

Usage (OpenAI baselines example)
```bash
cd /path/to/install/nvidia-gpu-scheduler
# Run job server
python example_openaibaselines.py --identity scheduler
```
```bash
# Run worker
python example_openaibaselines.py --identity worker
```