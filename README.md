# TD Hardware for Signal Processing

This repository contains my work for the course **Hardware for Signal Processing**.

## Report
The full report (answers, explanations and results) is available in the PDF:
- **CR_Hardwar_HSAINI_REZZOUQA.pdf**

## Project structure
- `wallet_cpp/` : C++ multi-threading wallet (sequential, threads, mutex, virtual wallet)
- `spd_pytorch/` : SPD matrix operations in PyTorch + performance benchmarks
- `lab4_image/` and `flops_analysis/` : FLOPs and performance analysis based on TIV Lab 4

## How to run

### C++ wallet
```bash
cd wallet_cpp
cmake -S . -B build
cmake --build build -j
./build/wallet_demo --mode sequential
./build/wallet_demo --mode threaded
./build/wallet_demo --mode mutex
./build/wallet_demo --mode virtual

### PyTorch SPD operations
cd spd_pytorch
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python benchmark.py

