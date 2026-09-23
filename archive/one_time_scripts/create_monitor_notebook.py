"""
Script to generate Monitor_Training.ipynb.
Creates an interactive dashboard notebook for monitoring offline training.
"""

import json
import os

def create_monitor_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (.venv)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.9.18"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    def add_cell(cell_type, source):
        if isinstance(source, str):
            lines = [line + "\n" for line in source.split("\n")]
            if lines and lines[-1] == "\n":
                lines = lines[:-1]
            if lines:
                lines[-1] = lines[-1].rstrip("\n")
        else:
            lines = source

        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": lines
        }
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        nb["cells"].append(cell)

    # Header Markdown
    add_cell("markdown", """# 📊 BANGHTR-X v2: Real-Time Offline Training Monitor
### Non-Intrusive Dashboard for Shared Research Server
Use this notebook anytime you connect online to monitor your background training run without interrupting other researchers or altering server state.

---
""")

    # Cell 1: Process & GPU Diagnostic
    add_cell("code", """# Cell 1: Process & GPU Resource Diagnostic
import os
import subprocess
import time
import json
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(".")
PID_FILE = os.path.join(PROJECT_ROOT, "logs", "train.pid")
STATUS_FILE = os.path.join(PROJECT_ROOT, "logs", "training_status.json")

print("=" * 65)
print("🔍 SYSTEM & PROCESS STATUS")
print("=" * 65)

# Check if training process is alive
is_running = False
pid = None
if os.path.exists(PID_FILE):
    with open(PID_FILE) as f:
        try:
            pid = int(f.read().strip())
            # Safe read-only process check
            result = subprocess.run(["ps", "-p", str(pid), "-o", "pid,vsz,rss,%cpu,%mem,time,cmd"], capture_output=True, text=True)
            if result.returncode == 0:
                is_running = True
                print(f"🟢 Training Status : ACTIVE (PID {pid})")
                print("\\nProcess Resource Allocation:")
                print(result.stdout.strip())
            else:
                print(f"⚪ Training Status : IDLE or FINISHED (PID {pid} not currently active)")
        except Exception as e:
            print(f"Could not read PID: {e}")
else:
    print("⚪ Training Status : Not yet launched (no logs/train.pid found)")

print("\\n" + "-" * 65)
print("🚀 GPU STATUS (nvidia-smi):")
try:
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu", "--format=csv,noheader"], capture_output=True, text=True)
    if smi.returncode == 0:
        parts = [p.strip() for p in smi.stdout.strip().split(",")]
        if len(parts) >= 5:
            print(f"   Device     : {parts[0]}")
            print(f"   VRAM Usage : {parts[1]} / {parts[2]}")
            print(f"   GPU Util   : {parts[3]}")
            print(f"   GPU Temp   : {parts[4]}")
except Exception as e:
    print(f"   nvidia-smi query unavailable: {e}")
print("=" * 65)
""")

    # Cell 2: Training Status Summary Table
    add_cell("code", """# Cell 2: Training Metric Progress & Epoch Table
if os.path.exists(STATUS_FILE):
    with open(STATUS_FILE, "r", encoding="utf-8") as f:
        status = json.load(f)
        
    print("=" * 65)
    print(f"🎯 CURRENT STAGE : {status.get('current_stage', 'Unknown')} (Stage {status.get('stage_num', 0)}/{status.get('total_stages', 4)})")
    print(f"⏱️  Started At   : {status.get('start_time', 'N/A')} | Last Updated: {status.get('timestamp', 'N/A')}")
    print(f"🏆 Best Supervised CER : {status.get('best_supervised_cer', 'N/A')}")
    print(f"🎮 Best RL-Refined CER : {status.get('best_rl_cer', 'N/A')}")
    print(f"📦 Production Export   : {status.get('production_model_path', 'Pending...')}")
    print("=" * 65)
    
    # Display Stage 2 Epoch History Table
    s2 = status.get("stage2", {})
    if s2.get("train_loss"):
        epochs_done = len(s2["train_loss"])
        df_history = pd.DataFrame({
            "Epoch": range(1, epochs_done + 1),
            "Train Loss": [f"{l:.4f}" for l in s2["train_loss"]],
            "Val CER (%)": [f"{c * 100:.2f}%" for c in s2["val_cer"]],
            "Val WER (%)": [f"{w * 100:.2f}%" for w in s2["val_wer"]]
        })
        print(f"\\n📈 Supervised Line HTR History ({epochs_done}/{s2.get('epochs', 25)} Epochs):")
        display(df_history.tail(10))
    else:
        print("\\nℹ️ Stage 2 training has not recorded epochs yet.")
else:
    print("ℹ️ No training_status.json file found yet. Has the training started?")
""")

    # Cell 3: Live Convergence Plots
    add_cell("code", """# Cell 3: Live Training Convergence Curves
if os.path.exists(STATUS_FILE):
    with open(STATUS_FILE, "r", encoding="utf-8") as f:
        status = json.load(f)
        
    s2 = status.get("stage2", {})
    if s2.get("train_loss") and len(s2["train_loss"]) > 0:
        epochs = list(range(1, len(s2["train_loss"]) + 1))
        train_loss = s2["train_loss"]
        val_cer = [c * 100 for c in s2["val_cer"]]
        val_wer = [w * 100 for w in s2["val_wer"]]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. Loss Curve
        axes[0].plot(epochs, train_loss, "o-", color="#1f77b4", linewidth=2, label="Train Hybrid Loss")
        axes[0].set_xlabel("Epoch", fontsize=11)
        axes[0].set_ylabel("Loss", fontsize=11)
        axes[0].set_title("Hybrid CTC + Attention Loss", fontsize=12, fontweight="bold")
        axes[0].grid(True, linestyle="--", alpha=0.6)
        axes[0].legend()
        
        # 2. CER Curve
        axes[1].plot(epochs, val_cer, "s-", color="#d62728", linewidth=2, label="Validation CER (%)")
        best_cer = min(val_cer)
        best_epoch = epochs[val_cer.index(best_cer)]
        axes[1].axhline(best_cer, color="gray", linestyle=":", label=f"Best: {best_cer:.2f}% (Ep {best_epoch})")
        axes[1].set_xlabel("Epoch", fontsize=11)
        axes[1].set_ylabel("Character Error Rate (%)", fontsize=11)
        axes[1].set_title("Validation CER Progress", fontsize=12, fontweight="bold")
        axes[1].grid(True, linestyle="--", alpha=0.6)
        axes[1].legend()
        
        # 3. WER Curve
        axes[2].plot(epochs, val_wer, "^-", color="#2ca02c", linewidth=2, label="Validation WER (%)")
        axes[2].set_xlabel("Epoch", fontsize=11)
        axes[2].set_ylabel("Word Error Rate (%)", fontsize=11)
        axes[2].set_title("Validation WER Progress", fontsize=12, fontweight="bold")
        axes[2].grid(True, linestyle="--", alpha=0.6)
        axes[2].legend()
        
        plt.suptitle(f"BANGHTR-X v2 Live Training Progress (Epoch {len(epochs)})", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.show()
    else:
        print("ℹ️ Not enough data points to plot Stage 2 curves yet.")
""")

    # Cell 4: Live Log Stream
    add_cell("code", """# Cell 4: Recent Execution Log Stream (Last 25 lines)
LOG_FILE = os.path.join(PROJECT_ROOT, "logs", "train_progress.log")
if os.path.exists(LOG_FILE):
    print("📜 RECENT TRAINING LOGS:")
    print("-" * 75)
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines[-25:]:
            print(line.rstrip())
    print("-" * 75)
else:
    print("ℹ️ Log file logs/train_progress.log not found yet.")
""")

    # Cell 5: Model Artifact & Export Verification
    add_cell("code", """# Cell 5: Saved Model Artifacts & Export Bundle Inspection
import torch

print("=" * 65)
print("📦 SAVED MODEL ARTIFACTS VERIFICATION")
print("=" * 65)

model_files = [
    ("Stage 1 Stem", "checkpoints/stage1_convnext_stem_best.pt"),
    ("Stage 2 Supervised Best", "checkpoints/best_banghtr_x_v2.pt"),
    ("Stage 3 RL Best", "checkpoints/best_banghtr_x_v2_rl.pt"),
    ("Stage 4 Production Export", "exports/banghtr_x_production.pt")
]

for label, rel_path in model_files:
    full_path = os.path.join(PROJECT_ROOT, rel_path)
    if os.path.exists(full_path):
        sz_mb = os.path.getsize(full_path) / (1024 * 1024)
        mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(full_path)))
        print(f"✅ {label:24s} : {rel_path}")
        print(f"   Size: {sz_mb:.2f} MB | Modified: {mtime}")
        
        # If production export, inspect metadata
        if "production" in rel_path:
            try:
                bundle = torch.load(full_path, map_location="cpu")
                print(f"   👉 Architecture : {bundle.get('model_architecture')}")
                print(f"   👉 Vocab Size   : {len(bundle.get('tokenizer_vocab', []))} tokens")
                print(f"   👉 Saved CER    : {bundle.get('final_cer')}")
                print(f"   👉 Export Time  : {bundle.get('export_time')}")
                print(f"   👉 Input Spec   : {bundle.get('input_spec')}")
            except Exception as e:
                print(f"   (Could not parse bundle details: {e})")
        print()
    else:
        print(f"⏳ {label:24s} : Pending generation ({rel_path})")
print("=" * 65)
""")

    # Write notebook
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Monitor_Training.ipynb")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Successfully generated {output_path}")

if __name__ == "__main__":
    create_monitor_notebook()
