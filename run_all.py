# run_all.py
import os
import subprocess

scripts = ["processing.py", "training.py", "simulation.py"]

for script in scripts:
    print(f"\n🚀 Running {script}...")
    subprocess.run(["python", script], check=True)

print("\n✅ All stages completed successfully!")
