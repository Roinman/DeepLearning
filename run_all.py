"""Run meander then spiral sequentially."""
import subprocess, sys, time

py = r"D:\Develop\Anaconda3\envs\Pytorch\python.exe"
base = r"D:\Develop\Project\NewHandPD"

for task in ["meander", "spiral"]:
    print(f"\n{'='*60}")
    print(f"Starting {task} training at {time.strftime('%H:%M:%S')}")
    print(f"{'='*60}\n")
    t0 = time.time()
    ret = subprocess.run(
        [py, "train.py", "--task", task, "--epochs", "20",
         "--batch-size", "1", "--accum-steps", "8",
         "--lr", "1e-4", "--pretrained"],
        cwd=base
    )
    dt = (time.time() - t0) / 60
    print(f"\n{task} finished with exit={ret.returncode} in {dt:.0f} min\n")
