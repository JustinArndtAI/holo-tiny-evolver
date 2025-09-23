import subprocess
import time

def check_gpu_temp(max_temp=80):
    try:
        output = subprocess.check_output("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader", shell=True)
        temp = int(output.decode().strip())
        if temp > max_temp:
            print(f"Hot ({temp}C)! Pausing 30s...")
            time.sleep(30)
    except:
        pass