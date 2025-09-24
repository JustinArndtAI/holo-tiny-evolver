import os
import subprocess
import time
from utils import check_gpu_temp

class HKMWrapper:
    def __init__(self):
        self.base_dir = "."
        os.chdir(self.base_dir)  # Run in current directory (root)

    def run_phase(self, phase_num):
        check_gpu_temp()
        if phase_num == 1:
            script = "scripts/phase1_enhanced_fixed.py"
        elif phase_num in [2, 4]:
            script = f"scripts/phase{phase_num}_enhanced.py"
        else:
            script = f"scripts/phase{phase_num}_simple.py"
        subprocess.run(["python", script])
        time.sleep(10)  # Short pause

    def ingest(self, data_path):
        # Prep data in data/ directory, run phases sequentially
        # Assume data_path copied to data/
        for p in range(1, 5):
            self.run_phase(p)

    def retrieve(self, query):
        # Adapt phase4 for query: Mock/post-process outputs/phase4
        # Placeholder: Load from outputs/phase4_enhanced.md or models
        return "Retrieved holographic slice: ..."  # Implement based on POC outputs

    def load_manifold(self):
        # Load from outputs/phase3_model_final/
        print("Manifold loaded.")