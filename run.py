import subprocess
import sys
from settings import *

bits_list = [16, 32, 64, 128]
seeds = [1024, 1307, 2021, 2026, 3407]
# epochs_list = [100, 120, 140, 150, 160, 170]
epochs_list = [200, 100, 300]  # nus-wide

Lambda_list = [1e-5, 1e-3, 1e-1, 1, 10, 1e3]
Eta_list = [1e-5, 1e-3, 1e-1, 1, 10, 1e3]
Beta_list = [1e-5, 1e-3, 1e-1, 1, 10, 1e3]
Gamma_list = [1e-5, 1e-3, 1e-1, 1, 10, 1e3]

Alpha_list = [1e-5, 1e-3, 1e-1, 1, 10, 1e3]
Mu_list = [1e-5, 1e-3, 1e-1, 1, 10, 1e3]

for Alpha_list1 in Alpha_list:
    command = [
        sys.executable, 'main.py',
        '--alpha_', str(Alpha_list1),
    ]
    subprocess.run(command)
