import subprocess
import sys
from settings import *

bits_list = [16, 32, 64, 128]
seeds = [1024, 1307, 2021, 2026, 3407]
# epochs_list = [100, 120, 140, 150, 160, 170]
epochs_list = [200, 100, 300]  # nus-wide
# alpha_train_list = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
alpha_train_list = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
# alpha_train_list = [0.5, 0.3, 0.1]
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
# for alpha_train in alpha_train_list:
#     for bits in bits_list:
#         command = [
#             sys.executable, 'main.py',
#             '--alpha_train', str(alpha_train),
#             '--bits', str(bits),
#         ]
#         subprocess.run(command)


# for alpha_train in alpha_train_list:
#     command = [
#         sys.executable, 'main.py',
#         '--alpha_train', str(alpha_train),
#     ]
#     subprocess.run(command)

MU_list = [1e-5, 1e-3, 1e-1, 1, 10, 1e3]
# for MU in MU_list:
#     command = [
#         sys.executable, 'main.py',
#         '--mu', str(MU),
#     ]
#     subprocess.run(command)


Lambda1_list = [1e-5, 1e-3, 1e-1, 1, 10, 1e3]
# for Lambda1 in Lambda1_list:
#     command = [
#         sys.executable, 'main.py',
#         '--lambda1', str(Lambda1),
#     ]
#     subprocess.run(command)

Lambda2_list = [1e-5, 1e-3, 1e-1, 1, 10, 1e3]
# for Lambda2 in Lambda2_list:
#     command = [
#         sys.executable, 'main.py',
#         '--lambda2', str(Lambda2),
#     ]
#     subprocess.run(command)
Lambda3_list = [1e-5, 1e-3, 1e-1, 1, 10, 1e3]
# for Lambda3 in Lambda3_list:
#     command = [
#         sys.executable, 'main.py',
#         '--lambda3', str(Lambda3),
#     ]
#     subprocess.run(command)
Gamma1_list = [1e-5, 1e-3, 1e-1, 1, 10, 1e3]
# for Gamma1 in Gamma1_list:
#     command = [
#         sys.executable, 'main.py',
#         '--gamma1', str(Gamma1),
#     ]
#     subprocess.run(command)
Gamma2_list = [1e-5, 1e-3, 1e-1, 1, 10, 1e3]
# for Gamma2 in Gamma2_list:
#     command = [
#         sys.executable, 'main.py',
#         '--gamma2', str(Gamma2),
#     ]
#     subprocess.run(command)
Eta1_list = [1e-5, 1e-3, 1e-1, 1, 10, 1e3]
# for Eta1 in Eta1_list:
#     command = [
#         sys.executable, 'main.py',
#         '--eta1', str(Eta1),
#     ]
#     subprocess.run(command)
Eta2_list = [1e-5, 1e-3, 1e-1, 1, 10, 1e3]
# for Eta2 in Eta2_list:
#     command = [
#         sys.executable, 'main.py',
#         '--eta2', str(Eta2),
#     ]
#     subprocess.run(command)



# for alpha_train in alpha_train_list:
#     command = [
#         sys.executable, 'main.py',
#         '--alpha_train', str(alpha_train),
#     ]
#     subprocess.run(command)
# for seed in seeds:
#     command = [
#         sys.executable, 'main.py',
#         '--seed', str(seed),
#     ]
#     subprocess.run(command)
# for alpha_train in alpha_train_list:
#     for bit in bits_list:
#         command = [
#             sys.executable, 'main.py',
#             '--alpha_train', str(alpha_train),
#             '--bits', str(bit),
#         ]
#         subprocess.run(command)