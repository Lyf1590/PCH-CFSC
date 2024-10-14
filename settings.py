import logging
import time
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataname', type=str, default='flickr', help='Dataset name: flickr/coco/nus-wide/')
parser.add_argument('--bits', type=int, default=128, help='16/32/64/128')
parser.add_argument('--gpu', default=0, type=int, help='0/1/2/3')
parser.add_argument('--alpha_train', type=float, default=0.6, help='Missing ratio of train set.')  # 完整的
parser.add_argument('--beta_train', type=float, default=0.5)  # 缺失的
parser.add_argument('--alpha_query', type=float, default=1.0, help='Missing ratio of query set.')
parser.add_argument('--beta_query', type=float, default=0.5)
parser.add_argument('--seed', type=int, default=3407, help='1307,1024,2026,3407')
parser.add_argument('--epochs', type=int, default=100, help='100,120,140,150,160,170')

parser.add_argument('--alpha_', type=float, default=1, help='Completing Missing modality Representations  LCMR')
parser.add_argument('--mu_', type=float, default=1, help='Semantic Contrast Learning  LSCL')
parser.add_argument('--lambda_', type=float, default=1, help='Predicting Loss  ')
parser.add_argument('--eta_', type=float, default=10, help='Similarity Loss')
parser.add_argument('--beta_', type=float, default=0.001, help='Discriminative hashing Loss')
parser.add_argument('--gamma_', type=float, default=10, help='Binary Loss')

parser.add_argument('--topN_number', type=int, nargs='+', default=[0, 300, 600, 900, 1200, 1500])
args = parser.parse_args()

if args.dataname == 'flickr':
    DIR = '/home/user3/lyf/Datasets/mir_clip_all.mat'
    # DIR = '/home/user3/lyf/pttcmh/dataset/mir_clip_all.mat'
elif args.dataname == 'nus-wide':
    DIR = '/home/user3/lyf/Datasets/nus_clip_all.mat'
elif args.dataname == 'coco': 
    DIR = '/home/user3/lyf/Datasets/coco_clip_all.mat'


DATASET_NAME = args.dataname
CODE_LEN = args.bits
GPU = args.gpu
alpha_train = args.alpha_train
beta_train = args.beta_train
alpha_query = args.alpha_query
beta_query = args.beta_query
NUM_EPOCH = args.epochs
topN_Number = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

SEED = args.seed  # 随机种子 1024/1307/2021/2026/3407
ANCHOR_NUMS = 10  # 锚点个数
EVAL_INTERVAL = 5  # Evaluation: Calculate top MAP，5次结果取最好的best值
BATCH_SIZE = 512    # MIRFlickr 256     nusswide and  coco 512
NUM_WORKERS = 2    # 0 或 2  线程占内存
EPOCH_INTERVAL = 4  # 周期间隔
COM_Part = False  # 完整还是缺失
EVAL = False  # EVAL = True: just test, EVAL = False: train and eval


Alpha = args.alpha_
Mu = args.mu_
Lambda = args.lambda_
Eta = args.eta_
Beta = args.beta_
Gamma = args.gamma_

logger = logging.getLogger('train')
logger.setLevel(logging.INFO)
now = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
log_name = f"{args.dataname}_{NUM_EPOCH}_{BATCH_SIZE}_{CODE_LEN}_{now}_log.txt"  
log_dir = './log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
txt_log = logging.FileHandler(os.path.join(log_dir, log_name))
txt_log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
txt_log.setFormatter(formatter)
logger.addHandler(txt_log)

stream_log = logging.StreamHandler()
stream_log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_log.setFormatter(formatter)
logger.addHandler(stream_log)