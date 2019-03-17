import torch

NUM_WORKERS = 10
BATCH_SIZE = 128
LR = [(0, 2.), (50, 2./5), (64, 2./5/5)]
WEIGHTS_DECAY = 0.00001
EPOCHS_PER_TASK = 70
TASK_SIZE = 10
K = 2000

SAVE_PATH = "model.pth"

DEVICE = torch.device("cuda:0")
