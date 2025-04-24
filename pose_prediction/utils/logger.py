import os
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.step = 0

    def log_scalar(self, name, value, step=None):
        self.writer.add_scalar(name, value, step if step is not None else self.step)

    def next_step(self):
        self.step += 1

    def close(self):
        self.writer.close()