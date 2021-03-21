import os
import numpy as np
from numbers import Number
from typing import Union
from torch.utils.tensorboard import SummaryWriter

from .basic import get_datetime

class Logger:
    """
    A loggger to log and visualize (based on Tensorboard) statistics.

    Args:
        root (str): Root directory of the log files
        n_iters_per_epoch (int): Number of the iterations per epoch
        log_basename (str, optional, default=''): Base name of the log file
        log_interval (int, optional, default=100): Log interval
        tensorboard (bool, optional, default=True): Enable tensorboard or not
    """

    def __init__(
        self,
        root: str,
        n_iters_per_epoch: int,
        log_basename: str = '',
        log_interval: int = 100,
        tensorboard: bool = True
    ) -> None:
        self.root = os.path.expanduser(root)
        self.timestamp = get_datetime()

        self.n_iters_per_epoch = n_iters_per_epoch
        self.log_interval = log_interval
        self.log_basename = log_basename

        self.text_path = os.path.join(self.root, 'text')
        self.mkdir(self.text_path)

        if tensorboard:
            self.tensorboard_path = os.path.join(
                self.root, 'tensorboard', self.log_basename + '_' + self.timestamp
            )
            self.mkdir(self.tensorboard_path)
            self.writter = SummaryWriter(self.tensorboard_path)

    @staticmethod
    def mkdir(path: str):
        if not os.path.exists(path):
            os.makedirs(path)

    def write_tensorboard(
        self, key: str, x: Union[Number, np.number], y: Union[Number, np.number]
    ) -> None:
        """
        Log data into Tensorboard.

        Args:
            key (str): Namespace which the input data tuple belongs to
            x (Union[Number, np.number]): Ordinate of the input data
            y (Union[Number, np.number]): Abscissa of the input data
        """
        self.writter.add_scalar(key, y, global_step=x)

    def write_text(self, text: str) -> None:
        """
        Log data into text files.

        Args:
            text (str): A string to be logged
        """
        log_file_path = os.path.join(
            self.text_path,
            self.log_basename + '_' + self.timestamp + '.log'
        )
        with open(log_file_path, "a") as f:
            f.write(text)

    def log(self, data: dict, epoch: int, i_iter: int, stage: str) -> None:
        """
        Log statistics generated during updating.

        Args:
            data (dict): Data to be logged
            epoch (int): Epoch of the data to be logged
            i_iter (int): Iteration of the data to be logged
            stage (str): Name of the current stage
        """
        text = 'Epoch ({stage}): [{0}][{1}/{2}]\t'.format(
            epoch, i_iter, self.n_iters_per_epoch, stage=stage
        )
        step = (epoch - 1) * self.n_iters_per_epoch + i_iter

        if step % self.log_interval == 0:
            for name, value in data.items():
                # log statistics to Tensorboard
                if self.writter is not None:
                    self.write_tensorboard(name, step, value[-1])
                # log statistics to text files
                text += '{name} {recent:.3f} ({mean:.3f})\t'.format(
                    name = name, recent = value[-1], mean = value.mean()
                )
            self.write_text(text + '\n')
            self.last_log_step = step
