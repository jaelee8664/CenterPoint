import os.path as osp
import pdb
import torch

from ...utils import master_only
from .base import LoggerHook


class TensorboardLoggerHook(LoggerHook):
    def __init__(self, log_dir=None, interval=10, ignore_last=True, reset_flag=True):
        super(TensorboardLoggerHook, self).__init__(interval, ignore_last, reset_flag)
        self.log_dir = log_dir

    @master_only
    def before_run(self, trainer):
        if torch.__version__ >= "1.1":
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise ImportError(
                    'Please run "pip install future tensorboard" to install '
                    "the dependencies to use torch.utils.tensorboard "
                    "(applicable to PyTorch 1.1 or higher)"
                )
        else:
            try:
                from tensorboardX import SummaryWriter
            except ImportError:
                raise ImportError(
                    "Please install tensorboardX to use " "TensorboardLoggerHook."
                )

        if self.log_dir is None:
            self.log_dir = osp.join(trainer.work_dir, "tf_logs")
        self.writer = SummaryWriter(self.log_dir)

    @master_only
    def log(self, trainer):
        for var in trainer.log_buffer.output:
            # keys: odict_keys(['data_time', 'transfer_time', 'forward_time', 'loss_parse_time', 'loss', 'hm_loss', 'loc_loss', 'loc_loss_elem', 'num_positive', 'time'])
            
            if var in ["time", "data_time", 'transfer_time', 'forward_time', 'loss_parse_time', 'loc_loss_elem', 'num_positive']:
                continue
            tag = "{}/{}".format(var, trainer.mode)
            record = trainer.log_buffer.output[var]
            class_names = trainer.model.module.bbox_head.class_names
            try:
                if isinstance(record, str):
                    self.writer.add_text(tag, record, trainer.iter)
                elif isinstance(record, list):
                    for idx, task_class_names in enumerate(class_names):
                        tag_cls = tag + "/" + str(task_class_names)
                        self.writer.add_scalar(
                            tag_cls, trainer.log_buffer.output[var][idx], trainer.iter
                        )
                else:
                    self.writer.add_scalar(
                        tag, trainer.log_buffer.output[var], trainer.iter
                    )
            except:
                pdb.set_trace()

    @master_only
    def after_run(self, trainer):
        self.writer.close()
