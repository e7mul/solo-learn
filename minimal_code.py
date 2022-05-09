from multiprocessing.sharedctypes import Value
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin

from solo.methods import BarlowTwins  # imports the method class
from solo.utils.checkpointer import Checkpointer
import pytorch_lightning as pl

# some data utilities
# we need one dataloader to train an online linear classifier
# (don't worry, the rest of the model has no idea of this classifier, so it doesn't use label info)
from solo.utils.classification_dataloader import prepare_data as prepare_data_classification

# and some utilities to perform data loading for the method itself, including augmentation pipelines
from solo.utils.pretrain_dataloader import (
    prepare_dataloader,
    prepare_datasets,
    prepare_n_crop_transform,
    prepare_transform,
)
from torch.utils.data.dataset import Dataset


from pytorch_lightning import LightningDataModule, LightningModule
from argparse import Namespace

from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10
import numpy as np



class AppendName(torch.utils.data.Dataset):
    def __init__(self, dataset, name):
        super().__init__()
        self.dataset = dataset
        self.name = name
        self.indices = dataset.indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        all_the_stuff = self.dataset[index]
        return (*all_the_stuff, self.name)


class AppendRemappedClass(Dataset):
    def __init__(self, dataset, classes_per_task):
        super().__init__()
        for key, val in dataset.__dict__.items():
            setattr(self, key, val)
        self.dataset = dataset
        self.classes_per_task = classes_per_task

    
    def __getitem__(self, index):
        *_, image, target = self.dataset[index] #This can either return two items (test dataset) or multiple items (e.g. index of the sample)
        remapped_target = target % self.classes_per_task
        return (*_, image, target, remapped_target)


num_tasks = 5
num_classes = 10
epochs_per_task = 100

# common parameters for all methods
# some parameters for extra functionally are missing, but don't mind this for now.
base_kwargs = {
    "backbone": "resnet18",
    "num_classes": 10,
    "cifar": True,
    "zero_init_residual": True,
    "max_epochs": 100,
    "optimizer": "sgd",
    "lars": True,
    "lr": 0.01,
    "gpus": [1],
    "grad_clip_lars": True,
    "weight_decay": 0.00001,
    "classifier_lr": 0.5,
    "exclude_bias_n_norm": True,
    "accumulate_grad_batches": 1,
    "extra_optimizer_args": {"momentum": 0.9},
    "scheduler": "warmup_cosine",
    "min_lr": 0.0,
    "warmup_start_lr": 0.0,
    "warmup_epochs": 10,
    "num_crops_per_aug": [2, 0],
    "num_large_crops": 2,
    "num_small_crops": 0,
    "eta_lars": 0.02,
    "lr_decay_steps": None,
    "dali_device": "gpu",
    "batch_size": 256,
    "num_workers": 4,
    "data_dir": "/data/datasets",
    "train_dir": "cifar10/train",
    "val_dir": "cifar10/val",
    "dataset": "cifar10",
    "name": "barlow-cifar10",
    "num_tasks": num_tasks
}

# barlow specific parameters
method_kwargs = {
    "proj_hidden_dim": 2048,
    "proj_output_dim": 2048,
    "lamb": 5e-3,
    "scale_loss": 0.025,
    "backbone_args": {"cifar": True, "zero_init_residual": True},
}

kwargs = {**base_kwargs, **method_kwargs}

model = BarlowTwins(**kwargs)


# we first prepare our single transformation pipeline
transform_kwargs = {
    "brightness": 0.4,
    "contrast": 0.4,
    "saturation": 0.2,
    "hue": 0.1,
    "gaussian_prob": 0.0,
    "solarization_prob": 0.0,
}


class CLDataModule(LightningDataModule):
    def __init__(self, train_tasks_split, epochs_per_dataset, val_tasks_split=None):
        """
        Args:
            task_split: list containing list of class indices per task
        """
        super().__init__()
        self.train_tasks_split = train_tasks_split
        self.val_tasks_split = (
            val_tasks_split if val_tasks_split is not None else train_tasks_split
        )
        self.curr_index = 0
        self.epochs_per_dataset = epochs_per_dataset

    def prepare_data(self):        
        transform = [prepare_transform("cifar10", **transform_kwargs)]
        transform = prepare_n_crop_transform(transform, num_crops_per_aug=[2])

        train_dataset, test_dataset = prepare_datasets(
        "cifar10",
        transform,
        data_dir="./",
        train_dir=None,
        no_labels=False,
        )

        train_dataset = AppendRemappedClass(train_dataset, num_classes//num_tasks)
        test_dataset = AppendRemappedClass(test_dataset, num_classes//num_tasks)

        self.train_datasets = self._split_dataset(train_dataset, self.train_tasks_split)
        self.test_datasets = self._split_dataset(test_dataset, self.val_tasks_split)
        self.train_dataset = train_dataset

    @staticmethod
    def _split_dataset(dataset, tasks_split):
        split_dataset = []
        for e, current_classes in enumerate(tasks_split):
            task_indices = np.isin(np.array(dataset.targets), current_classes)
            split_dataset.append(AppendName(Subset(dataset, np.where(task_indices)[0]), e))
        return split_dataset

    def train_dataloader(self):
        loader = DataLoader(
            self.train_datasets[self.curr_index], batch_size=base_kwargs['batch_size'], num_workers=base_kwargs['num_workers']
        )
        self.curr_index += 1
        return loader

    def val_dataloader(self):
        return [
            DataLoader(dataset, batch_size=base_kwargs['batch_size'], num_workers=base_kwargs['num_workers'])
            for dataset in self.test_datasets
        ]

    def test_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=base_kwargs['batch_size'], num_workers=base_kwargs['num_workers'])


def classic_tasks_split(num_classes, num_tasks):
    one_split = num_classes // num_tasks
    return [list(range(i * one_split, (i + 1) * one_split)) for i in range(num_tasks)]


wandb_logger = WandbLogger(
    name="barlow-cifar10",  # name of the experiment
    project="self-supervised",  # name of the wandb project
    entity=None,
    offline=False,
)
wandb_logger.watch(model, log="gradients", log_freq=100)

callbacks = []

# automatically log our learning rate
lr_monitor = LearningRateMonitor(logging_interval="epoch")
callbacks.append(lr_monitor)

# checkpointer can automatically log your parameters,
# but we need to wrap it on a Namespace object

args = Namespace(**kwargs)
# saves the checkout after every epoch
ckpt = Checkpointer(
    args,
    logdir="checkpoints/barlow",
    frequency=1,
)
callbacks.append(ckpt)


################################################################################

# trainer = Trainer.from_argparse_args(
#     args,
#     logger=wandb_logger,
#     callbacks=callbacks,
#     plugins=DDPPlugin(find_unused_parameters=True),
#     checkpoint_callback=False,
#     terminate_on_nan=True,
#     accelerator="ddp",
# )



# for task, train_loader in train_loaders.items():
#     trainer.fit(model, train_loader, val_loaders)
#     trainer.save_checkpoint("example.ckpt")
#     model = BarlowTwins(**kwargs)
#     model = model.load_from_checkpoint(checkpoint_path="example.ckpt", **kwargs)

################################################################################


pl.seed_everything(42)
train_tasks_split = [list(range(num_classes))[i * 2 :] for i in range(num_tasks)]
val_tasks_split = classic_tasks_split(num_classes, num_tasks)
cl_data_module = CLDataModule(train_tasks_split, epochs_per_task, val_tasks_split)
model = BarlowTwins(**kwargs)
trainer = Trainer.from_argparse_args(
    args,
    max_epochs=num_tasks * epochs_per_task,
    reload_dataloaders_every_n_epochs=epochs_per_task,
    logger=wandb_logger,
    gpus="0,",
    callbacks=callbacks,
    plugins=DDPPlugin(find_unused_parameters=True),
    checkpoint_callback=False,
    terminate_on_nan=True,
    accelerator="ddp",
)
trainer.fit(model, datamodule=cl_data_module)
# trainer.test()