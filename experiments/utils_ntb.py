# %%

import math
import matplotlib
from badger_utils.torch.torch_utils import id_to_one_hot
from badger_utils.view.bokeh_utils import plot_dataframe, DataTableBuilder
from matplotlib.colors import LinearSegmentedColormap
from torch.optim import Optimizer
from tqdm import tqdm

import os
import sys
import traceback
from dataclasses import dataclass, field
from functools import partial, reduce
from typing import List, Dict, Optional, Any, Callable, Tuple

import time
import pylab as plt
from IPython import display
# plt.style.use('bmh')
# plt.style.use('dark_background')

import torch
import torch.nn.functional as F
import pandas as pd
from badger_utils.sacred import SacredReader, SacredUtils
from badger_utils.sacred.sacred_config import SacredConfigFactory
from torch import Tensor, optim
from badger_utils.view.observer_utils import MultiObserver, CompoundObserver, ObserverLevel, Observer

from gtn.datasets.datasets import Datasets
from gtn.models.learning_loop import LearningLoop
from gtn.models.mnist_learner import MNISTLearnerConfig, MNISTLearner
from gtn.models.mnist_teacher import MNISTTeacher
from experiments.mnist_experiment import Config, create_teacher, create_learner_factory

sacred_utils = SacredUtils(SacredConfigFactory.local())


def plot_df(df: pd.DataFrame, include_min_max: bool = True, p=plt):
    # p.figure()
    df = df.dropna()
    df_agg = df.agg(['min', 'max', 'mean'], axis=1)
    if include_min_max:
        p.fill_between(x=df_agg.index, y1='min', y2='max', data=df_agg, alpha=0.25)
    p.plot(df_agg['mean'])


def cmap_red_green():
    return LinearSegmentedColormap.from_list("", [[0.0, "#ff0000"], [0.5, "#000000"], [1.0, "#00ff00"]])


def norm_red_green():
    return matplotlib.colors.Normalize(-1, 1, clip=True)


def plot_image(tensor: Tensor, p=plt):
    # p.figure()
    p.axis('off')
    img = p.imshow(tensor.cpu().numpy(), norm=norm_red_green())
    img.set_cmap(cmap_red_green())


def plot_images(data: Tensor, columns=0, size=(28, 28), tight_pad=0):
    # plt.figure()
    count = data.shape[0]
    if columns == 0:
        columns = math.ceil(math.sqrt(count))
    rows = math.ceil(count / columns)
    f, axarr = plt.subplots(rows, columns)
    dpi = 72
    # f.set_dpi(dpi)
    f.set_size_inches((columns * size[0] / dpi, rows * size[1] / dpi))
    flat_axes = axarr.reshape(-1)
    # setup figure
    for ax in flat_axes:
        ax.axis('off')
    for i in range(count):
        plot_image(data[i, 0], flat_axes[i])
    f.tight_layout(pad=tight_pad)
    return f


# Teacher output data generation
def generate_teacher_images(batch_size: int, teacher: MNISTTeacher):
    loader = Datasets.random_dataloader_with_targets(batch_size, teacher.config.input_size, batch_size,
                                                     teacher.config.target_classes, device='cuda')
    with torch.no_grad():
        (data, target) = next(x for x in loader)
        target_one_hot = id_to_one_hot(target, teacher.config.target_classes).squeeze(1)
        teacher_output, teacher_target = teacher(data, target_one_hot)
    return teacher_output, teacher_target


def generate_teacher_images_for_class(batch_size: int, teacher: MNISTTeacher, cls: int, config: Config):
    # loader = Datasets.random_dataloader(batch_size, config.teacher_input_size, batch_size, device='cuda')
    loader = Datasets.random_dataloader_with_targets(batch_size, config.teacher_input_size, batch_size,
                                                     config.mnist_classes, device='cuda')
    with torch.no_grad():
        data, target = next(x for x in loader)
        classes = torch.tensor([[cls]], device='cuda').expand(batch_size, 1)
        # classes = torch.randint(7, 9, (batch_size, 1), device = 'cuda').expand(batch_size, 1)
        target_one_hot = id_to_one_hot(classes, config.mnist_classes).squeeze(1)
        teacher_output, teacher_target = teacher(data, target_one_hot)
    return teacher_output, teacher_target


# Learner training
def train_step(model: MNISTLearner, data: Tensor, target: Tensor, optimizer: Optimizer, observer: Observer):
    model.train()
    data, target = data.to(model.device), target.to(model.device)
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    observer.add_scalar('training_loss', loss.mean().item())


def load_teacher(experiment_id: int, epoch: int) -> Tuple[MNISTTeacher, Config]:
    reader = sacred_utils.get_reader(experiment_id)
    config = Config(**reader.config)
    learner_factory = create_learner_factory(config)
    teacher = create_teacher(config)
    teacher.eval()
    # teacher_loader = Datasets.random_dataloader_with_targets(config.batch_size, config.teacher_input_size, config.batch_size * epochs, config.mnist_classes, device='cuda')
    mnist_loader = Datasets.mnist_dataloader(config.batch_size, train=False)
    reader.load_model(teacher, 'teacher', epoch=epoch)
    return teacher, config


def plot_teacher_outputs(teacher: MNISTTeacher, config: Config):
    # teacher_output, teacher_target = generate_teacher_images(32, teacher)
    # teacher_output, teacher_target = generate_teacher_images_for_class(32, teacher, 3, config)
    data = torch.cat([generate_teacher_images_for_class(10, teacher, i, config)[0] for i in range(10)])
    fig = plot_images(data)
    return fig


def generate_teacher_curriculum_images(teacher: MNISTTeacher, config: Config):
    with torch.no_grad():
        data = teacher.input_data.view(-1, teacher.input_data.shape[-1])
        target = teacher.input_target.reshape(-1, teacher.input_target.shape[-1])
        # classes = torch.tensor([[cls]], device = 'cuda').expand(batch_size, 1)
        # classes = torch.randint(7, 9, (batch_size, 1), device = 'cuda').expand(batch_size, 1)
        target_one_hot = id_to_one_hot(target, config.mnist_classes).squeeze(1)
        teacher_output, teacher_target = teacher(data, target_one_hot)
    return teacher_output, teacher_target


def plot_teacher_outputs_curriculum(teacher: MNISTTeacher):
    # teacher_output, teacher_target = generate_teacher_images(32, teacher)
    data, target = generate_teacher_curriculum_images(teacher)

    # data = torch.cat([generate_teacher_images_for_class(10, teacher, i)[0] for i in range(10)])
    fig = plot_images(data)
    return fig
