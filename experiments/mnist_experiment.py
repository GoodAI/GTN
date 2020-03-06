import dataclasses
from typing import Optional

import itertools
import random
from dataclasses import dataclass

from badger_utils.sacred import SacredConfigFactory, SacredUtils
from badger_utils.torch.device_aware import DeviceAwareModule, default_device
from sacred.run import Run
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import StepLR

from badger_utils.view.observer_utils import Observer, MultiObserver, ObserverLevel, CompoundObserver

from sacred import Experiment

from gtn.datasets.datasets import Datasets
from gtn.models.learning_loop import LearningLoop, TrainData
from gtn.models.mnist_learner import MNISTLearner, MNISTLearnerConfig
from gtn.models.mnist_teacher import MNISTTeacherConfig, MNISTTeacher, TeacherInputType

ex = Experiment("GTN mnist")
sacred_utils = SacredUtils(SacredConfigFactory.local())
sacred_writer = sacred_utils.get_writer(ex)


@dataclass
class Config:
    seed: int
    batch_size: int = 16
    learning_rate: float = 1e-2
    learning_rate_learner: float = 1e-2
    learning_rate_gamma: float = 0.7
    epochs: int = 2000
    inner_loop_steps: int = 10
    mnist_classes: int = 10
    train_samples: int = 1024

    experiment_agent_save_period: int = 500
    experiment_loss_save_period: int = 1

    training_log_interval: int = 10
    learner_conv1_filters: int = 64  # sample from [32 to 128]
    learner_conv2_filters: int = 128  # sample from [64 to 256]
    learner_fc_size: int = 64  # not used

    teacher_input_size: int = 64
    teacher_output_size: int = 28  # width and height of output image size
    teacher_conv1_filters: int = 64
    teacher_fc1_size: int = 1024
    teacher_fc2_filters: int = 128
    teacher_input_type: str = 'learned'  # random, random_fixed, learned


ex.add_config(dataclasses.asdict(Config(random.randint(0, 2 ** 32))))


def load_teacher(experiment_id: int, epoch: int, teacher):
    sacred_utils.get_reader(experiment_id).load_model(teacher, 'teacher', epoch)


def create_teacher(c: Config) -> MNISTTeacher:
    teacher_config = MNISTTeacherConfig(input_size=c.teacher_input_size, output_size=c.teacher_output_size,
                                        fc1_size=c.teacher_fc1_size, fc2_filters=c.teacher_fc2_filters,
                                        conv1_filters=c.teacher_conv1_filters, target_classes=c.mnist_classes,
                                        input_count=c.inner_loop_steps, batch_size=c.batch_size,
                                        input_type=TeacherInputType.from_string(c.teacher_input_type))
    return MNISTTeacher(teacher_config)


def create_learner_factory(c: Config) -> MNISTLearner:
    learner_config = MNISTLearnerConfig(conv1_filters=c.learner_conv1_filters, conv2_filters=c.learner_conv2_filters,
                                        fc_size=c.learner_fc_size, target_classes=c.mnist_classes)
    return lambda: MNISTLearner(learner_config)


def run_train(model: DeviceAwareModule, train_loader: DataLoader, optimizer: Optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(model.device), target.to(model.device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def run_test(observer: Observer, model: DeviceAwareModule, test_loader: DataLoader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(model.device), target.to(model.device)
            output = model(data)
            test_loss += F.cross_entropy(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)

    observer.add_scalar('loss', test_loss)
    observer.add_scalar('accuracy', accuracy)


@ex.automain
def main(_run: Run, _config):
    c = Config(**_config)
    learner_factory = create_learner_factory(c)
    teacher = create_teacher(c)
    train_data = TrainData(lambda: teacher.input_data, teacher.input_target)
    test_loader = Datasets.mnist_dataloader(c.batch_size, train=False)

    learning_loop = LearningLoop(teacher, learner_factory, train_data, test_loader, learning_rate=c.learning_rate,
                                 learning_rate_learner=c.learning_rate_learner, train_samples=c.train_samples)
    for epoch in tqdm(range(1, c.epochs + 1)):
        observer = CompoundObserver(ObserverLevel.training) if epoch % c.experiment_loss_save_period == 0 else None
        learning_loop.train_step(observer, c.inner_loop_steps, c.mnist_classes)
        if observer is not None:
            sacred_writer.save_observer(observer.main, epoch)

        if epoch % c.experiment_agent_save_period == 0:
            sacred_writer.save_model(teacher, 'teacher', epoch)
