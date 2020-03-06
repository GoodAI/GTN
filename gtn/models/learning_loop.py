from dataclasses import dataclass

import higher
import itertools
from typing import Optional, List, Callable

import torch
from badger_utils.torch.device_aware import DeviceAware
from badger_utils.torch.torch_utils import id_to_one_hot
from badger_utils.view.observer_utils import Observer, MultiObserver, CompoundObserver, ObserverLevel
from torch import nn
from torch.nn import Module
from torch.optim import Optimizer, Adam, SGD
from torch.utils.data import DataLoader
from torch import Tensor
import torch.nn.functional as F

from gtn.datasets.datasets import Datasets
from gtn.models.learner import Learner
from gtn.models.mnist_teacher import Teacher


class TrainData:
    def __init__(self, data_factory: Callable[[], Tensor], target: Tensor):
        self.data_factory = data_factory
        self.target = target

    @property
    def data(self) -> Tensor:
        return self.data_factory()


class LearningLoop(DeviceAware):
    teacher: Teacher

    # learner: Module

    def __init__(self, teacher: Teacher, learner_factory: Callable[[], Learner],
                 train_data: TrainData, test_loader: DataLoader, learning_rate: float,
                 learning_rate_learner: float, train_samples: int,
                 device: Optional[str] = None):
        super().__init__(device)
        # self.optimizer = optimizer
        self.learning_rate_learner = learning_rate_learner
        self.train_data = train_data
        self.test_loader = test_loader
        self.learner_factory = learner_factory
        self.teacher = teacher
        self.train_samples = train_samples
        # Create learner - TODO crate new learner in train step
        self.init()

        # params_all = itertools.chain(self.teacher.parameters(), self.learner.parameters())
        self.optimizer_teacher: Optimizer = Adam(self.teacher.parameters(), lr=learning_rate)
        # self.optimizer_learner: Optimizer = Adam(self.learner.parameters(), lr=learning_rate_learner)

    def init(self):
        # self.learner = self.learner_factory()
        pass

    @property
    def models(self) -> List[nn.Module]:
        return [self.teacher]

    def train_step(self, observer: Optional[CompoundObserver], inner_loop_steps: int, target_classes: int):
        # self.learner.reset()

        for model in self.models:
            model.train()

        teacher_input, teacher_input_target = self.train_data.data.to(self.device), self.train_data.target.to(self.device)
        self.optimizer_teacher.zero_grad()

        losses = []

        learner = self.learner_factory()
        learner.to(self.device)
        learner.train()

        learner_lr = self.teacher.learner_optim_params[0]
        learner_momentum = self.teacher.learner_optim_params[1]
        optim = SGD(learner.parameters(), lr=learner_lr.item(), momentum=learner_momentum.item())
        with higher.innerloop_ctx(learner, optim,
                                  override={'lr': [learner_lr], 'momentum': [learner_momentum]}) as (flearner, diffopt):
            for step in range(inner_loop_steps):
                teacher_output, teacher_target = self.teacher(teacher_input[step],
                                                              id_to_one_hot(teacher_input_target[step],
                                                                            target_classes).squeeze(1))
                learner_output = flearner(teacher_output)
                # loss = F.nll_loss(learner_output, teacher_target)
                # loss = F.cross_entropy(learner_output, teacher_target)
                loss = F.kl_div(learner_output, teacher_target)
                diffopt.step(loss)

                losses.append(loss)
                if observer is not None:
                    o = observer.rollout.add_observer()
                    o.add_tensor('teacher_output', teacher_output[0], ObserverLevel.inference)
                    o.add_tensor('teacher_target', teacher_target[0], ObserverLevel.inference)
                    o.add_tensor('learner_output', learner_output[0], ObserverLevel.inference)

            # test on Train MNIST
            # train_samples = 512*3
            # loss = torch.zeros([1], device=self.device)
            train_loader = Datasets.mnist_dataloader(self.train_samples, train=True)
            correct = 0
            # train_batches_limit = 3
            # for data, target in train_loader:
            data, target = next(x for x in train_loader)
            data, target = data.to(self.device), target.to(self.device)
            output = flearner(data)
            loss = F.nll_loss(output, target)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            # train_batches_limit -= 1
            # if train_batches_limit == 0:
            #     break
            accuracy_train = correct / self.train_samples

            # Compute accuracy on Test MNIST
            test_batch_size = 512
            test_loader = Datasets.mnist_dataloader(test_batch_size, train=False)
            correct = 0
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = flearner(data)
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
            accuracy = correct / len(test_loader.dataset)

            if observer is not None:
                observer.main.add_scalar('loss', loss.item())
                observer.main.add_scalar('accuracy_train', accuracy_train)
                observer.main.add_scalar('accuracy', accuracy)
                observer.main.add_scalar('learner_lr', learner_lr.item())
                observer.main.add_scalar('learner_momentum', learner_momentum.item())
                observer.main.add_tensor('teacher_input', teacher_input, ObserverLevel.training)
                observer.main.add_tensor('teacher_target', teacher_input_target, ObserverLevel.training)
            loss.backward()

        self.optimizer_teacher.step()

    def run_inference(self, observer: Optional[CompoundObserver], data_loader: DataLoader, target_classes: int):
        for model in self.models:
            model.eval()

        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(self.device), target.to(self.device)
            # self.optimizer_learner.zero_grad()
            learner = self.learner_factory()
            losses = []
            target_one_hot = id_to_one_hot(target, target_classes).squeeze(1)
            teacher_output, teacher_target = self.teacher(data, target_one_hot)

            learner_output = learner(teacher_output)
            loss = F.kl_div(learner_output, teacher_target)
            # loss = F.nll_loss(learner_output, teacher_target.squeeze(1))
            losses.append(loss)
            if observer is not None:
                o = observer.rollout.add_observer()
                o.add_tensor('teacher_input', data[0], ObserverLevel.inference)
                o.add_tensor('teacher_output', teacher_output[0], ObserverLevel.inference)
                o.add_tensor('teacher_target', teacher_target[0], ObserverLevel.inference)
                o.add_tensor('learner_output', learner_output[0], ObserverLevel.inference)

            # loss = F.nll_loss(output, target)
            loss = torch.stack(losses).mean()
            if observer is not None:
                observer.main.add_scalar('loss', loss.item())
            # loss.backward()
            # self.optimizer.step()

    def run_test(self, observer: Observer, test_loader: DataLoader):
        pass
