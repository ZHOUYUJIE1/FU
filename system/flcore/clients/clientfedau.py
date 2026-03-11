import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from flcore.clients.clientavg import clientAVG
from flcore.clients.clientscaffold import clientSCAFFOLD
from utils.data_utils import read_client_data


def _local_only_prefixes(model):
    return tuple(getattr(model, "local_only_param_prefixes", ()))


def _is_local_only_key(model, key):
    return any(key.startswith(prefix) for prefix in _local_only_prefixes(model))


def _sample_fixed_wrong_label(label, num_classes, rng):
    if num_classes <= 1:
        return label
    sampled = int(rng.integers(0, num_classes - 1))
    return sampled + int(sampled >= label)


def _one_hot_cross_entropy(probabilities, labels):
    probabilities = probabilities.clamp_min(1e-12)
    return -(labels * probabilities.log()).sum(dim=1).mean()


class _FedAUMixin:
    def _init_fedau(self, args):
        self.is_target_client = self.id == getattr(args, "target_client_id", -1)
        self.fedau_client_gamma = float(getattr(args, "fedau_client_gamma", 0.5))
        self._parameter_names = [name for name, _ in self.model.named_parameters()]
        self._fedau_seed = int(getattr(args, "random_seed", 0)) * 1000 + int(self.id)
        self._fedau_target_train_data = None

    def _sync_non_local_state(self, model, copy_buffers=False):
        source_state = model.state_dict()
        target_state = self.model.state_dict()

        for key in target_state.keys():
            if _is_local_only_key(self.model, key):
                continue
            if not copy_buffers and key not in dict(model.named_parameters()):
                continue
            if key in source_state:
                target_state[key] = source_state[key].detach().clone()

        self.model.load_state_dict(target_state, strict=False)

    def _prepare_batch(self, x, y):
        if type(x) == type([]):
            x[0] = x[0].to(self.device)
            x = x[0]
        else:
            x = x.to(self.device)
        return x, y.to(self.device)

    def _prepare_target_batch(self, x, y, wrong_y):
        x, y = self._prepare_batch(x, y)
        return x, y, wrong_y.to(self.device)

    def _build_target_train_loader(self):
        if not self.is_target_client:
            return self.load_train_data()
        if self._fedau_target_train_data is None:
            base_data = read_client_data(
                self.dataset,
                self.id,
                is_train=True,
                few_shot=self.few_shot,
            )
            rng = np.random.default_rng(self._fedau_seed)
            fixed_wrong_labels = []
            for _, label in base_data:
                label_value = int(label.item())
                wrong_label = _sample_fixed_wrong_label(label_value, self.num_classes, rng)
                fixed_wrong_labels.append(torch.tensor(wrong_label, dtype=label.dtype))
            self._fedau_target_train_data = [
                (x, y, wrong_y)
                for (x, y), wrong_y in zip(base_data, fixed_wrong_labels)
            ]
        return DataLoader(
            self._fedau_target_train_data,
            self.batch_size,
            drop_last=True,
            shuffle=True,
        )

    def _blend_aux_head(self):
        if not self.is_target_client:
            return

        gamma = min(max(self.fedau_client_gamma, 0.0), 1.0)
        with torch.no_grad():
            self.model.head_ul.weight.data.copy_(
                (1.0 - gamma) * self.model.head.weight.data
                + gamma * self.model.head_ul.weight.data
            )
            if self.model.head.bias is not None and self.model.head_ul.bias is not None:
                self.model.head_ul.bias.data.copy_(
                    (1.0 - gamma) * self.model.head.bias.data
                    + gamma * self.model.head_ul.bias.data
                )

    def _fedau_target_loss(self, x, y, wrong_y):
        main_logits, aux_logits = self.model.forward_with_aux(x)

        main_probs = F.softmax(main_logits, dim=1)
        aux_probs = F.softmax(aux_logits, dim=1)
        probabilities = torch.cat((main_probs, aux_probs), dim=1)

        labels = torch.cat(
            (
                F.one_hot(y, num_classes=self.num_classes),
                F.one_hot(wrong_y, num_classes=self.num_classes),
            ),
            dim=1,
        ).float()
        return _one_hot_cross_entropy(probabilities, labels)

    def _compute_loss(self, x, y, wrong_y=None):
        if self.is_target_client:
            if wrong_y is None:
                raise ValueError("FedAU target client requires fixed auxiliary labels.")
            return self._fedau_target_loss(x, y, wrong_y)
        return self.loss(self.model(x), y)


class clientFedAUAVG(_FedAUMixin, clientAVG):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self._init_fedau(args)

    def set_parameters(self, model, copy_buffers=False):
        self._sync_non_local_state(model, copy_buffers=copy_buffers)

    def train(self):
        trainloader = self._build_target_train_loader()
        self.model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for _ in range(max_local_epochs):
            self._blend_aux_head()
            for batch in trainloader:
                if self.is_target_client:
                    x, y, wrong_y = batch
                    x, y, wrong_y = self._prepare_target_batch(x, y, wrong_y)
                else:
                    x, y = batch
                    x, y = self._prepare_batch(x, y)
                    wrong_y = None

                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                loss = self._compute_loss(x, y, wrong_y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start_time


class clientFEDAUSCAFFOLD(_FedAUMixin, clientSCAFFOLD):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self._init_fedau(args)

    def set_parameters(self, model, global_c=None, copy_buffers=False):
        self._sync_non_local_state(model, copy_buffers=copy_buffers)
        if global_c is not None:
            self.global_c = global_c
        self.global_model = model

    def train(self):
        trainloader = self._build_target_train_loader()
        self.model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for _ in range(max_local_epochs):
            self._blend_aux_head()
            for batch in trainloader:
                if self.is_target_client:
                    x, y, wrong_y = batch
                    x, y, wrong_y = self._prepare_target_batch(x, y, wrong_y)
                else:
                    x, y = batch
                    x, y = self._prepare_batch(x, y)
                    wrong_y = None

                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                loss = self._compute_loss(x, y, wrong_y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step(self.global_c, self.client_c)

        self.num_batches = len(trainloader)
        self.update_yc(max_local_epochs)

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start_time

    def update_yc(self, max_local_epochs=None):
        if max_local_epochs is None:
            max_local_epochs = self.local_epochs
        for name, ci, c, x, yi in zip(
            self._parameter_names,
            self.client_c,
            self.global_c,
            self.global_model.parameters(),
            self.model.parameters(),
        ):
            if _is_local_only_key(self.model, name):
                ci.data.zero_()
                continue
            ci.data = ci - c + 1 / self.num_batches / max_local_epochs / self.learning_rate * (x - yi)

    def delta_yc(self, max_local_epochs=None):
        if max_local_epochs is None:
            max_local_epochs = self.local_epochs
        delta_y = []
        delta_c = []
        for name, c, x, yi in zip(
            self._parameter_names,
            self.global_c,
            self.global_model.parameters(),
            self.model.parameters(),
        ):
            if _is_local_only_key(self.model, name):
                delta_y.append(torch.zeros_like(yi))
                delta_c.append(torch.zeros_like(c))
                continue
            delta_y.append(yi - x)
            delta_c.append(-c + 1 / self.num_batches / max_local_epochs / self.learning_rate * (x - yi))

        return delta_y, delta_c
