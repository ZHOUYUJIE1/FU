import copy
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        torch.manual_seed(0)
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs
        self.few_shot = args.few_shot
        # Training-time backdoor injection is opt-in so legacy experiments stay unchanged.
        self.backdoor_train_poison_enabled = bool(getattr(args, "backdoor_train_poison_enabled", False))
        poison_client_id = int(getattr(args, "backdoor_train_poison_client_id", -1))
        if poison_client_id < 0:
            poison_client_id = int(getattr(args, "target_client_id", -1))
        self.backdoor_train_poison_client_id = poison_client_id
        self.backdoor_train_poison_rate = min(max(float(getattr(args, "backdoor_train_poison_rate", 0.0)), 0.0), 1.0)
        self.backdoor_train_target_label = int(getattr(args, "backdoor_target_label", 0))
        self.backdoor_label_mode = str(getattr(args, "backdoor_label_mode", "fixed_target")).strip().lower()
        self.backdoor_train_patch_size = max(1, int(getattr(args, "backdoor_trigger_patch_size", 3)))
        self.backdoor_train_patch_value = float(getattr(args, "backdoor_trigger_patch_value", 1.0))
        self.backdoor_fedosd_target_client = bool(getattr(args, "backdoor_fedosd_target_client", False))
        self.backdoor_train_exclude_target_label = bool(
            getattr(args, "backdoor_train_exclude_target_label", True)
        )
        self.backdoor_train_seed = int(getattr(args, "backdoor_train_seed", 0)) + int(self.id) * 100003
        self._backdoor_train_poison_logged = False

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.loss = nn.CrossEntropyLoss()
        default_momentum = 0.9 if "Cifar" in self.dataset else 0.0
        default_weight_decay = 5e-4 if "Cifar" in self.dataset else 0.0
        momentum = getattr(args, "sgd_momentum", None)
        weight_decay = getattr(args, "weight_decay", None)
        if momentum is None:
            momentum = default_momentum
        if weight_decay is None:
            weight_decay = default_weight_decay
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay


    def load_train_data(self, batch_size=None, apply_train_transform=True, shuffle=True, drop_last=True):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(
            self.dataset,
            self.id,
            is_train=True,
            few_shot=self.few_shot,
            apply_train_transform=apply_train_transform,
        )
        train_data = self._maybe_apply_backdoor_train_poison(
            train_data,
            apply_train_transform=apply_train_transform,
        )
        return DataLoader(train_data, batch_size, drop_last=drop_last, shuffle=shuffle)

    def _apply_backdoor_patch_to_sample(self, x):
        tensor = None
        if torch.is_tensor(x):
            tensor = x.detach().clone()
            payload = tensor
        elif isinstance(x, tuple):
            payload = list(copy.deepcopy(x))
            if payload and torch.is_tensor(payload[0]):
                tensor = payload[0].detach().clone()
                payload[0] = tensor
            else:
                return x, False
        elif isinstance(x, list):
            payload = copy.deepcopy(x)
            if payload and torch.is_tensor(payload[0]):
                tensor = payload[0].detach().clone()
                payload[0] = tensor
            else:
                return x, False
        else:
            return x, False

        if tensor is None or tensor.dim() < 3:
            return x, False

        patch = min(self.backdoor_train_patch_size, int(tensor.shape[-2]), int(tensor.shape[-1]))
        if patch <= 0:
            return x, False
        tensor[..., -patch:, -patch:] = self.backdoor_train_patch_value

        if torch.is_tensor(x):
            return payload, True
        if isinstance(x, tuple):
            return tuple(payload), True
        return payload, True

    def _maybe_apply_backdoor_train_poison(self, train_data, apply_train_transform=True):
        # Keep clean loaders intact; only poison true local-training data paths.
        if not apply_train_transform:
            return train_data
        if not self.backdoor_train_poison_enabled:
            return train_data
        if int(self.id) != int(self.backdoor_train_poison_client_id):
            return train_data
        if self.backdoor_train_poison_rate <= 0.0 and not self.backdoor_fedosd_target_client:
            return train_data
        if len(train_data) == 0:
            return train_data

        rng = np.random.default_rng(self.backdoor_train_seed)
        poisoned_data = []
        poisoned_count = 0
        eligible_count = 0
        for x, y in train_data:
            label_value = int(y.item()) if torch.is_tensor(y) else int(y)
            if self.backdoor_fedosd_target_client:
                eligible = True
            else:
                eligible = not (
                    self.backdoor_train_exclude_target_label
                    and label_value == self.backdoor_train_target_label
                )
            if eligible:
                eligible_count += 1
            if self.backdoor_fedosd_target_client:
                do_poison = eligible
            else:
                do_poison = eligible and (rng.random() < self.backdoor_train_poison_rate)
            if do_poison:
                poisoned_x, trigger_applied = self._apply_backdoor_patch_to_sample(x)
                if not trigger_applied:
                    poisoned_data.append((x, y))
                    continue
                if torch.is_tensor(y):
                    poisoned_y = y.detach().clone()
                    poisoned_y.fill_(int(self._map_backdoor_label(label_value)))
                else:
                    poisoned_y = int(self._map_backdoor_label(label_value))
                poisoned_data.append((poisoned_x, poisoned_y))
                poisoned_count += 1
            else:
                poisoned_data.append((x, y))

        if not self._backdoor_train_poison_logged:
            effective_rate = 1.0 if self.backdoor_fedosd_target_client else self.backdoor_train_poison_rate
            mode_name = "fedosd_target_client" if self.backdoor_fedosd_target_client else "sample_ratio"
            print(
                f"客户端{self.id}训练期后门注毒启用: mode={mode_name}, label_mode={self.backdoor_label_mode}, rate={effective_rate:.3f}, "
                f"poisoned={poisoned_count}/{len(train_data)}, eligible={eligible_count}, "
                f"target_label={self.backdoor_train_target_label}"
            )
            self._backdoor_train_poison_logged = True
        return poisoned_data

    def _map_backdoor_label(self, label_value):
        if self.backdoor_label_mode == "class_shift":
            return int((int(label_value) + max(1, self.num_classes // 2)) % self.num_classes)
        return int(self.backdoor_train_target_label)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False, few_shot=self.few_shot)
        # 测试时不应该shuffle，避免数据与标签不匹配的问题
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=False)
        
    def set_parameters(self, model, copy_buffers=False):
        try:
            if copy_buffers:
                self.model.load_state_dict(model.state_dict(), strict=True)
                return

            # 默认只同步可训练参数，保留客户端本地buffer。
            # 对FedAvg在强非IID场景下，反复覆盖BN running stats会显著破坏收敛。
            source_params = dict(model.named_parameters())
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    param.data.copy_(source_params[name].data)
        except Exception as e:
            print(f"设置参数时出错: {e}")
            try:
                for new_param, old_param in zip(model.parameters(), self.model.parameters()):
                    old_param.data = new_param.data.clone()
            except Exception as e2:
                print(f"备用方法也失败: {e2}")
                raise e2

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
        return test_acc, test_num, auc

    def train_metrics(self):
        trainloader = self.load_train_data(apply_train_transform=False, shuffle=False, drop_last=False)
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num

    # def get_next_train_batch(self):
    #     try:
    #         # Samples a new batch for persionalizing
    #         (x, y) = next(self.iter_trainloader)
    #     except StopIteration:
    #         # restart the generator if the previous generator is exhausted.
    #         self.iter_trainloader = iter(self.trainloader)
    #         (x, y) = next(self.iter_trainloader)

    #     if type(x) == type([]):
    #         x = x[0]
    #     x = x.to(self.device)
    #     y = y.to(self.device)

    #     return x, y


    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))
