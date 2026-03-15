import copy
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from torchvision import transforms
from utils.data_utils import read_client_data


_VISION_DATASET_MARKERS = (
    "MNIST",
    "FashionMNIST",
    "Cifar",
    "CINIC10",
    "CINIC-10",
    "STL10",
    "GTSRB",
    "TinyImagenet",
    "Tiny-ImageNet",
    "Digit5",
    "Omniglot",
)


def _contains_any(dataset_name, markers):
    return any(marker in dataset_name for marker in markers)


def _is_vision_dataset(dataset_name):
    return _contains_any(dataset_name, _VISION_DATASET_MARKERS)


def _build_train_transform(dataset_name):
    if any(tag in dataset_name for tag in ("Cifar", "CINIC10", "CINIC-10")):
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
        ])
    if "STL10" in dataset_name:
        return transforms.Compose([
            transforms.RandomCrop(96, padding=12, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
        ])
    if any(tag in dataset_name for tag in ("TinyImagenet", "Tiny-ImageNet")):
        return transforms.Compose([
            transforms.RandomCrop(64, padding=8, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
        ])
    return None


class _TransformDataset(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y


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
        self.sync_full_state = bool(getattr(args, "sync_client_buffers", True))
        self.train_data_augmentation = bool(getattr(args, "train_data_augmentation", True))
        self.optimizer_momentum = float(getattr(args, "optimizer_momentum", 0.9))
        self.weight_decay = float(getattr(args, "weight_decay", 5e-4))
        self.optimizer_nesterov = bool(getattr(args, "optimizer_nesterov", True))
        self.train_transform = _build_train_transform(self.dataset) if self.train_data_augmentation else None

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
        self.optimizer = self._build_optimizer()
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay

    def _build_optimizer(self):
        if _is_vision_dataset(self.dataset):
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=self.optimizer_momentum,
                weight_decay=self.weight_decay,
                nesterov=self.optimizer_nesterov,
            )
        return torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def load_train_data(self, batch_size=None, augment=False):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True, few_shot=self.few_shot)
        if augment and self.train_transform is not None:
            train_data = _TransformDataset(train_data, self.train_transform)
        return DataLoader(train_data, batch_size, drop_last=False, shuffle=True)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False, few_shot=self.few_shot)
        # 测试时不应该shuffle，避免数据与标签不匹配的问题
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=False)
        
    def set_parameters(self, model, copy_buffers=None):
        try:
            if copy_buffers is None:
                copy_buffers = self.sync_full_state
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
        trainloader = self.load_train_data()
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
