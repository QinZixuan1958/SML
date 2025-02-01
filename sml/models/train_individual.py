import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import copy
from prefetch_generator import BackgroundGenerator
import torch.nn.functional as F
import numpy as np
import warnings

warnings.filterwarnings("ignore")


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label, im_id, classname = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label), torch.tensor(im_id), classname


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class IndividualTrainer:
    def __init__(self, args, dataset, train_data_users, local_net, now_model, global_ep, performance):
        self.args = args
        self.dataset = dataset
        self.train_data_users = train_data_users
        self.local_net = local_net
        self.now_model = now_model
        self.global_ep = global_ep
        self.performance = performance
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.best_model_wts = None
        self.best_acc = 0.0

    def set_optimizer(self, net):
        """Set the optimizer for the individual model."""
        self.optimizer = optim.Adam(net.parameters(), lr=self.args.lr, weight_decay=1e-5)

    def prepare_data_loader(self):
        """Prepare DataLoader for the individual training data."""
        idxs_train = self.train_data_users
        trainloader = DataLoaderX(DatasetSplit(self.dataset, idxs_train),
                                  batch_size=self.args.local_bs, num_workers=8, pin_memory=True, shuffle=True)
        return trainloader

    def get_dynamic_loss(self, outputs, labels):
        """Compute dynamic loss based on the epoch."""

        R, pm, sm = self._get_proximity_params()
        alpha = self._get_alpha()
        
        loss = (1 - alpha) * self.criterion(outputs, labels) + alpha * R * torch.norm(pm - sm, p=2)
        return loss

    def get_init_loss(self, outputs, labels):
        """Compute loss."""

        
        loss = self.criterion(outputs, labels)
        return loss        

    def _get_proximity_params(self):
        """
        Retrieve proximity parameters based on epoch.
        R is the regularized strength of the regular constraints, which needs to be adjusted differently for different tasks,
        with the aim of allowing social knowledge constraints to have an effective role on local losses.
        We give some suggested values here: R_ina17 = 2, R_imagenet = 5, R_31 = 10, R_cifar100 = 1.5.
        If the regular influence is too large or too small, it can be adjusted.
        """

        R = 0.06
        pm = torch.cat([p.view(-1) for p in list(self.local_net.parameters())], dim=0)
        sm = torch.cat([p.view(-1) for p in list(self.now_model.parameters())], dim=0)

        return R, pm, sm

    def _get_kl_divergence(self, data):
        """
        Calculate the KL divergence between the model outputs of `self.local_net` and `self.now_model`.
        This serves as a regularization term to enforce similarity between the two models.
        """

        # Get outputs from both models
        local_output = self.local_net(data)  # Ensure self.data is available or passed
        now_output = self.now_model(data)  # Ensure self.data is available or passed

        # Apply softmax to the outputs to get probabilities (log_softmax for numerical stability)
        local_probs = F.softmax(local_output, dim=1)  # Assuming classification, dim=1 is class dimension
        now_probs = F.softmax(now_output, dim=1)

        # Calculate the KL divergence between the local and now model outputs
        kl_div = F.kl_div(F.log_softmax(local_output, dim=1), now_probs, reduction='batchmean')

        return kl_div

    def _get_proximity_params_part(self):
        """
        Retrieve proximity parameters based on epoch.
        This is to learn only a part of the parameters of the aggregation model.
        Learning the shallow layer of the model can improve the feature extraction ability and converge faster;
        learning the deep layer can improve the classification ability and converge slower.
        """
        if self.global_ep % 3 == 0:
            R = 5.5
            pm = torch.cat([p.view(-1) for p in list(self.local_net.parameters())[0:54]], dim=0)
            sm = torch.cat([p.view(-1) for p in list(self.now_model.parameters())[0:54]], dim=0)
        else:
            R = 1.5
            pm = torch.cat([p.view(-1) for p in list(self.local_net.parameters())[20:]], dim=0)
            sm = torch.cat([p.view(-1) for p in list(self.now_model.parameters())[20:]], dim=0)
        return R, pm, sm

    def _get_alpha(self):
        """
        Compute the dynamic 'alpha' parameter based on the epoch.
        For different tasks, alpha needs to be adapted and is given here only for reference
        alpha controls the degree of social learning, with larger alpha resulting in a larger share of social learning.
        alpha = 0.3-0.5
        """
        if self.args.dataset == 'ina17':
            alpha_sl = self.args.alpha + 0.01 * int(self.global_ep / 100)
            return alpha_sl
        elif self.args.dataset == 'imagenet':
            alpha_sl = self.args.alpha + 0.05 * int(self.global_ep / 50)
            return alpha_sl
        elif self.args.dataset == 'cifar100':
            alpha_sl = self.args.alpha + 0.01 * int(self.global_ep / 50)
            return alpha_sl
        elif self.args.dataset == 'office31':
            alpha_sl = self.args.alpha + 0.08 * int(self.global_ep / 50)
            return alpha_sl

    def _get_alpha_performance(self):
        """
        Compute the dynamic 'alpha' parameter based on the epoch.
        For different tasks, alpha needs to be adapted and is given here only for reference
        alpha controls the degree of social learning, with larger alpha resulting in a larger share of social learning.
        alpha = 0.3-0.5
        Adjust the percentage of different social learning according to the performance of different individuals.
        """

        if self.performance == 1:
            alpha_new = self.args.alpha
        else:
            alpha_new = self.args.alpha + 0.1

        if self.args.dataset == 'ina17':
            alpha_sl = alpha_new + 0.02 * int(self.global_ep / 50)
            return alpha_sl
        elif self.args.dataset == 'imagenet':
            alpha_sl = alpha_new + 0.05 * int(self.global_ep / 50)
            return alpha_sl
        elif self.args.dataset == 'cifar100':
            alpha_sl = alpha_new + 0.01 * int(self.global_ep / 50)
            return alpha_sl
        elif self.args.dataset == 'office31':
            alpha_sl = alpha_new + 0.08 * int(self.global_ep / 50)
            return alpha_sl

    def train(self):
        """Train a individual model."""
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        train_loader = self.prepare_data_loader()
        self.set_optimizer(self.local_net)

        self.local_net.train()
        accumulation_steps = 1
        sum_loss = []
        sum_acc = []

        for iter in range(self.args.local_ep):
            epoch_loss = 0.0
            running_corrects = 0.0
            # (images, labels, _, _) for ina17; (images, labels) for others
            for batch_idx, (images, labels, _, _) in enumerate(train_loader):
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                self.optimizer.zero_grad()
                outputs = self.local_net(images)

                # Compute loss
                loss = self.get_dynamic_loss( outputs, labels)
                loss = loss / accumulation_steps  # Gradient accumulation
                loss.backward()

                # Update model every 'accumulation_steps' steps
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                epoch_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)

                # Clear memory
                del images, labels, outputs, loss, preds
                torch.cuda.empty_cache()

            epoch_loss = epoch_loss / len(train_loader.dataset)
            epoch_acc = running_corrects.double() / len(train_loader.dataset)

            if epoch_acc >= self.best_acc:
                self.best_acc = epoch_acc
                self.best_model_wts = copy.deepcopy(self.local_net.state_dict())

            sum_loss.append(epoch_loss)
            sum_acc.append(epoch_acc)

        return sum(sum_loss) / len(sum_loss), sum(sum_acc) / len(sum_acc), self.best_model_wts

    def init_train(self):
        """Train a individual model."""
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        train_loader = self.prepare_data_loader()
        self.set_optimizer(self.local_net)

        self.local_net.train()
        accumulation_steps = 1
        sum_loss = []
        sum_acc = []

        for iter in range(5):
            epoch_loss = 0.0
            running_corrects = 0.0
            # (images, labels, _, _) for ina17; (images, labels) for others
            for batch_idx, (images, labels, _, _) in enumerate(train_loader):
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                self.optimizer.zero_grad()
                outputs = self.local_net(images)

                # Compute loss
                loss = self.get_init_loss(outputs, labels)
                loss = loss / accumulation_steps  # Gradient accumulation
                loss.backward()

                # Update model every 'accumulation_steps' steps
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                epoch_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)

                # Clear memory
                del images, labels, outputs, loss, preds
                torch.cuda.empty_cache()

            epoch_loss = epoch_loss / len(train_loader.dataset)
            epoch_acc = running_corrects.double() / len(train_loader.dataset)

            if epoch_acc >= self.best_acc:
                self.best_acc = epoch_acc
                self.best_model_wts = copy.deepcopy(self.local_net.state_dict())

            sum_loss.append(epoch_loss)
            sum_acc.append(epoch_acc)

        return sum(sum_loss) / len(sum_loss), sum(sum_acc) / len(sum_acc), self.best_model_wts        

    def update_model_weights(self):
        """Return the best model weights."""
        self.local_net.load_state_dict(self.best_model_wts)
        return self.local_net.state_dict()


