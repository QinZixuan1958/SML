import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
import numpy as np
from torch.utils.data import DataLoader, Dataset
from prefetch_generator import BackgroundGenerator


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class."""

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

class IndividualTester:
    def __init__(self, args, dataset, net, test_idx):
        """
        Initialize the individual model tester

        Parameters:
            args (dict) : User-defined parameters
            dataset (Dataset) : Dataset
            net (nn.Module) : Individual model
            test_idx (list) : Indices of the test samples
        """
        self.args = args
        self.dataset = dataset
        self.test_idx = test_idx
        self.net = net

    def prepare_data_loader(self):
        """Prepare DataLoader for the individual testing data."""
        testloader = DataLoaderX(DatasetSplit(self.dataset, self.test_idx),
                                  batch_size=self.args.local_bs, num_workers=8, pin_memory=True, shuffle=True)
        return testloader

    def compute_f1_score(self, predictions, true_labels):
        """
        Compute the weighted F1 score

        Parameters:
            predictions (array-like) : Predicted labels
            true_labels (array-like) : True labels

        Returns:
            f1 (float) : Weighted F1 score
        """
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        f1 = f1_score(true_labels, predictions, average='weighted')
        return f1

    def test_individual(self):
        """
        Test the performance of the individual model

        Parameters:
            test_idx (list) : Indices of the test samples in the local dataset

        Returns:
            accuracy (float) : Test set accuracy
            test_loss (float) : Test set loss
            f1 (float) : Weighted F1 score
        """
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        test_loader = self.prepare_data_loader()

        self.net.to(self.args.device)
        self.net.eval()

        test_loss = 0
        correct = 0
        predictions = []
        true_labels = []

        with torch.no_grad():
            for idx, (data, target, im_id, classname) in enumerate(test_loader):
                if self.args.gpu != -1:
                    data, target = data.to(self.args.device), target.to(self.args.device)

                log_probs = self.net(data)
                test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()

                y_pred = log_probs.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(target.data.view_as(y_pred)).float().cpu().sum()

                predictions.extend(y_pred.cpu().numpy())
                true_labels.extend(target.cpu().numpy())

                del data, target, log_probs
                torch.cuda.empty_cache()

        test_loss /= len(test_loader.dataset)
        accuracy = 100.00 * correct / len(test_loader.dataset)

        # Compute the F1 score
        f1 = self.compute_f1_score(predictions, true_labels)

        del predictions, true_labels

        return accuracy, test_loss, f1
