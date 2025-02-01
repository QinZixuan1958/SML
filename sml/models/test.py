from sklearn.metrics import f1_score
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
import numpy as np
from torch.utils.data import DataLoader, Dataset
from prefetch_generator import BackgroundGenerator


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


def test_client(args, dataset, test_idx, net):
    '''

    Test the performance of the client models on their datasets

    Parameters:

        net (state_dict) : Client Model

        datatest (dataset) : The data on which we want the performance of the model to be evaluated

        args (dictionary) : The list of arguments defined by the user

        test_idx (list) : List of indices of those samples from the actual complete dataset that are there in the local dataset of this client

    Returns:

        accuracy (float) : Percentage accuracy on test set of the model

        test_loss (float) : Cumulative loss on the data

    '''
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # data_loader = dataset
    data_loader = DataLoader(DatasetSplit(dataset, test_idx), batch_size=args.local_bs, num_workers=8, pin_memory=True)
    net.to(args.device)
    net.eval()
    # print (test_data)
    test_loss = 0
    correct = 0

    with torch.no_grad():

        for idx, (data, target, im_id, classname) in enumerate(data_loader):
            if args.gpu != -1:
                data, target = data.cuda(), target.cuda()

            log_probs = net(data)

            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]

            correct += y_pred.eq(target.data.view_as(y_pred)).float().cpu().sum()

            del data, target, log_probs
            torch.cuda.empty_cache()
        test_loss /= len(data_loader.dataset)
        accuracy = 100.00 * correct / len(data_loader.dataset)

        return accuracy, test_loss
