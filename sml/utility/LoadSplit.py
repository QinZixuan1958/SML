import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets
import torch
import os
from torchvision.models import vit_b_16
from models.Nets import CNNCifar, ResNet, AlexNet
from data_process.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid, \
    office31_noniid, officehome_noniid, ina_noniid, imagenet_noniid, data_reading
from models.vit import vit_base_patch16_224
from options import args_parser
from torchvision import transforms
from PIL import Image
import json

# Initialize argument dictionary
args = args_parser()

def default_loader(path):
    return Image.open(path).convert('RGB')


def get_images_and_labels(dir_path):
    '''
    从图像数据集的根目录dir_path下获取所有类别的图像名列表和对应的标签名列表
    :param dir_path: 图像数据集的根目录
    :return: images_list, labels_list
    '''
    dir_path = Path(dir_path)
    classes = []  # 类别名列表

    for category in dir_path.iterdir():
        if category.is_dir():
            classes.append(category.name)
    images_list = []  # 文件名列表
    labels_list = []  # 标签列表

    for index, name in enumerate(classes):
        class_path = dir_path / name
        if not class_path.is_dir():
            continue
        for img_path in class_path.glob('*.JPEG'):
            images_list.append(str(img_path))
            labels_list.append(int(index))

    return images_list, labels_list

class MyDataset(Dataset):
    def __init__(self, img_path, transform=None):
        super(MyDataset, self).__init__()
        self.root = img_path

        self.txt_root = self.root  + 'data.txt'

        f = open(self.txt_root, 'r')
        data = f.readlines()

        imgs = []
        labels = []
        for line in data:
            line = line.rstrip()
            word = line.split()
            # print(word[0], word[1], word[2])
            # word[0]是图片名字.jpg  word[1]是label  word[2]是文件夹名，如sunflower
            imgs.append(os.path.join(self.root, word[2], word[0]))

            labels.append(word[1])
        self.img = imgs
        self.label = labels
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        img = self.img[item]
        label = self.label[item]

        img = Image.open(img).convert('RGB')

        # 此时img是PIL.Image类型   label是str类型

        if self.transform is not None:
            img = self.transform(img)

        label = np.array(label).astype(np.int64)
        label = torch.from_numpy(label)

        return img, label

class IGNAT_Loader(Dataset):

    def __init__(self, root, ann_file, transform=None, target_transform=None,
                 loader=default_loader):

        # assumes classes and im_ids are in same order
        super(IGNAT_Loader, self).__init__()
        # load annotations
        print('Loading annotations from: ' + os.path.basename(ann_file))
        with open(ann_file) as data_file:
            ann_data = json.load(data_file)

        # set up the filenames and annotations
        imgs = [aa['file_name'] for aa in ann_data['images']]
        im_ids = [aa['id'] for aa in ann_data['images']]

        if 'annotations' in ann_data.keys():
            # if we have class labels
            classes = [aa['category_id'] for aa in ann_data['annotations']]
        else:
            # otherwise dont have class info so set to 0
            classes = [0]*len(im_ids)

        idx_to_class = [ cc['supercategory'] for cc in ann_data['categories']]

        print('\t' + str(len(imgs)) + ' images')
        print('\t' + str(len(idx_to_class)) + ' classes')
        
        
        #print(idx_to_class)

        self.ids = im_ids
        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.idx_to_class = idx_to_class
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __len__(self):
        return len(self.imgs)
        
    def __getitem__(self, index):
        
        path = self.root + self.imgs[index]
        target = self.classes[index]
        im_id = self.ids[index]
        img = self.loader(path)
        classname = (path.split('/')[6])

        #if('Plantae' in str(self.imgs[index])):
        #  self.plan.append(im_id)
        #print(self.idx_to_class[index])
        #print(self.classss1)
        #print((path.split('/')[6]))
        #print(im_id)
        #print(target)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, im_id, classname


# function to load predefined datasets; can make custom dataloader here as well
def Load_Dataset(args):
    '''
    Function to load predefined datasets such as CIFAR-10, CIFAR-100 and MNIST via pytorch dataloader

    Declare Custom Dataloaders here if you want to change the dataset

    Also, the function to split training data among all the individuals is called from here

    '''

    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(args, dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(args, dataset_train, args.num_users)
        return dataset_train, dataset_test, dict_users
    elif args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        dataset_train = datasets.CIFAR10(data_dir, train=True, download=True,
                                         transform=apply_transform)

        dataset_test = datasets.CIFAR10(data_dir, train=False, download=True,
                                        transform=apply_transform)
        if args.iid:
            dict_users = cifar_iid(args, dataset_train, args.num_users)
        else:
            dict_users, group, subgroup1, dict_user_class, G = cifar_noniid(args, dataset_train, args.num_users)
        return dataset_train, dataset_test, dict_users, group, subgroup1, dict_user_class, G
    elif args.dataset == 'cifar100':
        trans_cifar = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR100('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR100('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(args, dataset_train, args.num_users)
        else:
            dict_users, group, subgroup1, dict_user_class, G = cifar_noniid(args, dataset_train, args.num_users)
        return dataset_train, dataset_test, dict_users, group, subgroup1, dict_user_class, G
    elif args.dataset == 'office31':
        transforms_office31 = transforms.Compose([
            transforms.Resize(32),  # 将图片短边缩放至256，长宽比保持不变：
            transforms.CenterCrop(32),  # 将图片从中心切剪成3*224*224大小的图片
            transforms.ToTensor()
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        dataset_amazon = MyDataset('/data/qzx/data/office31/amazon/', transform=transforms_office31)
        dataset_dslr = MyDataset('/data/qzx/data/office31/dslr/', transform=transforms_office31)
        dataset_webcam = MyDataset('/data/qzx/data/office31/webcam/', transform=transforms_office31)
        # dataset_test = datasets.CIFAR100('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(args, dataset_amazon, args.num_users)
        else:
            dict_users, dict_users_train, dict_users_test, group, subgroup1, dict_user_class, G = office31_noniid(args,
                                                                                                         dataset_amazon,
                                                                                                         args.num_users)
        return dataset_amazon, dataset_dslr, dataset_webcam, dict_users, dict_users_train, dict_users_test, group, subgroup1, dict_user_class, G
    elif args.dataset == 'officehome':
        transforms_office = transforms.Compose([
            transforms.Resize(32),    # 将图片短边缩放至256，长宽比保持不变：
            transforms.CenterCrop(32),    #将图片从中心切剪成3*224*224大小的图片
            transforms.ToTensor()
            #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        dataset_Art = MyDataset('/data/qzx/data/OfficeHome/Art/', transform=transforms_office)
        dataset_Clipart = MyDataset('/data/qzx/data/OfficeHome/Clipart/', transform=transforms_office)
        dataset_RealWorld = MyDataset('/data/qzx/data/OfficeHome/RealWorld/', transform=transforms_office)
        #dataset_test = datasets.CIFAR100('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(args, dataset_Art, args.num_users)
        else:
            dict_users, dict_users_train, dict_users_test, group, subgroup1, dict_user_class, G  = officehome_noniid(args, dataset_Art, args.num_users)
        return dataset_Art, dataset_Clipart, dataset_RealWorld, dict_users, dict_users_train, dict_users_test, group, subgroup1, dict_user_class, G
    elif args.dataset == 'ina17':
        dataset = IGNAT_Loader('/home/qinzixuan/data/ina17/', '/home/qinzixuan/data/ina17/train2017.json',
            transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std =[0.229, 0.224, 0.225])
        ]))
        dict_users_train, dict_users_test, dict_user_class = data_reading()
        #if args.iid:
        #    dict_users = cifar_iid(args, dataset, args.num_users)
        #else:
        #    dict_users_train, dict_users_test, dict_user_class = ina_noniid(args, dataset, args.num_users)
        return dataset, dict_users_train, dict_users_test, dict_user_class
    elif args.dataset == 'imagenet':
        dataset = datasets.ImageFolder(root='/home/qinzixuan/data/ImageNet2012/train/', transform=transforms.Compose([
            transforms.Resize(224),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]))

        # dataset_test = datasets.CIFAR100('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(args, dataset, args.num_users)
        else:

            dict_users_train, dict_users_test, dict_user_class = imagenet_noniid(args, dataset, args.num_users)
        return dataset, dict_users_train, dict_users_test, dict_user_class
    else:
        exit('Error: unrecognized dataset')


# function to use the model architectures present in Nets.py file present in models folder

def Load_Model(args):

    '''

    Function to load the required architecture (model) for federated learning

    '''

    if args.model == 'cnn' and args.dataset == 'office31':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'cifar100':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'AlexNet' and args.dataset == 'cifar100':
        net_glob = AlexNet(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'officehome':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'resnet' and args.dataset == 'ina17':
        """
        可以选择使用预训练模型进行训练，这个预训练模型是在imagenet1k上预训练的，论文里面使用的是预训练的模型
        net_glob = ResNet.ResNet18().cuda()
        weights_path = "/root/autodl-tmp/fedsocial_ina17_1/utility/resnet18_imagenet.pth"
        checkpoint = torch.load(os.path.join(weights_path), map_location={'cuda:5': 'cuda:0'})
        # 加载权重到模型
        net_glob.load_state_dict(checkpoint)
        fc_inDim = net_glob.linear.in_features
        # 修改为10个类别
        net_glob.linear = torch.nn.Linear(fc_inDim, 5089)
        net_glob = net_glob.cuda()
        """
        net_glob = ResNet.ResNet18().cuda()
    elif args.model == 'resnet' and args.dataset == 'imagenet':
        """
        可以选择使用预训练模型进行训练，这个预训练模型是在cifar100上预训练的，论文里面使用的是预训练的模型
        net_glob = ResNet.ResNet18().cuda()
        weights_path = "/root/...../resnet18_cifar80.pth"
        checkpoint = torch.load(os.path.join(weights_path), map_location={'cuda:5': 'cuda:0'})
        # 加载权重到模型
        net_glob.load_state_dict(checkpoint)
        fc_inDim = net_glob.linear.in_features
        # 修改为10个类别
        net_glob.linear = torch.nn.Linear(fc_inDim, 5089)
        net_glob = net_glob.cuda()
        """
        net_glob = ResNet.ResNet18().cuda()
    elif args.model == 'ResNet34':
        net_glob = ResNet.ResNet34(args=args).cuda()
    elif args.model == 'ResNet50':
        net_glob = ResNet.ResNet50(args=args).cuda()
    elif args.model == 'ResNet101':
        print('ResNet101')
        net_glob = ResNet.ResNet101(args=args).cuda()
    elif args.model == 'ResNet152':
        print('ResNet152')
        net_glob = ResNet.ResNet152(args=args).cuda()
    elif args.model == 'resnet' and args.dataset == 'imagenet':
        #net_glob = vit_base_patch16_224()
        net_glob = vit_b_16(weights=None).cuda()
    elif args.model == 'resnet' and args.dataset == 'ina17':
        net_glob = vit_b_16(pretrained=True)
        net_glob.heads = torch.nn.Sequential(torch.nn.Linear(net_glob.heads[0].in_features, 5089))
        net_glob = net_glob.cuda()


    return net_glob
        
