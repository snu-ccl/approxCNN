from __future__ import print_function
from tqdm import *

import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from models.resnet_cifar10 import *
from models.vgg_cifar10 import *
from models.utils_approx import rangeException

parser = argparse.ArgumentParser(description='Implementation of of Section V-A for `Precise Approximation of Convolutional Neural'
                                             + 'Networks for Homomorphically Encrypted Data.`')
parser.add_argument('--mode', default='inf', dest='mode', type=str,
                    help='Program mode. `train`: train randomly initialized model, '\
                         '`inf`: inference the proposed approximate deep learning model')
parser.add_argument('--gpu', default=0, dest='gpuid', type=int,
                    help='ID of GPU that is used for training and inference.')
parser.add_argument('--backbone', default='resnet20', dest='backbone', type=str,
                    help='Backbone model.')
parser.add_argument('--approx_method', default='proposed', dest='approx_method', type=str,
                    help='Method of approximating non-arithmetic operations. `proposed`: proposed composition of minimax polynomials, '\
                         '`square`: approximate ReLU as x^2, `relu_aq`: approximate ReLU as 2^-3*x^2+2^-1*x+2^-2. '\
                         'For `square` and `relu_aq`, we use exact max-pooling function.')
parser.add_argument('--batch_inf', default=128, dest='batch_inf', type=int,
                    help='Batch size for inference.')
parser.add_argument('--alpha', default=14, dest='alpha', type=int,
                    help='The precision parameter. Integers from 4 to 14 can be used.')
parser.add_argument('--B_relu', default=50.0, dest='B_relu', type=float,
                    help='The bound of approximation range for the approximate ReLU function.')
parser.add_argument('--B_max', default=50.0, dest='B_max', type=float,
                    help='The bound of approximation range for the approximate max-pooling function.')
parser.add_argument('--B_search', default=5.0, dest='B_search', type=float,
                    help='The size of the interval to find B such that all input values fall within the approximate region.')
parser.add_argument('--dataset_path', default='../dataset/CIFAR10', dest='dataset_path', type=str,
                    help='The path which contains the CIFAR10.')
parser.add_argument('--params_name', default='ours', dest='params_name', type=str,
                    help='The pre-trained parameters file name. Please omit `.pt`.')
args = parser.parse_args()

torch.cuda.set_device(args.gpuid)

params_path = ''.join(['./pretrained/cifar10/', args.backbone, '_', args.params_name, '.pt'])

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

cifar10_train = datasets.CIFAR10(args.dataset_path, train=True, download=True,
                               transform=transform_train)
loader_train = DataLoader(cifar10_train, batch_size=128)

cifar10_test = datasets.CIFAR10(args.dataset_path, train=False, download=True,
                              transform=transform_test)
loader_test = DataLoader(cifar10_test, batch_size=args.batch_inf)

dtype = torch.FloatTensor # the CPU datatype
gpu_dtype = torch.cuda.FloatTensor


def train(model, loss_fn, optimizer, scheduler, num_epochs=1):
    for epoch in range(num_epochs):
        print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        model.train()
        print('Training...')
        for t, (x, y) in tqdm(enumerate(loader_train)):
            torch.cuda.empty_cache()
            x_var = Variable(x.cuda())
            y_var = Variable(y.cuda().long())
            scores = model(x_var)
            loss = loss_fn(scores, y_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Evaluating...')
        test_acc = check_accuracy(model, loader_test) * 100
        print('Loss: %.4f, test accuracy: %.2f' % (loss.data, test_acc))
        scheduler.step()
        print('--------------------------')


def check_accuracy(model, loader, use_tqdm = False):
    num_correct = 0
    num_samples = 0
    model.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():
        for x, y in (tqdm(loader) if use_tqdm else loader):
            x_var = Variable(x.cuda())

            scores = model(x_var)
            _, preds = scores.data.cpu().max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    return acc


def checking_batchsize_inference(model):
    model.eval()
    with torch.no_grad():
        for t, (x, y) in enumerate(loader_test):
            x_var = Variable(x.cuda())
            try:
                _ = model(x_var)
            except rangeException as e:
                e.show()
                print('The validity of the batch size cannot be checked since the given B is to small.')
                print('Please give larger B_relu or B_max.')
                sys.exit("Terminated.")
            except:
                print('The batch size of INFERENCE seems to be large for your GPU.')
                print('Your current batch size is ' + str(args.batch_inf) + '. Try reducing `--batch_inf`.')
                sys.exit("Terminated.")
            break

approx_dict_list = [{'alpha': args.alpha, 'B': args.B_relu, 'type': args.approx_method},
                    {'alpha': args.alpha, 'B': args.B_max, 'type': args.approx_method}]

if args.backbone == 'resnet20':
    original_model = resnet20()
    approx_model = resnet20(approx_dict_list)
elif args.backbone == 'resnet32':
    original_model = resnet32()
    approx_model = resnet32(approx_dict_list)
elif args.backbone == 'resnet44':
    original_model = resnet44()
    approx_model = resnet44(approx_dict_list)
elif args.backbone == 'resnet56':
    original_model = resnet56()
    approx_model = resnet56(approx_dict_list)
elif args.backbone == 'resnet110':
    original_model = resnet110()
    approx_model = resnet110(approx_dict_list)
elif args.backbone == 'vgg11bn':
    original_model = vgg11_bn()
    approx_model = vgg11_bn(approx_dict_list)
elif args.backbone == 'vgg13bn':
    original_model = vgg13_bn()
    approx_model = vgg13_bn(approx_dict_list)
elif args.backbone == 'vgg16bn':
    original_model = vgg16_bn()
    approx_model = vgg16_bn(approx_dict_list)
elif args.backbone == 'vgg19bn':
    original_model = vgg19_bn()
    approx_model = vgg19_bn(approx_dict_list)

original_model.cuda()
approx_model.cuda()

if args.mode == 'train':
    if args.params_name == 'ours':
        print('Please set your own name or use another name rather than `ours` '
              'to avoid overwriting our pre-trained parameters used in the paper.')
        sys.exit("Terminated.")

    print("Training random initialized", args.backbone, "for CIFAR10")
    print("")

    loss_fn = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(original_model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-3, nesterov=True)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], last_epoch=-1)

    train(original_model, loss_fn, optimizer, scheduler, num_epochs=200)

    torch.save(original_model.state_dict(), params_path)
    print("Saved pre-trained parameters. Path:", params_path)


if args.mode == 'inf':
    original_model.load_state_dict(torch.load(params_path))
    approx_model.load_state_dict(torch.load(params_path))
    print("Used pre-trained parameter:", params_path)

    print('==========================')
    print("Inference the pre-trained original", args.backbone, "for CIFAR10")

    original_model.load_state_dict(torch.load(params_path))
    original_acc = check_accuracy(original_model, loader_test, use_tqdm=True) * 100

    print("Test accuracy: %.2f" % original_acc)

    print('==========================')
    print("Inference the approximate", args.backbone, "with same pre-trained parameters for CIFAR10")
    print("Precision parameter:", args.alpha)
    print("")

    # Check if given batch size is valid.
    checking_batchsize_inference(approx_model)

    while True:
        try:
            print("Trying to approximate inference...")
            print("with B_ReLU = %.1f," % approx_dict_list[0]['B'])
            print("and B_max = %.1f," % approx_dict_list[1]['B'])

            approx_acc = check_accuracy(approx_model, loader_test, use_tqdm=True) * 100

            print("Approximation success!")
            break

        except rangeException as e:
            e.show()
            if e.type == 'relu':
                print("We increase B_ReLU", args.B_search, "and try inference again.")
                approx_dict_list[0]['B'] += args.B_search
            elif e.type == 'max':
                print("We increase B_maxpooling", args.B_search, "and try inference again.")
                approx_dict_list[1]['B'] += args.B_search
            print('--------------------------')

    print("")
    print("Test accuracy: %.2f" % approx_acc)
    rate = (approx_acc - original_acc) / original_acc * 100
    print("Difference from the baseline: %.2f%%" % rate)

