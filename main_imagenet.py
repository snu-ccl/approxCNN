from __future__ import print_function
from tqdm import *

import sys
import argparse
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from models.resnet_imagenet import *
from models.vgg_imagenet import *
from models.inception_imagenet import *
from models.googlenet_imagenet import *
from models.utils_approx import rangeException

parser = argparse.ArgumentParser(description='Implementation of Section V-B for `Precise Approximation of Convolutional Neural'
                                             + 'Networks for Homomorphically Encrypted Data.`')
parser.add_argument('--gpu', default=0, dest='gpuid', type=int,
                    help='ID of GPU that is used for training and inference.')
parser.add_argument('--backbone', default='resnet152', dest='backbone', type=str,
                    help='Backbone model.')
parser.add_argument('--alpha', default=14, dest='alpha', type=int,
                    help='The precision parameter. Integers from 4 to 14 can be used.')
parser.add_argument('--B_relu', default=100.0, dest='B_relu', type=float,
                    help='The bound of approximation range for the approximate ReLU function.')
parser.add_argument('--B_max', default=10.0, dest='B_max', type=float,
                    help='The bound of approximation range for the approximate max-pooling function.')
parser.add_argument('--B_search', default=10.0, dest='B_search', type=float,
                    help='The size of the interval to find B such that all input values fall within the approximate region.')
parser.add_argument('--dataset_path', default='../dataset/imagenet', dest='dataset_path', type=str,
                    help='The path which contains the ImageNet.')
parser.add_argument('--batch_train', default=32, dest='batch_train', type=int,
                    help='Batch size for train (or retrain).')
parser.add_argument('--batch_inf', default=16, dest='batch_inf', type=int,
                    help='Batch size for inference.')
args = parser.parse_args()

torch.cuda.set_device(args.gpuid)

traindir = args.dataset_path + '/train'
valdir = args.dataset_path + '/val'
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))


loader_train = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_train, shuffle=True, num_workers=4, pin_memory=True)

test_dataset = datasets.ImageFolder(
    valdir,
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))

loader_test = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.batch_inf, shuffle=False, num_workers=4, pin_memory=True)

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
            torch.cuda.empty_cache()
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

approx_dict_list = [{'alpha': args.alpha, 'B': args.B_relu, 'type': 'proposed'},
                    {'alpha': args.alpha, 'B': args.B_max, 'type': 'proposed'}]

if args.backbone == 'resnet18':
    original_model = resnet18(pretrained=True)
    approx_model = resnet18(approx_dict_list, pretrained=True)
elif args.backbone == 'resnet34':
    original_model = resnet34(pretrained=True)
    approx_model = resnet34(approx_dict_list, pretrained=True)
elif args.backbone == 'resnet50':
    original_model = resnet50(pretrained=True)
    approx_model = resnet50(approx_dict_list, pretrained=True)
elif args.backbone == 'resnet101':
    original_model = resnet101(pretrained=True)
    approx_model = resnet101(approx_dict_list, pretrained=True)
elif args.backbone == 'resnet152':
    original_model = resnet152(pretrained=True)
    approx_model = resnet152(approx_dict_list, pretrained=True)
elif args.backbone == 'vgg11bn':
    original_model = vgg11_bn(pretrained=True)
    approx_model = vgg11_bn(approx_dict_list, pretrained=True)
elif args.backbone == 'vgg13bn':
    original_model = vgg13_bn(pretrained=True)
    approx_model = vgg13_bn(approx_dict_list, pretrained=True)
elif args.backbone == 'vgg16bn':
    original_model = vgg16_bn(pretrained=True)
    approx_model = vgg16_bn(approx_dict_list, pretrained=True)
elif args.backbone == 'vgg19bn':
    original_model = vgg19_bn(pretrained=True)
    approx_model = vgg19_bn(approx_dict_list, pretrained=True)
elif args.backbone == 'googlenet':
    original_model = googlenet(pretrained=True)
    approx_model = googlenet(approx_dict_list, pretrained=True)
elif args.backbone == 'inception_v3':
    original_model = inception_v3(pretrained=True)
    approx_model = inception_v3(approx_dict_list, pretrained=True)


original_model.cuda()
approx_model.cuda()


print("Used pre-trained parameter: default (PyTorch)")

print('==========================')
print("Inference the pre-trained original", args.backbone, "for ImageNet")

original_acc = check_accuracy(original_model, loader_test, use_tqdm=True) * 100

print("Test accuracy: %.2f" % original_acc)

print('==========================')
print("Inference the approximate", args.backbone, "with same pre-trained parameters for ImageNet")
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
            print("We increase B_max", args.B_search, "and try inference again.")
            approx_dict_list[1]['B'] += args.B_search
        print('--------------------------')

print("")
print("Test accuracy: %.2f" % approx_acc)
rate = (approx_acc - original_acc) / original_acc * 100
print("Difference from the baseline: %.2f%%" % rate)

