import torch
import torch.nn as nn
import numpy as np
import math
import itertools

class rangeException(Exception):
    def __init__(self, type, val):
        self.type = type
        self.val = val

    def show(self):
        if self.type == 'relu':
            print("STOP! There is an input value", self.val.item(), "for the approximate ReLU function.")
        elif self.type == 'max':
            print("STOP! There is an input value", self.val.item(), "for the approximate max-pooling function.")


# Approximate ReLU modules

class ReLU_approx_module(nn.Module):
    def __init__(self, relu_dict):
        super(ReLU_approx_module, self).__init__()
        self.relu_dict = relu_dict

    def forward(self, x):
        return ReLU_approx(x, self.relu_dict)

class ReLU_square_module(nn.Module):
    def __init__(self):
        super(ReLU_square_module, self).__init__()

    def forward(self, x):
        return x ** 2

class ReLU_AQ_module(nn.Module):
    def __init__(self):
        super(ReLU_AQ_module, self).__init__()

    def forward(self, x):
        return (x**2 / 8.0 + x / 2.0 + 1 / 4.0)


def ReLU_maker(relu_dict):
    if relu_dict['type'] == 'pure':
        return nn.ReLU(inplace=True)
    elif relu_dict['type'] == 'proposed':
        return ReLU_approx_module(relu_dict)
    elif relu_dict['type'] == 'square':
        return ReLU_square_module()
    elif relu_dict['type'] == 'relu_aq':
        return ReLU_AQ_module()
    return 0


# Approximate max-pooling modules

class maxpool_approx_module(nn.Module):
    def __init__(self, maxpool_dict, maxpool_basic_dict):
        super(maxpool_approx_module, self).__init__()
        self.maxpool_dict = maxpool_dict
        self.maxpool_basic_dict = maxpool_basic_dict

    def forward(self, x):
        return maxpool_approx(x, self.maxpool_dict, self.maxpool_basic_dict)


def maxpool_maker(maxpool_dict, maxpool_basic_dict):
    if maxpool_dict['type'] == 'proposed':
        return maxpool_approx_module(maxpool_dict, maxpool_basic_dict)
    else:
        return nn.MaxPool2d(kernel_size=maxpool_basic_dict['kernel_size'],
                            stride=maxpool_basic_dict['stride'],
                            padding=maxpool_basic_dict['padding'], )
    return 0


# Functions to evaluate the proposed approximate ReLU and max-pooling.

def poly_eval(x, coeff):
    coeff = torch.tensor(coeff).cuda()
    if len(x.size()) == 2:
        return torch.sum(x[:,:,None]**torch.arange(coeff.size(0)).cuda()[None,None,:]*coeff, dim=-1)
    elif len(x.size()) == 4:
        return torch.sum(x[:,:,:,:,None]**torch.arange(coeff.size(0)).cuda()[None,None,None,None,:]*coeff, dim=-1)


def sgn_approx(x, relu_dict):
    alpha = relu_dict['alpha']
    B = torch.tensor(relu_dict['B']).cuda().double()

    # Get degrees
    f = open('./degreeResult/deg_' + str(alpha) + '.txt')
    readed = f.readlines()
    comp_deg = [int(i) for i in readed]

    # Get coefficients
    f = open('./coeffResult/coeff_' + str(alpha) + '.txt')
    coeffs_all_str = f.readlines()
    coeffs_all = [torch.tensor(np.double(i), dtype=torch.double) for i in coeffs_all_str]
    i = 0

    if (torch.sum(torch.abs(x) > B) != 0.0):
        max_val = torch.max(torch.abs(x))
        raise rangeException('relu', max_val)

    x.double()
    x = torch.div(x, B)

    for deg in comp_deg:
        coeffs_part = coeffs_all[i:(i + deg + 1)]
        x = poly_eval(x, coeffs_part)
        torch.cuda.empty_cache()
        i += deg + 1

    return x.float()

def ReLU_approx(x, relu_dict):
    sgnx = sgn_approx(x, relu_dict)
    return x * (1.0 + sgnx) / 2.0

def max_approx(x, y, maxpool_dict):
    relu_dict = maxpool_dict.copy()
    relu_dict['B'] = 1.0
    sgn = sgn_approx(x-y, relu_dict)
    return (x+y + (x-y)*sgn) / 2.0

def max_approx_ker2(x, maxpool_dict):
    alpha = maxpool_dict['alpha']
    B = maxpool_dict['B']
    scaling_factor = (1.0 - 4.0 * (0.5**alpha)) / (B * 2.0)

    # Check
    for i in range(4):
        if (torch.sum(torch.abs(x[i]) > B) != 0.0):
            max_val = torch.max(torch.abs(x[i]))
            raise rangeException('max', max_val)

        x[i].double()
        x[i] = x[i] * scaling_factor + 0.5

    step1_1 = max_approx(x[0], x[1], maxpool_dict)
    step1_2 = max_approx(x[2], x[3], maxpool_dict)

    res = max_approx(step1_1, step1_2, maxpool_dict)

    res = (res - 0.5) / scaling_factor

    return res.float()

def max_approx_ker3(x, maxpool_dict):
    alpha = maxpool_dict['alpha']
    B = maxpool_dict['B']
    scaling_factor = (1.0 - 6.0 * (0.5**alpha)) / (B * 2.0)

    # Check
    for i in range(9):
        if (torch.sum(torch.abs(x[i]) > B) != 0.0):
            max_val = torch.max(torch.abs(x[i]))
            raise rangeException('max', max_val)

        x[i].double()
        x[i] = x[i] * scaling_factor + 0.5

    step1_1 = max_approx(x[0], x[1], maxpool_dict)
    step1_2 = max_approx(x[2], x[3], maxpool_dict)
    step2_1 = max_approx(step1_1, step1_2, maxpool_dict)

    step1_3 = max_approx(x[4], x[5], maxpool_dict)
    step1_4 = max_approx(x[6], x[7], maxpool_dict)
    step2_2 = max_approx(step1_3, step1_4, maxpool_dict)

    step3 = max_approx(step2_1, step2_2, maxpool_dict)

    res = max_approx(step3, x[8], maxpool_dict)

    res = (res - 0.5) / scaling_factor

    return res.float()

def maxpool_approx(x, maxpool_dict, maxpool_basic_dict):
    kernel_size = maxpool_basic_dict['kernel_size']
    stride = maxpool_basic_dict['stride']
    padding = maxpool_basic_dict['padding']
    ceil_check = False
    if 'ceil' in maxpool_basic_dict:
        ceil_check = maxpool_basic_dict['ceil']

    if ceil_check:
        N, C, W_init, H_init = x.size()
        W_out = math.ceil((W_init + 2*padding - kernel_size)/stride + 1)
        W_out_floor = math.floor((W_init + 2*padding - kernel_size)/stride + 1)
        H_out = math.ceil((H_init + 2*padding - kernel_size)/stride + 1)
        H_out_floor = math.floor((H_init + 2*padding - kernel_size)/stride + 1)
        pad_right = W_out - W_out_floor
        pad_bottom = H_out - H_out_floor
        m = nn.ZeroPad2d((padding, padding+pad_right, padding, padding+pad_bottom))
    else:
        m = nn.ZeroPad2d(padding)

    padded_x = m(x)
    N, C, W, H = padded_x.size()

    extracted_tensors = []

    for i in range(kernel_size):
        for j in range(kernel_size):
            i_num = (W - kernel_size) // stride + 1
            j_num = (H - kernel_size) // stride + 1
            mask = torch.zeros(size=padded_x.size())
            for (i_, j_) in itertools.product(range(i_num), range(j_num)):
                mask[:,:,(i+stride*i_), (j+stride*j_)] = 1
            extracted_tensors.append(torch.reshape(padded_x[mask.bool()], (N, C, i_num, j_num)))

    if kernel_size == 3:
        return max_approx_ker3(extracted_tensors, maxpool_dict)
    if kernel_size == 2:
        return max_approx_ker2(extracted_tensors, maxpool_dict)
    return 0