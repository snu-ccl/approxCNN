# Precise Approximation of Convolutional Neural Networks for Homomorphically Encrypted Data

These source files are the implementation for the simulations in Section V-A and Section V-B of "Precise Approximation of Convolutional Neural Networks for Homomorphically Encrypted Data", which give the result of the inference of approximate standard deep learning models for the CIFAR-10 and the ImageNet.  
## Requirements

We simulate the program with the following packages.

* python 3.7.10
* numpy 1.19.2
* torch 1.8.0
* torchvision 0.9.0
* Cython 0.29.22
* Pillow 8.1.2
* requests
* tqdm

For conda users, `requirements.yaml` is available.

## Simulations for Section V-A (CIFAR-10) 

### Inference the approximate model

The inference of the approximation of the standard deep learning model for CIFAR-10 can be performed by `main_cifar10.py`.
The code can reproduce the exact result of the simulation in Section V-A (i.e., Table IV).
First, you have to download the pre-trained parameter that we used by following the command below at the main directory.

```
./downloadparams.sh
```

As an example for reproducing our results, the command below gives an inference result of the approximate ResNet20 by the proposed composition of minimax polynomials with the precision parameter 14.
For inference, the argument `--mode` must be set to `inf`.
The code assumes that the CIFAR-10 dataset is located in `../dataset/cifar10`. You can change the directory of the dataset by setting the argument `--dataset_path`.

```
python main_cifar10.py --mode inf --backbone resnet20 --alpha 14
```

Also, you can customize the precision parameter, approximation range, batch size, and pre-trained parameters of the inference by setting proper arguments of the code.
Please check the details below of the arguments.

- `--gpu`(default: `0`): ID of GPU that is used for training and inference.
- `--backbone`(default: `resnet20`): Backbone model. For CIFAR-10, the available arguments are followings: 
  - ResNet: `resnet20` `resnet32` `resnet44` `resnet56` `resnet110`
  - VGGNet (with batch normalization): `vgg11bn` `vgg13bn` `vgg16bn` `vgg19bn`
- `--approx_method`(default: `proposed`): Method of approximating non-arithmetic operations. For `square` and `relu_aq`, we use exact max-pooling function.
  - `proposed`: Proposed composition of minimax polynomials.
  - `square`: approximate ReLU as $x^2$, 
  - `relu_aq`: approximate ReLU as $2^-3x^2+2^-1x+2^-2$.
- `--batch_inf`(default: `128`): Batch size for inference. The program terminates if the batch size is small to execute inference.
- `--alpha`(default: `14`): The precision parameter $\alpha$ for approximation. Integers from 4 to 14 can be used.
- `--dataset_path`(default: `../dataset/CIFAR10`): The path of the directory which contains the CIFAR-10.

**Note.** The approximate polynomials for ReLU and max-pooling functions require more GPU memories than the original functions.
Therefore, the batch size should be determined in consideration of the GPU environment.


### Training mode for achieving another pre-trained parameter

One would like to check the inference result for another pre-trained parameter.
The file `main_cifar10.py` includes training mode for ResNet and VGGNet with the optimizer and learning schedule we used in Section V-A.
For example, if you want to achieve your own pre-trained parameters from another random initialization for ResNet20, follow the command below.
You have to set the argument `--mode` to `train`, and give a name of parameter name with setting the argument `--params_name`.

```
python main_cifar10.py --mode train --backbone resnet20 --params_name new
```

After training, the pre-trained parameters will be saved at `./pretrained/cifar10/`, with filename `resnet20_new.pt`.

With your own pre-trained parameters, an approximate inference can be done by running `main_cifar.py` setting the argument `--mode` to `inf`, same as before.
You can customize the precision parameter, approximation range, batch size, and pre-trained parameters of the inference by setting proper arguments of the code.

**Note.** For the proposed method, we have to determine appropriate approximation ranges $[-B_relu, B_relu]$, and $[-B_max, B_max]$, for the approximate ReLU and max-pooling function, respectively.
If $B_relu$ and $B_max$ are not large enough, there may be an input value that does not fall in the approximation range for those approximate functions.
On the other hand, if $B_relu$ and $B_max$ are too large, the error of the inference increases (see Lemma 1 (b) and (c)). 
To search an appropriate $B_relu$ and $B_max$, we provide the argument `--B_search`. 
During the inference, if the code detects the input value which does not fall in the approximation range, the code terminates the inference.
Then, the code increases the value of $B_relu$ (or $B_max$) by `B_search` and starts a new inference from the beginning.
The value of $B_relu$ (or $B_max$) will continue to increase until the code finishes the whole inference.

- `--gpu`(default: `0`): ID of GPU that is used for training and inference.
- `--backbone`(default: `resnet20`): Backbone model. For CIFAR-10, the available arguments are followings: 
  - ResNet: `resnet20` `resnet32` `resnet44` `resnet56` `resnet110`
  - VGGNet (with batch normalization): `vgg11bn` `vgg13bn` `vgg16bn` `vgg19bn`
- `--approx_method`(default: `proposed`): Method of approximating non-arithmetic operations. For `square` and `relu_aq`, we use exact max-pooling function.
  - `proposed`: Proposed composition of minimax polynomials.
  - `square`: approximate ReLU as $x^2$, 
  - `relu_aq`: approximate ReLU as $2^-3*x^2+2^-1*x+2^-2$.
- `--batch_inf`(default: `128`): Batch size for inference. The program terminates if batch size is small to execute inference.
- `--alpha`(default: `14`): The precision parameter $\alpha$ for approximation. Integers from 4 to 14 can be used.
- `--B_relu`(default: `50.0`): The value of $B$, where $[-B,B]$ is the approximation range for the approximate ReLU function. 
- `--B_max`(default: `50.0`): The value of $B$, where $[-B,B]$ is the approximation range for the approximate max-pooling function.
- `--B_search`(default: `5.0`): The step size for searching $B$ such that all input values fall within the approximation range.
- `--dataset_path`(default: `../dataset/CIFAR10`): The path of directory which contains the CIFAR-10.
- `--params_name`: The name of the pre-trained parameter file that you set at the training step. If this argument is not set, the code inference the proposed approximate model with pre-trained parameters we used.
If this argument is set, the code loads the pre-trained parameter `./pretrained/cifar10/<<backbone>>_<<params_name>.pt`.

## Simulation for Section V-B (ImageNet)

We implemented the inference code for ImageNet in the file `main_imagenet.py`. You can reproduce the exact result of the simulation in Section V-B. (i.e., Table V.)
Unlike CIFAR-10, it takes a lot of time to achieve pre-trained parameters for a standard deep learning model for ImageNet. 
Therefore, we did not implement the training mode for ImageNet, and we apply the pre-trained parameter given by PyTorch.
For example, the inference result for ResNet-152 in Table V can be reproduced by following the command below.
Similarly, the code assumes that the ImageNet dataset is located in `../dataset/imagenet`. You can change the directory of the dataset by setting the argument `--dataset_path`.

```
python main_imagenet.py --backbone resnet152
```


The options for `main_imagenet.py` are almost the same as for CIFAR-10, but the available backbone models are slightly different. 
The precision parameter or approximation range can be customized here also.
Please check the details below of the arguments and set the proper path of the dataset.

- `--backbone`(default: `resnet152`): Backbone model. For ImageNet, the available arguments are followings: 
  - ResNet: `resnet18` `resnet34` `resnet50` `resnet101` `resnet152`
  - VGGNet (with batch normalization): `vgg11bn` `vgg13bn` `vgg16bn` `vgg19bn`
  - GoogLeNet: `googlenet`
  - Inception_v3: `inception_v3`
- `--alpha`(default: `14`): The precision parameter $\alpha$ for approximation. Integers from 4 to 14 can be used.
- `--B_relu`(default: `100.0`): The value of $B$, which is the bound of approximation range $[-B,B]$ for the approximate ReLU function. 
- `--B_max`(default: `10.0`): The value of $B$, which is the bound of approximation range $[-B,B]$ for the approximate max-pooling function.
- `--B_search`(default: `10.0`): The step size for searching $B$ such that all input values fall within the approximation range.
- `--batch_inf`(default: `16`): Batch size for inference.
- `--dataset_path`: The path of the directory which contains the ImageNet. The default directory is `../dataset/imagenet`.

**Note.** Especially for inference ImageNet, the batch size should be determined more carefully due to the computation for the proposed polynomials. Refer to the batch size we used in our GPU environment, NVIDIA GeForce RTX 3090.

|   Backbone   | Batch size |
|:------------:|:----------:|
|  ResNet-152  |      32     |
|    VGG-19    |      8     |
|   GoogLeNet  |      32     |
| Inception-v3 | 16 |
