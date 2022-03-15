import argparse
import os
import torch
from torch.autograd import Variable
from torchvision import models
import sys
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from prune_cifar import *

import logging

class GroupResNet(torch.nn.Module):
    def __init__(self, model, dataset = 'cifar10'):
        super(GroupResNet, self).__init__()
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.avgpool = model.avgpool
        self.linear = model.linear

        self.n_clusters = model.n_clusters

        self.pruning_rate = model.pruning_rate
        self.kernel_gcd = model.kernel_gcd
        # self.pruning_rate = 0.4375
        # self.kernel_gcd = 16
        print(f'GroupResNet pruning rate: {self.pruning_rate}; kernel_gcd: {self.kernel_gcd}')
        imagenet_target_layers = [4, 5, 6, 7]
        # tiny_imagenet_target_layers = [2, 3, 4, 5]
        tiny_imagenet_target_layers = [2, 3, 4]
        cifar10_target_layers = [2, 3, 4]

        if dataset == 'imagenet':
            target_layers = imagenet_target_layers
        elif dataset == 'tiny_imagenet':
            target_layers = tiny_imagenet_target_layers
        elif dataset == 'cifar10':
            target_layers = cifar10_target_layers
        else:
            print(f'Invalid dataset input {dataset}.')
            sys.exit(0)

        global modules
        for layer, (name, modules) in enumerate(model._modules.items()):
            #self.modules = modules
            if layer in target_layers:
                for sublayer, (name,submodule) in enumerate(modules._modules.items()):
                    self.out_list = submodule.out_list
                    self.layer_list = []
                    self.in_list = []
                    self.in_planes = submodule.conv1.in_channels
                    self.planes = submodule.conv1.out_channels

                    for subsublayer, (name, module) in enumerate(submodule._modules.items()):
                        if isinstance(module, torch.nn.modules.conv.Conv2d):
                            new_conv, number_of_unpruned_kernels = make_new_conv(module, group_info = (self.n_clusters, self.pruning_rate, self.kernel_gcd))

                            old_weights = module.weight.cuda()
                            old_out_channels, old_in_channels, old_kernel_size, old_kernel_size = old_weights.data.size()
                            original_old_weights_shape = old_weights.data.size()
                            old_weights = old_weights.data.cpu().numpy()
                            new_weights = torch.zeros(module.out_channels, module.in_channels * number_of_unpruned_kernels // self.kernel_gcd, module.kernel_size[0], module.kernel_size[1])
                            new_weights = new_weights.data.cpu().numpy()

                            d_out = old_out_channels // self.n_clusters

                            self.j_list = []
                            if subsublayer == 0:
                                conv_num = 0
                            elif subsublayer == 2:
                                conv_num = 1
                            for i in range(self.n_clusters):
                                wi = old_weights[i * d_out:(i + 1) * d_out, :, :, :]
                                m = 0
                                # print('i is', i)
                                for j in range(old_in_channels):
                                    # if not np.all(wi[:, j, :, :] == 0):
                                    if j in submodule.preserved_kernel_index[conv_num][i]:
                                        # print(f'{layer, sublayer, subsublayer}; i: {i}; j: {j}; m: {m}; {original_old_weights_shape} --> {new_weights.shape}; module.in_channels {module.in_channels}')
                                        # first = new_weights[i * d_out:(i + 1) * d_out, m, :, :]
                                        # second = wi[:, j, :, :]
                                        new_weights[i * d_out:(i + 1) * d_out, m, :, :] = wi[:, j, :, :]
                                        self.j_list.append(j)
                                        m = m + 1
                                # print(f'i: {i}; m: {m}')

                            self.j_list = np.array(self.j_list)
                            self.j_list = torch.from_numpy(self.j_list)
                            self.in_list.append(self.j_list.cuda())
                            # print(f'j_list: {self.j_list} ({len(self.j_list)})')

                            new_weights = Variable(torch.from_numpy(new_weights))
                            self.new_weight = torch.nn.Parameter(new_weights)
                            self.new_weight.data = self.new_weight.type(torch.FloatTensor)
                            self.new_weight.data = self.new_weight.data.cuda()
                            new_conv.weight = self.new_weight
                            # print(new_conv.weight.size())
                            self.layer_list.append(new_conv)
                    # print(self.layer_list[0])
                    # print('--------------', self.in_list[0].size())
                    # print(f'{layer}-{sublayer} in_list: {self.in_list}')
                    modules[sublayer] = GroupBasicblock(submodule, self.in_list, self.out_list, self.layer_list, self.in_planes, self.planes)
                    # modules[sublayer] = GroupBasicblock(submodule, submodule.preserved_kernel_index, self.out_list, self.layer_list, self.in_planes, self.planes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

class GroupBasicblock(torch.nn.Module):
    expansion = 4

    def __init__(self, module, in_list, out_list, layer_list, in_planes, planes, stride=1, shortcut=None):
        super(GroupBasicblock, self).__init__()
        self.in_list = in_list
        self.out_list = out_list
        self.conv1 = layer_list[0]
        self.bn1 = module.bn1
        self.conv2 = layer_list[1]
        self.bn2 = module.bn2
        self.shortcut = shortcut
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = module.shortcut

    def forward(self, x):
        residual = x
        # print('x', x)
        # print('x.shape', x.shape)
        # print('self.in_list[0]', self.in_list[0])
        # print('self.in_list[0].shape', self.in_list[0].shape)
        out = torch.index_select(x, 1, self.in_list[0])
        # print('out.size()', out.size())
        out = self.conv1(out)
        out = torch.index_select(out, 1, self.out_list[0])
        out = self.bn1(out)
        out = F.relu(out)
        out = torch.index_select(out, 1, self.in_list[1])
        out = self.conv2(out)
        out = torch.index_select(out, 1, self.out_list[1])
        out = self.bn2(out)
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out += residual
        out = F.relu(out)
        return out

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='group.py')
    parser.add_argument('--model', type=str,
                        help='path of pruned model')
    parser.add_argument('--output', default='', type=str,
                        help='path of grouped model')
    args = parser.parse_args()

    model = torch.load(args.model).cuda()
    newmodel = GroupResNet(model).cuda()

    # print(newmodel)
    torch.save(newmodel, args.output)
