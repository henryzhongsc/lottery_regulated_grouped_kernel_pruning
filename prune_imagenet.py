import torch
from torch.autograd import Variable
from torchvision import models
#import cv2
import argparse
import sys
import time
import math
import copy
import logging
import json

import numpy as np
import torchvision
import torch.nn.functional as F
import torch.nn as nn

from sklearn.metrics.pairwise import rbf_kernel
from scipy.spatial.distance import pdist, squareform
import scipy.spatial as sp

from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from Spectral_Clustering.spectral_clustering import Spectral_Clustering
from same_size_dbscan import Same_Size_DBSCAN

from kernel_group_pruning import select_kernel_from_group
from utils import *

logger = logging.getLogger()

class ModifiedResNet(torch.nn.Module):
    # def __init__(self, model, snapshot_path, ticket_start_epoch = 35, ticket_end_epoch = 70, n_clusters = 8, pruning_rate = 0.4375, clustering_method = 'ALL', criterion = 'tickets magnitute increase'):
    def __init__(self, model, setting, original_model, dataset = 'imagenet', pruning_rate = 0.4375, pruned_flag = False):
        super(ModifiedResNet, self).__init__()


        self.conv1 = model.conv1
        self.bn1 = model.bn1

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        try:
            self.relu = model.relu
            self.maxpool = model.maxpool
            self.avgpool = model.avgpool
        except AttributeError:
            pass

        try:
            self.fc = model.fc
        except AttributeError:
            self.fc = model.linear


        self.dataset = dataset
        self.original_model = original_model
        self.kernel_gcd = float('inf')

        # self.model = model
        self.pruning_rate = pruning_rate
        self.n_clusters = setting['prune_params']['n_clusters']
        self.snapshot_path = setting['management']['snapshot_folder_dir']
        self.ticket_start_epoch = setting['prune_params']['ticket_start_epoch']
        self.ticket_end_epoch = setting['prune_params']['ticket_end_epoch']
        self.ticket_step = setting['prune_params']['ticket_step']
        self.clustering_method = setting['prune_params']['clustering_method']
        self.criterion = setting['prune_params']['criterion']

        self.eval_pruned_kernel_relationship = setting['prune_params']['eval_pruned_kernel_relationship']
        self.eval_kept_kernel_ratio = setting['prune_params']['eval_kept_kernel_ratio']
        self.cost_balancer = setting['prune_params']['cost_balancer']
        self.metric = setting['prune_params']['metric']

        if setting['prune_params']['clustering_assignment'] is not None:
            self.clustering_assignment_flag = True
            self.clustering_assignment_setting = setting['prune_params']['clustering_assignment']
            if setting['prune_params']['clustering_assignment'] == 'method':
                self.clustering_assignment_info_list = setting['cluster_info']['method_list']
            elif setting['prune_params']['clustering_assignment'] == 'permutation matrix':
                with open(setting['cluster_info']['permutation_matrix_list']) as permutation_matrix_list_f:
                    self.clustering_assignment_info_list = json.load(permutation_matrix_list_f)
            else:
                logger.error(f"No such setting['prune_params']['clustering_assignment']: {setting['prune_params']['clustering_assignment']} supported")
                sys.exit(0)

        else:
            self.clustering_assignment_flag = False

        for current_block, layer, sublayer, modules, submodule in self.gen_unpruned_block(model):
            modules[sublayer] = self.prune_block(current_block, submodule, (layer, sublayer), pruned_flag = pruned_flag)

        # print('$'*50)
        # print(f'ModifiedResNet.kernel_gcd after pruned: {self.kernel_gcd}')
        # print('$'*50)

    def gen_unpruned_block(self, model):

        imagenet_target_layers = [4, 5, 6, 7]
        tiny_imagenet_target_layers = [2, 3, 4, 5]
        cifar10_target_layers = [2, 3, 4]

        if self.dataset == 'imagenet':
            target_layers = imagenet_target_layers
        elif self.dataset == 'tiny_imagenet':
            target_layers = tiny_imagenet_target_layers
        elif self.dataset == 'cifar10':
            target_layers = cifar10_target_layers
        else:
            print(f'Invalid dataset input {dataset}.')
            sys.exit(0)


        for layer, (name, modules) in enumerate(model._modules.items()):
            if layer in target_layers:
                for sublayer, (name, submodule) in enumerate(modules._modules.items()):
                    current_block = modules[sublayer]
                    # if not isinstance(current_block, NewBasicblock):
                    yield current_block, layer, sublayer, modules, submodule

    def prune_block(self, current_block, submodule, block_info, pruned_flag):
        layer, sublayer = block_info
        block_out_list = []
        block_layer_list = []
        if pruned_flag:
            block_in_planes = None
            block_planes = None
        else:
            block_in_planes = submodule.conv1.in_channels
            block_planes = submodule.conv1.in_channels


        block_prune_masks = []
        block_candidate_methods_list = []
        block_preserved_kernel_index = []
        block_layer_info = []

        for subsublayer, (name, module) in enumerate(submodule._modules.items()):
            # second_conv_flag = False
            # if isinstance(module, torch.nn.modules.conv.Conv2d):
            if subsublayer == 2:
                new_conv = make_new_conv(module)


                old_weights = module.weight.cuda()
                old_out_channels, old_in_channels, old_kernel_size, old_kernel_size = old_weights.data.size()
                old_weights = old_weights.data.cpu().numpy()
                original_shape = old_weights.shape
                # print(f'Update kernel_gcd as min of {self.kernel_gcd}, {original_shape[0]}')
                self.kernel_gcd = min(self.kernel_gcd, original_shape[0])


                old_weights_float = torch.from_numpy(old_weights).float()


                layer_info = (layer, sublayer, subsublayer)
                # print(f'layer info: {layer_info}; ({current_block.layer_info[0]}, {current_block.layer_info[1]})')
                if not pruned_flag:
                    block_layer_info.append(layer_info)
                    # criterion_result = get_filters_LTH_metrics(old_weights_float, self.original_model, layer_info, ticket_start_epoch = self.ticket_start_epoch, ticket_end_epoch = self.ticket_end_epoch, snapshot_path = self.snapshot_path, criterion = self.criterion)

                    old_weights = old_weights.reshape(old_out_channels, old_in_channels*old_kernel_size*old_kernel_size)

                    if self.clustering_assignment_flag:
                        if self.clustering_assignment_setting == 'method':
                            conv_assigned_clustering_method = self.clustering_assignment_info_list.pop(0)

                            # preferred_permutation_matrix, preferred_clustering_method, clustering_methods_info, candidate_methods_list = get_preferred_permutation_matrix(criterion_result, old_weights, old_out_channels, criterion = self.criterion, n_clusters = self.n_clusters, clustering_method=conv_assigned_clustering_method)

                            preferred_permutation_matrix, preferred_clustering_method =  get_cluster_permutation_matrix(None, old_weights, old_out_channels, n_clusters = self.n_clusters, clustering_method=conv_assigned_clustering_method)

                            clustering_methods_info = 'Assigned'
                            candidate_methods_list = None

                        elif self.clustering_assignment_setting == 'permutation matrix':
                            preferred_permutation_matrix, preferred_clustering_method =  self.clustering_assignment_info_list.pop(0)
                            preferred_permutation_matrix = np.array(preferred_permutation_matrix)

                            clustering_methods_info = 'Assigned'
                            candidate_methods_list = None

                    else:
                        criterion_result = get_filters_LTH_metrics(old_weights_float, self.original_model, layer_info, ticket_start_epoch = self.ticket_start_epoch, ticket_end_epoch = self.ticket_end_epoch, snapshot_path = self.snapshot_path, criterion = self.criterion)

                        preferred_permutation_matrix, preferred_clustering_method, clustering_methods_info, candidate_methods_list = get_preferred_permutation_matrix(criterion_result, old_weights, old_out_channels, criterion = self.criterion, n_clusters = self.n_clusters, clustering_method=self.clustering_method)

                    clustering_info = (preferred_permutation_matrix, preferred_clustering_method, candidate_methods_list)

                    block_candidate_methods_list.append(clustering_info)


                    logger.info(f'Layer {layer}-{sublayer}-{subsublayer}; Shape {original_shape} -> {old_weights.shape}; Method: {preferred_clustering_method}; {clustering_methods_info}')



                    # preferred_permutation_matrix_transposed = preferred_permutation_matrix.transpose(1,0)
                    # block_out_index = get_out_index(preferred_permutation_matrix).cuda()
                    block_out_index = get_out_index(preferred_permutation_matrix.transpose(1,0)).cuda()
                    block_out_index = Variable(block_out_index)
                    block_out_list.append(block_out_index)
                    new_weights = np.dot(preferred_permutation_matrix, old_weights)
                    new_weights = new_weights.reshape(old_out_channels, old_in_channels, old_kernel_size, old_kernel_size)
                    new_weights = Variable(torch.from_numpy(new_weights)).cuda()

                else:
                    old_weights = old_weights.reshape(old_out_channels, old_in_channels, old_kernel_size, old_kernel_size)
                    new_weights = Variable(torch.from_numpy(old_weights)).cuda()

                    logger.info(f'Layer {layer}-{sublayer}-{subsublayer}; Shape {original_shape} -> {old_weights.shape} being pruned again.')


                conv_num = 0

                new_conv, new_conv_prune_mask, new_conv_preserved_kernel_index = prune_kernels(current_block, conv_num, new_conv, new_weights, old_out_channels, pruning_rate = self.pruning_rate,
                                    eval_pruned_kernel_relationship = self.eval_pruned_kernel_relationship,
                                    eval_kept_kernel_ratio = self.eval_kept_kernel_ratio,
                                    cost_balancer = self.cost_balancer,
                                    n_clusters = self.n_clusters,
                                    metric = self.metric,
                                    pruned_flag = pruned_flag)

                block_prune_masks.append(new_conv_prune_mask)
                block_layer_list.append(new_conv)
                block_preserved_kernel_index.append(new_conv_preserved_kernel_index)





        if pruned_flag:
            return NewBasicblock(submodule, current_block.out_list, block_layer_list, block_prune_masks, block_candidate_methods_list, block_preserved_kernel_index, downsample = current_block.downsample, pruned_flag = pruned_flag)
        else:

            return NewBasicblock(submodule, block_out_list, block_layer_list, block_prune_masks, block_candidate_methods_list, block_preserved_kernel_index, in_planes = block_in_planes, planes = block_planes, pruned_flag = pruned_flag)

    def forward(self, x):
        try:

            x = self.relu(self.bn1(self.conv1(x)))
        except AttributeError:

            x = F.relu(self.bn1(self.conv1(x)))
        try:
            x = self.maxpool(x)
        except AttributeError:
            pass
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        try:
            x = self.avgpool(x)
            x = self.avgpool(x)
        except AttributeError:
            x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x




class NewBasicblock(torch.nn.Module):
    expansion = 4
    def __init__(self, module, out_list, layer_list, prune_mask, candidate_methods_list, preserved_kernel_index, in_planes = None, planes = None, stride=1, downsample=None, pruned_flag = False):
        super(NewBasicblock, self).__init__()

        # self.layer_info = layer_info

        self.out_list = out_list
        self.conv1 = module.conv1
        self.bn1 = module.bn1
        self.conv2 = layer_list[0]
        self.bn2 = module.bn2
        self.conv3 = module.conv3
        self.bn3 = module.bn3
        try:
            self.relu = module.relu
        except AttributeError:
            pass
        self.downsample = downsample

        self.prune_mask = prune_mask
        self.candidate_methods_list = candidate_methods_list
        self.preserved_kernel_index = preserved_kernel_index

        if pruned_flag is False:
            if stride != 1 or in_planes != self.expansion * planes:
                try:
                    self.downsample = module.downsample
                except AttributeError:
                    self.downsample = module.shortcut

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        # out = torch.index_select(out, 1, self.out_list[0])
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = torch.index_select(out, 1, self.out_list[0])
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv3(out)
        # out = torch.index_select(out, 1, self.out_list[2])
        out = self.bn3(out)
        #x = self.relu(x)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        try:
            out = self.relu(out)
        except AttributeError:
            out = F.relu(out)
        return out


def zeroize_pruned_kernels(model):

    for layer, (name, modules) in enumerate(model._modules.items()):
        if layer == 4 or layer == 5 or layer == 6 or layer == 7:
            for sublayer, (name, submodule) in enumerate(modules._modules.items()):
                old_block = modules[sublayer]
                if isinstance(old_block, NewBasicblock):
                    conv2_prune_mask = old_block.prune_mask[0]

                    conv2_old_weights = old_block.conv2.weight.cuda()
                    conv2_old_weights = conv2_old_weights.data.cpu().numpy()

                    conv2_old_weights = Variable(torch.from_numpy(conv2_old_weights)).cuda()
                    conv2_new_weights = torch.mul(conv2_old_weights.double(), conv2_prune_mask, out=None)

                    old_block.conv2.weight = torch.nn.Parameter(conv2_new_weights)

                    old_block.conv2.weight.data = old_block.conv2.weight.type(torch.FloatTensor)
                    old_block.conv2.weight.data = old_block.conv2.weight.data.cuda()


