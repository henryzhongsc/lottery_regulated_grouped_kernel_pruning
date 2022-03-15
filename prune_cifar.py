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
    def __init__(self, model, setting, original_model, dataset = 'cifar10', pruning_rate = 0.4375, pruned_flag = False):
        super(ModifiedResNet, self).__init__()
        if not pruned_flag:
            self.conv1 = model.conv_1_3x3
            self.bn1 = model.bn_1
            self.layer1 = model.stage_1
            self.layer2 = model.stage_2
            self.layer3 = model.stage_3
            self.avgpool = model.avgpool
            self.linear = model.classifier

        else:
            self.conv1 = model.conv1
            self.bn1 = model.bn1
            self.layer1 = model.layer1
            self.layer2 = model.layer2
            self.layer3 = model.layer3
            self.avgpool = model.avgpool
            self.linear = model.linear

        self.original_model = original_model
        self.kernel_gcd = float('inf')

        # self.model = model
        self.dataset = dataset
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
        self.pruning_strategy = setting['prune_params']['pruning_strategy']

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

        if self.pruning_strategy == 'assign' and setting['prune_params']['kernel_pruning_assignment'] is not None:
            self.kernel_pruning_assignment_flag = True
            with open(setting['prune_params']['kernel_pruning_assignment']) as kernel_pruning_index_list_f:
                self.kernel_pruning_assignment_info_list = json.load(kernel_pruning_index_list_f)

        else:
            self.kernel_pruning_assignment_flag = False

        logger.info(f'pruning_strategy: {self.pruning_strategy} ({self.criterion}); n_clusters: {self.n_clusters}; eval_pruned_kernel_relationship: {self.eval_pruned_kernel_relationship}; cost_balancer: {self.cost_balancer}; eval_kept_kernel_ratio: {self.eval_kept_kernel_ratio}')

        for current_block, layer, sublayer, modules, submodule in self.gen_unpruned_block(model):
            modules[sublayer] = self.prune_block(current_block, submodule, (layer, sublayer), pruned_flag = pruned_flag)


    def gen_unpruned_block(self, model):

        imagenet_target_layers = [4, 5, 6, 7]
        tiny_imagenet_target_layers = [2, 3, 4]
        cifar10_target_layers = [2, 3, 4]

        if self.dataset == 'imagenet':
            target_layers = imagenet_target_layers
        elif self.dataset == 'tiny_imagenet':
            target_layers = tiny_imagenet_target_layers
        elif self.dataset == 'cifar10':
            target_layers = cifar10_target_layers
        else:
            logger.error(f'Invalid dataset input {dataset}.')
            sys.exit(0)


        for layer, (name, modules) in enumerate(model._modules.items()):
            if layer in target_layers:
                for sublayer, (name, submodule) in enumerate(modules._modules.items()):
                    current_block = modules[sublayer]
                    yield current_block, layer, sublayer, modules, submodule

    def prune_block(self, current_block, submodule, block_info, pruned_flag):
        layer, sublayer = block_info
        block_out_list = []
        block_layer_list = []
        if pruned_flag:
            block_in_planes = None
            block_planes = None
        else:
            block_in_planes = submodule.conv_a.in_channels
            block_planes = submodule.conv_a.out_channels

        block_prune_masks = []
        block_candidate_methods_list = []
        block_preserved_kernel_index = []
        block_layer_info = []

        for subsublayer, (name, module) in enumerate(submodule._modules.items()):
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                new_conv = make_new_conv(module)


                old_weights = module.weight.cuda()
                old_out_channels, old_in_channels, old_kernel_size, old_kernel_size = old_weights.data.size()
                old_weights = old_weights.data.cpu().numpy()
                original_shape = old_weights.shape

                self.kernel_gcd = min(self.kernel_gcd, original_shape[0])
                old_weights_float = torch.from_numpy(old_weights).float()

                layer_info = (layer, sublayer, subsublayer)

                if not pruned_flag:
                    block_layer_info.append(layer_info)

                    old_weights = old_weights.reshape(old_out_channels, old_in_channels*old_kernel_size*old_kernel_size)

                    if self.clustering_assignment_flag:
                        if self.clustering_assignment_setting == 'method':
                            conv_assigned_clustering_method = self.clustering_assignment_info_list.pop(0)

                            preferred_permutation_matrix, preferred_clustering_method, clustering_methods_info, candidate_methods_list = get_preferred_permutation_matrix(None, old_weights, old_out_channels, criterion = self.criterion, n_clusters = self.n_clusters, clustering_method=conv_assigned_clustering_method)

                        elif self.clustering_assignment_setting == 'permutation matrix':
                            preferred_permutation_matrix, preferred_clustering_method =  self.clustering_assignment_info_list.pop(0)
                            preferred_permutation_matrix = np.array(preferred_permutation_matrix)

                            if self.clustering_method == 'tmi shuffled':
                                preferred_permutation_matrix = reassign_permutation_matrix(preferred_permutation_matrix, int(old_out_channels / self.n_clusters), force_shuffle = True)

                            if self.clustering_method == 'random shuffled':
                                preferred_permutation_matrix = reassign_permutation_matrix(preferred_permutation_matrix, int(old_out_channels / self.n_clusters), force_shuffle = False)

                            clustering_methods_info = 'Assigned'
                            candidate_methods_list = None

                    else:
                        criterion_result = get_filters_LTH_metrics(old_weights_float, self.original_model, layer_info, ticket_start_epoch = self.ticket_start_epoch, ticket_end_epoch = self.ticket_end_epoch, ticket_step = self.ticket_step, snapshot_path = self.snapshot_path, criterion = self.criterion)

                        preferred_permutation_matrix, preferred_clustering_method, clustering_methods_info, candidate_methods_list = get_preferred_permutation_matrix(criterion_result, old_weights, old_out_channels, criterion = self.criterion, n_clusters = self.n_clusters, clustering_method=self.clustering_method)

                    clustering_info = (preferred_permutation_matrix, preferred_clustering_method, candidate_methods_list)

                    block_candidate_methods_list.append(clustering_info)


                    logger.info(f'Layer {layer}-{sublayer}-{subsublayer}; Shape {original_shape} -> {old_weights.shape}; Method: {preferred_clustering_method}; {clustering_methods_info}')


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


                if subsublayer == 0:
                    conv_num = 0
                elif subsublayer == 2:
                    conv_num = 1
                else:
                    logger.error(f'subsublayer: {subsublayer} is neither 0 or 2.')
                    sys.exit(0)

                if self.kernel_pruning_assignment_flag:
                    conv_di = self.kernel_pruning_assignment_info_list.pop(0)
                    new_conv, new_conv_prune_mask, new_conv_preserved_kernel_index = prune_kernels(current_block, conv_num, new_conv, new_weights, old_out_channels, pruning_rate = self.pruning_rate,
                                        eval_pruned_kernel_relationship = self.eval_pruned_kernel_relationship,
                                        eval_kept_kernel_ratio = self.eval_kept_kernel_ratio,
                                        cost_balancer = self.cost_balancer,
                                        n_clusters = self.n_clusters,
                                        metric = self.metric,
                                        pruned_flag = pruned_flag,
                                        pruning_strategy = self.pruning_strategy,
                                        assignable_di = conv_di)
                else:
                    new_conv, new_conv_prune_mask, new_conv_preserved_kernel_index = prune_kernels(current_block, conv_num, new_conv, new_weights, old_out_channels, pruning_rate = self.pruning_rate,
                                        eval_pruned_kernel_relationship = self.eval_pruned_kernel_relationship,
                                        eval_kept_kernel_ratio = self.eval_kept_kernel_ratio,
                                        cost_balancer = self.cost_balancer,
                                        n_clusters = self.n_clusters,
                                        metric = self.metric,
                                        pruned_flag = pruned_flag,
                                        pruning_strategy = self.pruning_strategy)

                block_prune_masks.append(new_conv_prune_mask)
                block_layer_list.append(new_conv)
                block_preserved_kernel_index.append(new_conv_preserved_kernel_index)


        if pruned_flag:
            return NewBasicblock(submodule, submodule.bn1, submodule.bn2, current_block.out_list, block_layer_list, block_prune_masks, block_candidate_methods_list, block_preserved_kernel_index, shortcut = current_block.shortcut, pruned_flag = pruned_flag)
        else:
            return NewBasicblock(submodule, submodule.bn_a, submodule.bn_b, block_out_list, block_layer_list, block_prune_masks, block_candidate_methods_list, block_preserved_kernel_index, in_planes = block_in_planes, planes = block_planes, pruned_flag = pruned_flag)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        #x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x




class NewBasicblock(torch.nn.Module):
    expansion = 1
    # def __init__(self, module, out_list, layer_list, in_planes, planes, prune_mask, candidate_methods_list, preserved_kernel_index, stride=1, shortcut=None, layer_info = None, bn1=None, bn2=None):
    def __init__(self, module, bn1, bn2, out_list, layer_list, prune_mask, candidate_methods_list, preserved_kernel_index, in_planes = None, planes = None, stride=1, shortcut=None, pruned_flag = False):
        super(NewBasicblock, self).__init__()

        # self.layer_info = layer_info

        self.out_list = out_list
        self.conv1 = layer_list[0]
        self.bn1 = bn1
        self.conv2 = layer_list[1]
        self.bn2 = bn2
        self.shortcut = shortcut
        self.prune_mask = prune_mask
        self.candidate_methods_list = candidate_methods_list
        self.preserved_kernel_index = preserved_kernel_index

        if pruned_flag is False:
            if stride != 1 or in_planes != self.expansion * planes:
                self.shortcut = module.downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = torch.index_select(out, 1, self.out_list[0])
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = torch.index_select(out, 1, self.out_list[1])
        out = self.bn2(out)
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out += residual
        out = F.relu(out)
        return out


def zeroize_pruned_kernels(model):

    for layer, (name, modules) in enumerate(model._modules.items()):
        if layer == 2 or layer == 3 or layer == 4:
            for sublayer, (name, submodule) in enumerate(modules._modules.items()):
                old_block = modules[sublayer]
                if isinstance(old_block, NewBasicblock):
                    conv1_prune_mask, conv2_prune_mask = old_block.prune_mask

                    conv1_old_weights = old_block.conv1.weight.cuda()
                    conv2_old_weights = old_block.conv2.weight.cuda()
                    conv1_old_weights = conv1_old_weights.data.cpu().numpy()
                    conv2_old_weights = conv2_old_weights.data.cpu().numpy()


                    # conv1_old_weights = torch.from_numpy(conv1_old_weights).float()
                    # conv2_old_weights = torch.from_numpy(conv2_old_weights).float()
                    # print('conv1_old_weights', type(conv1_old_weights))
                    conv1_old_weights = Variable(torch.from_numpy(conv1_old_weights)).cuda()
                    conv2_old_weights = Variable(torch.from_numpy(conv2_old_weights)).cuda()
                    conv1_new_weights = torch.mul(conv1_old_weights.double(), conv1_prune_mask, out=None)
                    conv2_new_weights = torch.mul(conv2_old_weights.double(), conv2_prune_mask, out=None)

                    old_block.conv1.weight = torch.nn.Parameter(conv1_new_weights)
                    old_block.conv2.weight = torch.nn.Parameter(conv2_new_weights)

                    old_block.conv1.weight.data = old_block.conv1.weight.type(torch.FloatTensor)
                    old_block.conv1.weight.data = old_block.conv1.weight.data.cuda()
                    old_block.conv2.weight.data = old_block.conv2.weight.type(torch.FloatTensor)
                    old_block.conv2.weight.data = old_block.conv2.weight.data.cuda()