import copy
import numpy as np
import torch
import torchvision
from torchvision import models
from torch.autograd import Variable
import torch.nn.functional as F

import scipy.stats as ss
from sklearn.preprocessing import StandardScaler


from sklearn.metrics.pairwise import rbf_kernel
from scipy.spatial.distance import pdist, squareform
import scipy.spatial as sp

from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from Spectral_Clustering.spectral_clustering import Spectral_Clustering
from same_size_dbscan import Same_Size_DBSCAN
from EqualGroupKMeans.clustering.equal_groups import EqualGroupsKMeans
from kernel_group_pruning import select_kernel_from_group
from kernel_group_pruning import find_optimal

import os
import sys
import copy
import random

import collections
import logging
logger = logging.getLogger()

def reassign_permutation_matrix(m, group_size, force_shuffle = True):

    same_label_occur_count = 0
    label = 0
    r = []
    for row_index, row in enumerate(m):
        if same_label_occur_count == group_size:
            same_label_occur_count = 0
            label += 1
        for col_index, col in enumerate(row):
            if col == 1:
                r.append((col_index, label))
        same_label_occur_count += 1

    new_r = []
    n_clusters = len(set([i[1] for i in r]))

    if force_shuffle:
        current_original_label = -1
        current_new_label = -1
        for order, (index, label) in enumerate(r):
            if label == current_original_label:
                new_r.append((index, current_new_label % n_clusters))
                current_new_label += 1
            elif label != current_original_label:
                current_original_label = label
                current_new_label = label
                new_r.append((index, current_new_label))
                current_new_label += 1




        X_labels = copy.deepcopy(new_r)
        X_labels.sort(key=lambda t: t[0])
        X_labels = [i[1] for i in X_labels]

    else:

        r_sorted_by_index = sorted(r, key = lambda t: t[0])
        X_labels = [i[1] for i in r]
        random.shuffle(X_labels)


    index_label_LUT = [[i, v] for i, v in enumerate(X_labels)]
    index_label_LUT.sort(key=lambda t: (t[1], t[0]))
    permutation_matrix = np.zeros((m.shape[0], m.shape[0]))
    for order, (original_index, _) in enumerate(index_label_LUT):
        permutation_matrix[order, original_index] = 1

    return permutation_matrix


def get_permutation_matrix(X, X_labels):
    n_clusters = len(set(X_labels))
    index_label_LUT = [[i, v] for i, v in enumerate(X_labels)]
    index_label_LUT.sort(key=lambda t: (t[1], t[0]))

    permutation_matrix = np.zeros((X.shape[0], X.shape[0]))
    for order, (original_index, _) in enumerate(index_label_LUT):
        permutation_matrix[order, original_index] = 1

    return permutation_matrix

def get_equal_k_means_permutation_matrix(X, n_clusters):
    clf = EqualGroupsKMeans(n_clusters=n_clusters, random_state=0)

    clf.fit(X)
    X_labels = clf.labels_

    permutation_matrix = get_permutation_matrix(X, X_labels)

    return permutation_matrix, X_labels


def prune_kernels(current_block, conv_num, new_conv, new_weights, old_out_channels, pruning_rate = 0.4375, eval_pruned_kernel_relationship = True, eval_kept_kernel_ratio = 0.2, cost_balancer = 8, n_clusters = 8, metric = 'euclidean', pruned_flag = False, pruning_strategy = 'greedy', assignable_di = []):

    weights = new_weights.cuda()
    conv_prune_mask = torch.zeros(weights.data.size()).cuda()
    d_out = old_out_channels // n_clusters


    conv_preserved_kernel_index = []
    for i in range(n_clusters):
        wi = weights[i*d_out:(i+1)*d_out, :, :, :]
        wi_copy = copy.deepcopy(wi)
        _, wi_copy_in_channels, _, _ = wi_copy.data.size()
        wi_copy = wi_copy.data.cpu().numpy()
        wi = wi.transpose(1,0).contiguous()
        in_channels, out_channels, kernel_size, kernel_size = wi.data.size()
        wi = wi.view(in_channels, out_channels*kernel_size*kernel_size)
        wi = wi.data.cpu().numpy()


        # Greedy Pruning
        if pruning_strategy == 'greedy':
            sim_matrix = 1 - sp.distance.cdist(wi, wi, metric = metric)
            di = select_kernel_from_group(sim_matrix, pruning_rate = pruning_rate, eval_pruned_kernel_relationship = eval_pruned_kernel_relationship, eval_kept_kernel_ratio = eval_kept_kernel_ratio, cost_balancer = cost_balancer) # kept indices

        elif pruning_strategy == 'greedy_false_reverse':
            # False reverse graddy pruning:
            sim_matrix = 1 - sp.distance.cdist(wi, wi, metric = metric)
            di = select_kernel_from_group(sim_matrix, pruning_rate = pruning_rate, eval_pruned_kernel_relationship = eval_pruned_kernel_relationship, eval_kept_kernel_ratio = eval_kept_kernel_ratio, cost_balancer = cost_balancer, ablation_reverse = True) # kept indices

        elif pruning_strategy == 'greedy_complement':
            # Reverse Greedy Pruning:
            sim_matrix = 1 - sp.distance.cdist(wi, wi, metric = metric)
            remained_kernel_capacity = int((1 - pruning_rate) * (sim_matrix.shape[0] + 1))
            di_all = select_kernel_from_group(sim_matrix, pruning_rate = 0, eval_pruned_kernel_relationship = eval_pruned_kernel_relationship, eval_kept_kernel_ratio = eval_kept_kernel_ratio, cost_balancer = cost_balancer) # kept all indices

            di = di_all[::-1][:remained_kernel_capacity]

        elif pruning_strategy == 'L2':
            wi_tensor = torch.from_numpy(wi)
            norm_list = torch.norm(wi_tensor, p=2, dim=1)
            norm_list = [(i, v) for i, v in enumerate(norm_list)]
            norm_list.sort(reverse = True, key = lambda t: t[1])
            number_of_grouped_kernel_to_keep = int((1 - pruning_rate) * len(norm_list))

            di = [i for i, v in norm_list[:number_of_grouped_kernel_to_keep]]

        elif pruning_strategy == 'assign':
            di = assignable_di[i]


        else:
            logger.error(f"Invalid input on pruning_strategy: {pruning_strategy}")
            sys.exit()

        if pruned_flag:
            max_di_len = len(di)
            previous_di = current_block.preserved_kernel_index[conv_num][i]
            also_in_new_di = [k_i for k_i in di if k_i in previous_di]
            exclusive_to_previous_di = [k_i for k_i in previous_di if k_i not in also_in_new_di]

            di = also_in_new_di + exclusive_to_previous_di
            di = di[:max_di_len]


        conv_preserved_kernel_index.append(di)
        for d in di:
            conv_prune_mask[i*d_out:(i+1)*d_out, d, :, :].fill_(1)

    conv_prune_mask = conv_prune_mask.double()
    new_weights = torch.mul(new_weights.double(), conv_prune_mask, out=None)
    #print(new_weights)
    new_conv.weight = torch.nn.Parameter(new_weights)
    new_conv.weight.data = new_conv.weight.type(torch.FloatTensor)
    new_conv.weight.data = new_conv.weight.data.cuda()


    return new_conv, conv_prune_mask, conv_preserved_kernel_index

def same_index_checker(conv_preserved_kernel_index, conv_prune_mask):
    conv_prune_mask_perserved_index = []
    for i_prune_mask in conv_prune_mask:
        conv_prune_mask_perserved_index.append(get_prune_mask_preserved_index(i_prune_mask))


    dup_removed_conv_prune_mask_perserved_index = [conv_prune_mask_perserved_index[0]]


    for i in range(1, len(conv_prune_mask_perserved_index)):
        if conv_prune_mask_perserved_index[i] == conv_prune_mask_perserved_index[i-1]:
            continue
        else:
            dup_removed_conv_prune_mask_perserved_index.append(conv_prune_mask_perserved_index[i])


    sorted_conv_preserved_kernel_index = [sorted(i) for i in conv_preserved_kernel_index]
    logger.info(f'conv_prune_mask_perserved_index: {dup_removed_conv_prune_mask_perserved_index}')
    logger.info(f'conv_preserved_kernel_index:     {sorted_conv_preserved_kernel_index}')
    logger.info(f'Identical? {dup_removed_conv_prune_mask_perserved_index == sorted_conv_preserved_kernel_index}')


def get_epoch_snapshot_weights(model, layer_info, epoch_num, path, arch = 'resnet'):
    model = copy.deepcopy(model)

    snapshot_path = path + 'epoch_' + str(epoch_num) + '.pt'
    checkpoint = torch.load(snapshot_path)

    model.load_state_dict(checkpoint['model_state_dict'])

    if arch == 'resnet':
        target_layer, target_sublayer, target_subsublayer = layer_info
        for layer, (name, modules) in enumerate(model._modules.items()):
            # print(f'layer: {layer}')
            if layer == target_layer:
                for sublayer, (name, submodule) in enumerate(modules._modules.items()):
                    # print(f'sublayer: {sublayer}')
                    if sublayer == target_sublayer:
                        for subsublayer, (name, module) in enumerate(submodule._modules.items()):
                            # print(f'subsublayer: {subsublayer}')
                            if subsublayer == target_subsublayer:
                                if isinstance(module, torch.nn.modules.conv.Conv2d):
                                    # print(f'inside: {layer}-{sublayer}-{subsublayer}')
                                    old_weights = module.weight
                                    old_weights = old_weights.data.cpu().numpy()
                                    old_weights = torch.from_numpy(old_weights).float()
                                    return old_weights
    elif arch == 'vgg':
        for layer, module in enumerate(model.features):
            if isinstance(module, torch.nn.modules.conv.Conv2d) and layer == layer_info:
                old_weights = module.weight
                old_weights = old_weights.data.cpu().numpy()
                old_weights = torch.from_numpy(old_weights).float()
                return old_weights


def make_new_conv(module, group_info = None):
    in_channels = module.in_channels
    groups = module.groups

    if group_info:
        n_clusters, pruning_rate, kernel_gcd = group_info
        number_of_unpruned_kernels = float((1 - pruning_rate) * kernel_gcd)

        if not number_of_unpruned_kernels.is_integer:
            logger.error(f'Should have int amount of unpruned kernels, now with: (1 - {pruning_rate}) * {kernel_gcd} = {number_of_unpruned_kernels}')
            os.exit()



        number_of_unpruned_kernels = int(number_of_unpruned_kernels)
        in_channels = module.in_channels * number_of_unpruned_kernels // (kernel_gcd / n_clusters)

        in_channels = int(in_channels)
        groups = n_clusters


    new_conv = torch.nn.Conv2d(in_channels = in_channels,
                       out_channels=module.out_channels,
                       kernel_size=module.kernel_size,
                       stride=module.stride,
                       padding=module.padding,
                       dilation=module.dilation,
                       groups = groups,
                       bias=False)

    if group_info:
        return new_conv, number_of_unpruned_kernels
    else:
        return new_conv

def get_preferred_permutation_matrix(criterion_result, old_weights, old_out_channels, criterion, n_clusters = 8, clustering_method='ALL'):
    if criterion == 'tickets magnitute increase':
        permutation_matrix_candidate_list = []
        for candiate_criterion in criterion_result:
            candidate_permutation_matrix, candidate_clustering_method = get_cluster_permutation_matrix(candiate_criterion, old_weights, old_out_channels, n_clusters = n_clusters, clustering_method= clustering_method)
            permutation_matrix_candidate_list.append((candidate_permutation_matrix, candidate_clustering_method))

        methods_candidate_list = [i[1] for i in permutation_matrix_candidate_list]
        methods_occurrence_counter = collections.Counter(methods_candidate_list)

        max_occurrence = methods_occurrence_counter.most_common(1)[0][1]
        max_occurrence_methods = []
        for a_method, a_method_occurrence in methods_occurrence_counter.most_common():
            if a_method_occurrence == max_occurrence:
                max_occurrence_methods.append(a_method)



        candidate_methods_list = [i[1] for i in permutation_matrix_candidate_list]
        candidate_methods_sequence_counter_dict = longest_method_sequence(candidate_methods_list)

        if len(max_occurrence_methods) == 1:
            preferred_clustering_method = max_occurrence_methods[0]
        else:
            max_occurrence_methods_sequence_counter_dict = {k:v for k, v in candidate_methods_sequence_counter_dict.items() if k in max_occurrence_methods}

            max_occurrence_methods_sequence_counter_dict = dict(sorted(max_occurrence_methods_sequence_counter_dict.items(), key = lambda t: t[1], reverse = True))
            preferred_clustering_method = list(max_occurrence_methods_sequence_counter_dict.keys())[0]





        for candidate_permutation_matrix, candidate_clustering_method in permutation_matrix_candidate_list:
            if candidate_clustering_method == preferred_clustering_method:
                preferred_permutation_matrix = candidate_permutation_matrix

                clustering_methods_occurrence_info = f'Occurrence: {str(methods_occurrence_counter)[7:]};'

                clustering_methods_sequence_info = f'\tSequence: {str(candidate_methods_sequence_counter_dict)[27:]};'

                clustering_methods_info = clustering_methods_occurrence_info + clustering_methods_sequence_info

                return preferred_permutation_matrix, preferred_clustering_method, clustering_methods_info, candidate_methods_list


    else:
        preferred_permutation_matrix, preferred_clustering_method = get_cluster_permutation_matrix(criterion_result, old_weights, old_out_channels, n_clusters = n_clusters, clustering_method= clustering_method)

        return preferred_permutation_matrix, preferred_clustering_method, None, None


def get_cluster_permutation_matrix(criterion_result, old_weights, old_out_channels, n_clusters = 8, clustering_method='ALL'):
    permutation_matrices = []
    score_dicts = []

    old_weights_normalized = torch.from_numpy(old_weights).float()
    old_weights_normalized = F.normalize(old_weights_normalized, p=2, dim=1).numpy()

    if clustering_method == 'KPCA' or clustering_method == 'ALL':
        kpca = KernelPCA(n_components=None, kernel='precomputed')
        lambda_kpca = 0.5
        kernel_old = lambda_kpca*pow(np.dot(old_weights, old_weights.T), 2) + (1-lambda_kpca)*rbf_kernel(old_weights)
        old_kpca = kpca.fit_transform(kernel_old)

        try:
            permutation_matrix, labels = get_equal_k_means_permutation_matrix(old_kpca, n_clusters)
            if clustering_method == 'ALL':
                score_dicts.append(get_clusters_LTH_scores(criterion_result, labels))
                permutation_matrices.append((permutation_matrix, 'K-PCA'))
        except ValueError:
            # pass
            if clustering_method != 'ALL':
                adj_mat = squareform(pdist(old_weights_normalized, metric='cosine', p=2))
                V_K = Spectral_Clustering(adj_mat, K= n_clusters, sim_graph='mutual_knn', knn=16, normalized=1)
                clustering_method = 'SPECTRAL (KPCA fallback)'
                permutation_matrix, labels = get_equal_k_means_permutation_matrix(V_K, n_clusters)


    if clustering_method == 'SPECTRAL' or clustering_method == 'ALL':
        adj_mat = squareform(pdist(old_weights_normalized, metric='cosine', p=2))
        V_K = Spectral_Clustering(adj_mat, K= n_clusters, sim_graph='mutual_knn', knn=16, normalized=1)
        permutation_matrix, labels = get_equal_k_means_permutation_matrix(V_K, n_clusters)
        if clustering_method == 'ALL':
            score_dicts.append(get_clusters_LTH_scores(criterion_result, labels))
            permutation_matrices.append((permutation_matrix, 'Spectral'))

    if clustering_method == 'DBSCAN' or clustering_method == 'ALL':
        equal_dbscan = Same_Size_DBSCAN(X = old_weights_normalized, n_clusters = n_clusters, step = 0.05, display_logs = False)
        equal_group_labels = equal_dbscan.fit(priotize_noise = False)
        permutation_matrix = get_permutation_matrix(old_weights_normalized, equal_group_labels)
        if clustering_method == 'ALL':
            score_dicts.append(get_clusters_LTH_scores(criterion_result, equal_group_labels))
            permutation_matrices.append((permutation_matrix, 'DBSCAN'))


    if clustering_method == 'ALL':
        best_cluster_method_index, rank_matrix = rank_cluster_results(score_dicts)
        best_permutation_matrix = permutation_matrices[best_cluster_method_index][0]
        best_cluster_method = permutation_matrices[best_cluster_method_index][1]
    else:
        best_permutation_matrix = permutation_matrix
        best_cluster_method = clustering_method

    return best_permutation_matrix, best_cluster_method



def get_out_index(permutation_matrix):
    q = []
    n, m = permutation_matrix.shape
    for j in range(n):
        for i in range(m):
            if permutation_matrix[j, i] == 1:
                q.append(i)

    q = np.array(q)
    q = torch.from_numpy(q)
    return q


def get_layer_weights(model, layer_info):

    target_layer, target_sublayer, target_subsublayer = layer_info
    for layer, (name, modules) in enumerate(model._modules.items()):
        if layer == target_layer:
            for sublayer, (name, submodule) in enumerate(modules._modules.items()):
                if sublayer == target_sublayer:
                    for subsublayer, (name, module) in enumerate(submodule._modules.items()):
                        if subsublayer == target_subsublayer and isinstance(module, torch.nn.modules.conv.Conv2d):
                            old_weights = module.weight
                            old_weights = old_weights.data.cpu().numpy()
                            old_weights = torch.from_numpy(old_weights).float()
                            return old_weights, module
def stretch_conv(w):
    filter_num, kernel_num, kernel_m, kernel_n = w.shape
    w = w.reshape(filter_num, kernel_num*kernel_m*kernel_n)
    return w

def get_filters_LTH_metrics(w_f, model, layer_info, ticket_start_epoch, ticket_end_epoch, snapshot_path, criterion = 'movement', ticket_step = 1, arch = 'resnet'):

    if criterion == 'movement':
        result = []
        w_i = get_epoch_snapshot_weights(model, layer_info, ticket_start_epoch, snapshot_path, arch = arch)

        w_i = stretch_conv(w_i)
        w_f = stretch_conv(w_f)
        w_i = StandardScaler().fit_transform(w_i)
        w_f = StandardScaler().fit_transform(w_f)

        for filter_i, filter_f in zip(w_i, w_f):
            result.append(np.absolute(filter_f - filter_i))
        result = [sum(i) for i in result]

        return result

    elif criterion == 'magnitute increase':
        result = []
        w_i = get_epoch_snapshot_weights(model, layer_info, ticket_start_epoch, snapshot_path, arch = arch)

        w_i = stretch_conv(w_i)
        w_f = stretch_conv(w_f)
        w_i = StandardScaler().fit_transform(w_i)
        w_f = StandardScaler().fit_transform(w_f)
        for filter_i, filter_f in zip(w_i, w_f):
            result.append(np.absolute(np.absolute(filter_f) - np.absolute(filter_i)))
        result = [sum(i) for i in result]

        return result

    elif criterion == 'large final':
        result = []
        w_f = stretch_conv(w_f)
        for filter_f in w_f:
            result.append(filter_f)
        result = [sum(i) for i in result]

        return result

    elif criterion == 'tickets magnitute increase':
        candidate_results = []
        w_f = stretch_conv(w_f)

        for epoch_i in range(ticket_start_epoch, ticket_end_epoch + 1, ticket_step):
            w_i = get_epoch_snapshot_weights(model, layer_info, epoch_i, snapshot_path, arch = arch)
            w_i = stretch_conv(w_i)
            result = []
            for filter_i, filter_f in zip(w_i, w_f):
                result.append(np.absolute(np.absolute(filter_f) - np.absolute(filter_i)))
            result = [sum(i) for i in result]
            candidate_results.append(result)

        return candidate_results



    elif criterion == 'total movement':
        weights_since_ticket = []
        snapshots_folder_path = snapshot_path
        snapshots_list = os.listdir(snapshot_path)
        snapshots_list = [i for i in snapshots_list if '.pt' in i]

        snapshots_list.sort(key = lambda t: int(t.split('_')[1][:-3]))

        for snapshot_epoch_num, snapshot_file in enumerate(snapshots_list):
            if snapshot_epoch_num >= ticket_start_epoch:
                epoch_weights = get_epoch_snapshot_weights(model, layer_info, snapshot_epoch_num, snapshot_path)
                weights_since_ticket.append(epoch_weights)

        result = []
        w_i = None
        for j in range(1, len(weights_since_ticket)):
            w_j = StandardScaler().fit_transform(stretch_conv(weights_since_ticket[j]))
            if w_i is None:
                w_i = StandardScaler().fit_transform(stretch_conv(weights_since_ticket[j-1]))
            interval_result = []
            for filter_i, filter_j in zip(w_i, w_j):
                interval_result.append(np.absolute(filter_j - filter_i))
            interval_result = [sum(i) for i in interval_result]

            if not result:
                result = interval_result
            else:
                result = [(old_r + new_r) for old_r, new_r in zip(result, interval_result)]
            w_i = w_j

        return result

    else:
        logger.info(f'No such criterion as \'{criterion}\'.')
        sys.exit(0)


def get_clusters_LTH_scores(criterion_result, labels):

    labels_set = sorted(list(set(labels)))
    score_dict = dict.fromkeys(labels_set, 0)
    for i, label in enumerate(labels):
        score_dict[label] += criterion_result[i]

    return score_dict

def evaluate_clusters_LTH_scores(score_dict):
    scores = [i for i in score_dict.values()]
    scores.sort(reverse = True)
    intervals = []
    for i in range(len(scores) - 1):
        intervals.append(scores[i] - scores[i+1])
    intervals_mean = sum(intervals)/len(intervals)
    intervals_var = np.var(intervals)

    return intervals_mean, intervals_var


def rank_cluster_results(score_dict_list):

    rank_matrix = []
    for i, score_dict in enumerate(score_dict_list):
        intervals_mean, intervals_var = evaluate_clusters_LTH_scores(score_dict)
        rank_matrix.append([i, intervals_mean, intervals_var])

    mean_rank = ss.rankdata([i[-2] for i in rank_matrix])
    var_rank = ss.rankdata([i[-1] for i in rank_matrix])

    rank_matrix = [i + [j] + [k] for i, j, k in zip(rank_matrix, mean_rank, var_rank)]
    # best_clustering = sorted(rank_matrix, key = lambda t: t[-2] + 0.5 * t[-1])[0]
    rank_matrix = sorted(rank_matrix, key = lambda t: 0.5 * t[-2] + t[-1])
    best_clustering_index = rank_matrix[0][0]

    return best_clustering_index, rank_matrix


def get_LTH_cluster_result(LTH_result, n_clusters = 8):
    LTH_result = [(i, v) for i, v in enumerate(LTH_result)]
    LTH_result.sort(reverse=True, key = lambda t: t[1]) # sort by LTH criterion value.

    single_cluster_size = len(LTH_result)/n_clusters

    LTH_clusters = []

    current_cluster_index = 0
    number_of_pairs = int(len(LTH_result)/2)

    for i in range(number_of_pairs):
        if current_cluster_index > (n_clusters - 1):
            current_cluster_index = 0
        highest_candidate = LTH_result.pop(0)
        LTH_clusters.append((current_cluster_index,) + highest_candidate)
        lowest_candidate = LTH_result.pop(-1)
        LTH_clusters.append((current_cluster_index,) + lowest_candidate)

        current_cluster_index += 1

    LTH_clusters.sort(key = lambda t: t[1]) # sort by filter indices.

    LTH_clusters_labels = [i[0] for i in LTH_clusters]

    return LTH_clusters_labels

def get_LTH_filled_matrix(criterion_result, w):
    LTH_score_matrix = []

    for i in range(len(w)):
        filter_shape = w[i].shape
        LTH_score_submatrix = np.full(filter_shape, criterion_result[i])
        LTH_score_matrix.append(LTH_score_submatrix)


    LTH_score_matrix = np.array(LTH_score_matrix)

    return LTH_score_matrix


def select_LTH_preferred_filters(LTH_score_submatrix):
    # print('LTH_score_submatrix shape', LTH_score_submatrix.shape)
    LTH_filters_scores = [(i, v[0][0][0]) for i, v in enumerate(LTH_score_submatrix)]

    LTH_filters_scores.sort(reverse = True, key = lambda t : t[1])

    LTH_filters_scores_index = [i[0] for i in LTH_filters_scores]

    return LTH_filters_scores_index

def longest_method_sequence(candidate_methods_list):
    candidate_methods = list(set(candidate_methods_list))

    candidate_methods_sequence_counter_dict = collections.defaultdict(int)
    candidate_methods_sequence_max_dict = collections.defaultdict(int)

    last_method = 'placeholder method'
    for i in candidate_methods_list:
        if i != last_method:
            candidate_methods_sequence_counter_dict[i] = 0
        else:
            candidate_methods_sequence_counter_dict[i] += 1
            if candidate_methods_sequence_counter_dict[i] > candidate_methods_sequence_max_dict[i]:
                candidate_methods_sequence_max_dict[i] = candidate_methods_sequence_counter_dict[i]
        last_method = i

    return candidate_methods_sequence_max_dict


def get_prune_mask_preserved_index(prune_mask):
    preserved_index = []

    for i, mask in enumerate(prune_mask):
        mask = mask.data.cpu().numpy()
        # print('mask type', type(mask))
        # print('mask.shape', mask.shape)
        if not np.all(mask == 0):
            preserved_index.append(i)

    return preserved_index



def zeroize_pruned_filters(model):

    for layer, (name, modules) in enumerate(model._modules.items()):
        if layer == 2 or layer == 3 or layer == 4:
            for sublayer, (name, submodule) in enumerate(modules._modules.items()):
                old_block = modules[sublayer]

                conv1_prune_mask, conv2_prune_mask = old_block.prune_mask

                conv1_old_weights = old_block.conv_a.weight.cuda()
                conv2_old_weights = old_block.conv_b.weight.cuda()
                conv1_old_weights = conv1_old_weights.data.cpu().numpy()
                conv2_old_weights = conv2_old_weights.data.cpu().numpy()

                conv1_old_weights = Variable(torch.from_numpy(conv1_old_weights)).cuda()
                conv2_old_weights = Variable(torch.from_numpy(conv2_old_weights)).cuda()
                conv1_new_weights = torch.mul(conv1_old_weights.double(), conv1_prune_mask, out=None)
                conv2_new_weights = torch.mul(conv2_old_weights.double(), conv2_prune_mask, out=None)

                old_block.conv_a.weight = torch.nn.Parameter(conv1_new_weights)
                old_block.conv_b.weight = torch.nn.Parameter(conv2_new_weights)

                old_block.conv_a.weight.data = old_block.conv_a.weight.type(torch.FloatTensor)
                old_block.conv_a.weight.data = old_block.conv_a.weight.data.cuda()
                old_block.conv_b.weight.data = old_block.conv_b.weight.type(torch.FloatTensor)
                old_block.conv_b.weight.data = old_block.conv_b.weight.data.cuda()

def tmi_filter_ranking(tmi_scores):
    tmi_ranked = {i:[] for i in range(len(tmi_scores[0]))}
    for tmi_k in tmi_scores:
        tmi_k_ranked = [(i, v) for i, v in enumerate(tmi_k)]
        tmi_k_ranked.sort(key = lambda t: t[1], reverse = True)

        for filter_rank, (filter_i, _) in enumerate(tmi_k_ranked):
            tmi_ranked[filter_i] = tmi_ranked[filter_i] + [filter_rank]

    tmi_rank_sum = []
    for filter_i, filter_i_ranks in tmi_ranked.items():
        tmi_rank_sum.append((filter_i, sum(filter_i_ranks)))
    tmi_rank_sum.sort(key = lambda t: t[1])

    return [i[0] for i in tmi_rank_sum]







