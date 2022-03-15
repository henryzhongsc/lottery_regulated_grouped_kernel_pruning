from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.neighbors.nearest_centroid import NearestCentroid
from scipy.spatial.distance import pdist, squareform
import scipy.stats as ss

import numpy as np
import math
import sys


class Empety_Candidates_Error(Exception):
    pass


class Cluster_Size_Error(Exception):
    pass



def tune_params(decorated):
    # def purity_score(y_true, y_pred):
    #     # compute contingency matrix (also called confusion matrix)
    #     contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    #     # return purity
    #     return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

    def max_dist(X):
        D = squareform(pdist(X))
        return np.max(D)

    def f_range(start, stop, step):
        seq = [start]
        while start <= stop:
            seq.append(start + step)
            start += step
        return seq

    def inner(*args, **kwargs):
        n_clusters = kwargs['n_clusters']
        step = kwargs['step']
        X = kwargs['X']
        # tsne_dim_candidates = [i for i in range(2, X.shape[0] + 1)]
        tsne_dim_candidates = [i for i in range(2, 8)]
        min_sample_ratio_candidates = f_range(0.5, 1, 0.1)
        target_cluster_size = math.ceil(X.shape[0] / n_clusters)
        cluster_LUT = []


        # print(n_clusters, step, X.shape)

        for tsne_dim in tsne_dim_candidates:
            # print(f'tsne_dim: {tsne_dim}')
            candidate_tsne = TSNE(n_components = tsne_dim, random_state = 0, perplexity = target_cluster_size, method = 'exact')
            candidate_tsne_X = candidate_tsne.fit_transform(X)
            scaled_X = StandardScaler().fit_transform(candidate_tsne_X)
            max_distance = max_dist(scaled_X)*0.5
            eps_candidates = f_range(sys.float_info.min, max_distance, step)
            for min_samples_ratio in min_sample_ratio_candidates:
                min_samples = scaled_X.shape[0] / n_clusters * min_samples_ratio
                # print(f'min_samples: {min_samples}; min_samples_ratio: {min_samples_ratio}')
                eps_escape_counter = 0
                eps_escape_thld = int(len(eps_candidates) * 0.25)
                eps_escape_flag = False
                for eps in eps_candidates:
                    # print(f'start eps = {eps}')
                    candidate_dbscan = DBSCAN(eps = eps, min_samples = min_samples).fit(scaled_X)
                    dbscan_labels = candidate_dbscan.labels_
                    labels_set = set(dbscan_labels)
                    current_n_clusters = len(labels_set)
                    if n_clusters > 2:
                        if current_n_clusters <= 2:
                            eps_escape_counter += 1
                            if eps_escape_counter >= eps_escape_thld * 0.1 and eps_escape_flag == True:
                                # print(f'escape at eps = {eps}')
                                break
                            if eps_escape_counter >= eps_escape_thld:
                                break
                        else:
                            if current_n_clusters >= n_clusters * 0.5:
                                eps_escape_flag = True
                    clusters_sizes = [np.count_nonzero(dbscan_labels == i) for i in labels_set]
                    clusters_var = np.var(clusters_sizes)
                    log = f'tsne_dim = {tsne_dim}; eps = {eps : >5.3f}; min_samples = {min_samples : >5.3f} (ratio: {min_samples_ratio : >5.3f}); {current_n_clusters} clusters: {clusters_sizes}; var: {clusters_var}; '
                    # ##
                    # purity_score_r = 'purity:' + str(purity_score(X_label, dbscan_labels))
                    # log += purity_score_r
                    # ##
                    # print(log)
                    cluster_LUT.append([tsne_dim, eps, min_samples, min_samples_ratio, current_n_clusters, clusters_var, clusters_sizes])

        clusters_number_rank = ss.rankdata([abs(i[-3] - (n_clusters+1)) for i in cluster_LUT])
        clusters_var_rank = ss.rankdata([i[-2] for i in cluster_LUT])
        cluster_LUT = [v + [j] + [i] for v, i, j in zip(cluster_LUT, clusters_number_rank, clusters_var_rank)]
        best_trial = sorted(cluster_LUT, key = lambda t: t[-1] + 0.5 * t[-2])[0]
        # print('best_trial', best_trial)
        best_tsne_dim, best_eps, best_min_samples = best_trial[0], best_trial[1], best_trial[2]

        best_tsne = TSNE(n_components = best_tsne_dim, random_state = 0, perplexity = target_cluster_size, method = 'exact')
        best_tsne_X = best_tsne.fit_transform(X)
        best_scaled_X = StandardScaler().fit_transform(best_tsne_X)
        best_dbscan = DBSCAN(eps = best_eps, min_samples = best_min_samples).fit(best_scaled_X)
        best_dbscan_labels = best_dbscan.labels_
        best_log =  f'best params: \t tsne_dim = {best_tsne_dim} (perplexity = {target_cluster_size}); eps = {best_eps : >5.3f}; min_samples = {best_min_samples : >5.3f} (ratio: {best_trial[3] : >5.3f}; rank: {best_trial[-1], best_trial[-2]});\n' +  f'{best_trial[4]} clusters: {best_trial[-3]}; var: {best_trial[-4] : >5.3f}; '

        # ##
        # best_purity_score = 'purity: ' + str(purity_score(X_label, best_dbscan_labels)) + '.'
        # best_log += best_purity_score
        # ##
        if kwargs['display_logs']:
            print(best_log)

        kwargs['X'] = best_scaled_X
        kwargs['labels'] = best_dbscan_labels
        decorated(*args, **kwargs)
    return inner



class Same_Size_DBSCAN():
    @tune_params
    def __init__(self, X = None, labels = None, n_clusters=8, step = 0.05, display_logs = True):
        self.X = X
        self.labels = labels
        self.n_clusters = n_clusters
        self.clusters_status, self.current_n_clusters = self.get_clusters_status()
        self.priotize_noise = False
        self.display_logs = display_logs
        self.target_cluster_size = math.ceil(self.X.shape[0] / n_clusters)
        if self.X.shape[0] % n_clusters != 0:
            raise Cluster_Size_Error(f'{self.X.shape[0]} entries cannot divide to {n_clusters} clusters')


    def get_clusters_status(self):
        clf = NearestCentroid()
        clf.fit(self.X, self.labels)
        cluster_centroids = clf.centroids_
        labels_set = sorted(list(set(self.labels)))
        current_n_clusters = len(labels_set)
        cluster_size_per_label = [np.count_nonzero(self.labels == i) for i in labels_set]

        clusters_status = dict()
        for label, cluster_size, cluster_centroid in zip(labels_set, cluster_size_per_label, cluster_centroids):
            clusters_status[label] = (cluster_size, cluster_centroid)

        return clusters_status, current_n_clusters

    def fit(self, priotize_noise = False):
        self.priotize_noise = priotize_noise
        return self.get_equal_clusters()

    def get_equal_clusters(self):
        counter = 1
        while self.current_n_clusters != self.n_clusters or any(a_cluster_status[0] != self.target_cluster_size for a_cluster_status in self.clusters_status.values()):
            if self.display_logs:
                print(f'#{counter} Iteration')
            counter += 1

            if self.current_n_clusters >= self.n_clusters:
                candidate_entries = self.get_candidate_entries_for_reassignment()
                recent_assignment_msg = self.reassign_an_entry(candidate_entries)
            else:
                recent_assignment_msg = self.seperate_a_new_cluster()

            self.clusters_status, self.current_n_clusters = self.get_clusters_status()
            if self.display_logs:
                print(f'{recent_assignment_msg}\n'\
                f'current_n_clusters: {self.current_n_clusters} (target_n_clusters: {self.n_clusters})\n'\
                f'current_cluster_size: {[a_cluster[0] for a_cluster in self.clusters_status.values()]} (target_cluster_size: {self.target_cluster_size})\n'\
                f'labels: {[a_label for a_label in self.clusters_status.keys()]}\n')
        return self.labels

    def get_candidate_entries_for_reassignment(self):

        # current_n_clusters, cluster_size_per_label, labels_set = get_clusters_status(X, labels)
        # target_cluster_size = math.ceil(X.shape[0]/n_clusters)

        self.excessive_clusters_list = []
        if self.current_n_clusters > self.n_clusters:
            num_excessive_clusters = self.current_n_clusters - self.n_clusters
            sorted_clusters_sizes = sorted([(label, v[0]) for label, v in self.clusters_status.items()])
            self.excessive_clusters_list.extend([label for label, _ in sorted_clusters_sizes[: num_excessive_clusters]])
            if -1 in self.excessive_clusters_list:
                self.excessive_clusters_list.pop(self.excessive_clusters_list.index(-1))
                self.excessive_clusters_list = [-1] + self.excessive_clusters_list

        candidate_entries = []
        for a_label, status in self.clusters_status.items():
            cluster_size, cluster_centroid = status[0], status[1]
            if self.current_n_clusters == self.n_clusters and cluster_size <= self.target_cluster_size:
                continue

            cluster_X = [(i, self.X[i]) for i, v in enumerate(self.labels) if v == a_label]
            dist_to_self_centroid = [(i, np.linalg.norm(x - cluster_centroid)) for i, x in cluster_X]
            dist_to_self_centroid.sort(reverse = True, key = lambda t: t[1])

            if self.current_n_clusters == self.n_clusters and cluster_size > self.target_cluster_size:
                dist_to_self_centroid = dist_to_self_centroid[: (cluster_size - self.target_cluster_size)]
                candidate_entries.extend([(i, dist) for i, dist in dist_to_self_centroid])
            elif self.current_n_clusters > self.n_clusters and a_label in self.excessive_clusters_list:
                candidate_entries.extend([(i, dist) for i, dist in dist_to_self_centroid])

        candidate_entries.sort(reverse = True, key = lambda t: t[1])

        if len(candidate_entries) == 0:
            raise Empety_Candidates_Error(f'candidate_entries: {candidate_entries}')

        return candidate_entries



    def reassign_an_entry(self, candidate_entries):
        clusters_sizes = [(label, status[0]) for label, status in self.clusters_status.items()]
        filled_clusters = [label for label, size in clusters_sizes if size >= self.target_cluster_size]

        dist_delta_list = []
        for entry_index, entry_dist_to_self_centroid in candidate_entries:
            x = self.X[entry_index]
            x_label = self.labels[entry_index]

            dist_to_centroids = []
            for dest_label in self.clusters_status.keys():
                if dest_label != x_label and dest_label not in filled_clusters and dest_label not in self.excessive_clusters_list:
                    dest_centroid = self.clusters_status[dest_label][1]
                    dist_to_centroids.append((dest_label, np.linalg.norm(x - dest_centroid)))

            sorted_dist_to_centroids = sorted(dist_to_centroids, reverse = True, key = lambda t: t[1])
            top_assignment_delta = sorted_dist_to_centroids[0][1]
            if len(sorted_dist_to_centroids) > 1:
                top_assignment_delta = sorted_dist_to_centroids[0][1] - sorted_dist_to_centroids[1][1]
            preferred_dest_label = sorted_dist_to_centroids[0][0]

            dist_delta_list.append([entry_index, top_assignment_delta, preferred_dest_label])

        if not self.priotize_noise:
            dist_delta_list.sort(reverse = True, key = lambda t: t[1])
        else:
            noise_delta_list = []
            non_noise_delta_list = []
            for i, delta, dest_label in dist_delta_list:
                if self.labels[i] == -1:
                    noise_delta_list.append((i, delta, dest_label))
                else:
                    non_noise_delta_list.append((i, delta, dest_label))
            dist_delta_list = sorted(noise_delta_list, reverse = True, key = lambda t: t[1]) + sorted(non_noise_delta_list, reverse = True, key = lambda t: t[1])

        reassigned_entry_index, reassigned_entry_new_label = dist_delta_list[0][0], dist_delta_list[0][2]
        original_label = self.labels[reassigned_entry_index]
        self.labels[reassigned_entry_index] = reassigned_entry_new_label

        assignment_msg = f'Reassigned X[{reassigned_entry_index}] from cluster {original_label} to {reassigned_entry_new_label}'
        return assignment_msg

    def seperate_a_new_cluster(self):
        oversized_clusters = [(label, status[0]) for label, status in self.clusters_status.items() if status[0] > self.target_cluster_size]
        cluster_to_seperate = sorted(oversized_clusters, reverse = True)[0]

        seperatable_entries = []
        for i, x in enumerate(self.X):
            if self.labels[i] == cluster_to_seperate[0]:
                dist_to_self_centroid = np.linalg.norm(x - self.clusters_status[self.labels[i]][1])
                seperatable_entries.append((i, dist_to_self_centroid))
        seperatable_entries.sort(reverse = True, key = lambda t: t[1])
        seperatable_entries = seperatable_entries[: self.target_cluster_size]

        new_label = max(self.labels) + 1
        seperatable_entries_indices = [entry_index for entry_index, _ in seperatable_entries]
        for i in seperatable_entries_indices:
            self.labels[i] = new_label

        assignment_msg = f'Reassigned X[{seperatable_entries_indices}] (len: {len(seperatable_entries_indices)}) from cluster {cluster_to_seperate} to {new_label}'
        return assignment_msg



