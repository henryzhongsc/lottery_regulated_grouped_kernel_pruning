import numpy as np
import pandas as pd
import itertools

def select_kernel_from_group(sim_matrix, pruning_rate = 0.5, dim = None, eval_pruned_kernel_relationship = False, eval_kept_kernel_ratio = 'all', cost_balancer = 1, show_analysis = False, ablation_reverse = False):

    def evaluate_selection_cost(selection_list):
        selected_kernel_pairs = list(itertools.combinations(selection_list, 2))
        selected_cost = 0
        for i in selected_kernel_pairs:
            selected_cost += sim_matrix[i[0]][i[1]]
        cost = selected_cost
        analysis_info = f'{cost:.3f} = {selected_cost:.3f} kept ({len(selection_list)} kept kernels ({len(selected_kernel_pairs)} pairs)'

        if eval_pruned_kernel_relationship:
            pruned_cost = 0
            pruned_kernels = list(set([i for i in range(dim)]) - set(selection_list))
            nonlocal eval_kept_kernel_ratio
            eval_kept_kernel_number = len(selection_list) if eval_kept_kernel_ratio == 'all' else int(eval_kept_kernel_ratio * len(selection_list))

            # print(f'eval_kept_kernel_number {eval_kept_kernel_number} = eval_kept_kernel_ratio {eval_kept_kernel_ratio} * len(selection_list) {len(selection_list)}')
            # pruned_kernel_pairs = list(itertools.product(pruned_kernels, selection_list))
            # for i in pruned_kernel_pairs:
            #     pruned_cost += sim_matrix[i[0]][i[1]]
            for i in pruned_kernels:
                # min_dis_pair_cost = max([sim_matrix[j][i] for j in selection_list])
                min_dis_pairs_cost = sorted([sim_matrix[j][i] for j in selection_list], reverse = True)[:eval_kept_kernel_number]
                pruned_cost += sum(min_dis_pairs_cost)

            nonlocal cost_balancer
            if cost_balancer == 'auto':
                try:
                    cost_balancer = (len(selected_kernel_pairs) / (len(pruned_kernels) *  eval_kept_kernel_number))
                except ZeroDivisionError:
                    cost_balancer = 0

            cost = selected_cost - (pruned_cost * cost_balancer)
            analysis_info = f'{cost:.3f} = {selected_cost:.3f} kept - {cost_balancer:.3f} * {pruned_cost:.3f} pruned [{len(selection_list)} kept kernels ({len(selected_kernel_pairs)} pairs) & {eval_kept_kernel_number}-{len(pruned_kernels)} kept-pruned kernal pairs evaluated ({len(pruned_kernels) *  eval_kept_kernel_number} pairs)]'


        return cost, analysis_info

    dim = sim_matrix.shape[0] if dim is None else dim
    remained_kernel_capacity = int((1 - pruning_rate) * (dim + 1))

    all_selection_list = []

    for initial_kernel_index in range(dim):
        current_selection_list = [initial_kernel_index]
        if not ablation_reverse:
            cost_matrix = np.full((dim - 1, dim), np.inf)
        else:
            cost_matrix = np.full((dim - 1, dim), -np.inf)
        cost_matrix[0] = sim_matrix[initial_kernel_index]

        # for row in range(1, dim - 1):
        for row in range(1, dim):
            if len(current_selection_list) == remained_kernel_capacity:
                break
            if not ablation_reverse:
                previous_kernel_cost = cost_matrix[row - 1].min()
            else:
                previous_kernel_cost = cost_matrix[row - 1].max()
            previous_kernel_index = int(np.where(cost_matrix[row - 1] == previous_kernel_cost)[0][0])
            if not previous_kernel_index == current_selection_list[-1]:
                current_selection_list.append(previous_kernel_index)
            for col in range(dim):
                if col in current_selection_list:
                    continue
                # print(previous_kernel_index, row, col, len(current_selection_list), current_selection_list)
                # print(cost_matrix)
                # print('\n')
                cost_matrix[row][col] = cost_matrix[row-1][col] + sim_matrix[current_selection_list[-1]][col]
        all_selection_list.append(current_selection_list)
        # print(initial_kernel_index)
        # print(cost_matrix)

    selection_list_cost_arena = []
    for i, candidate_selection_list in enumerate(all_selection_list):
        kept_kernels, analysis_info = evaluate_selection_cost(candidate_selection_list)
        selection_list_cost_arena.append((i, kept_kernels, analysis_info))

    min_selection_index, min_selection_cost, min_analysis_info = min(selection_list_cost_arena, key = lambda x:x[1])

    if show_analysis:
        for i, j in zip(all_selection_list, [k for k in selection_list_cost_arena]):
            print(f'candidate list: {i}; cost: {j[1]} as {j[2]}')

        print(f'Cost: {min_analysis_info};\n\t {len(all_selection_list[min_selection_index])} kernels kept: {all_selection_list[min_selection_index]} (pruning rate {pruning_rate});')

    return all_selection_list[min_selection_index]


    # return all_selection_list[min_selection_index], min_selection_cost

def find_optimal(sim_matrix, num_of_kept, approx_cost):
    dim = sim_matrix.shape[0]
    index_selection_list = list(itertools.combinations([i for i in range(dim)], num_of_kept))
    index_selection_combo = [list(itertools.combinations(i, 2)) for i in index_selection_list]
    index_selection_cost = []
    for index, a_selection in enumerate(index_selection_combo):
        selection_cost = 0
        for pair_a, pair_b in a_selection:
            selection_cost += sim_matrix[pair_a][pair_b]
        index_selection_cost.append((index, selection_cost))

    index_selection_cost.sort(key=lambda t: t[1])
    optimal_cost = index_selection_cost[0][1]

    if approx_cost == optimal_cost:
        print('equal!')
    elif approx_cost/optimal_cost >= 0.995 or approx_cost/optimal_cost >= 1.015:
        print(f'loosly equal! {approx_cost/optimal_cost}')
    else:
        print(f'not equal! optimal: {optimal_cost}; approx_cost: {approx_cost}')
