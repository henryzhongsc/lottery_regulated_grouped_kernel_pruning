import sys
import torch
import numpy as np
import json
import copy

import argparse

parser = argparse.ArgumentParser(description='extract_model_information')
parser.add_argument('--task', default='permutation', type=str, help='what info to extract')
parser.add_argument('--dataset', default='imagenet', type=str, help='dataset in use')
parser.add_argument('--resnet_type', default='bottleneck', type=str, help='ResNet type, basicblock or bottleneck.')
parser.add_argument('--model', type=str, help='path of pruned model')
parser.add_argument('--output', type=str, help='path of output file model')



args = parser.parse_args()

print(f'Model {args.model} loaded (task: {args.task}, output: {args.output}, dataset: {args.dataset}, resnet_type: {args.resnet_type})')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.load(args.model)
model.to(device)

output_list = []

imagenet_target_layer_list = [4, 5, 6, 7]
cifar_target_laryer_list = [2, 3, 4]
tiny_imagenet_target_layer_list_basicblock = [2, 3, 4]
tiny_imagenet_target_layer_list_bottleneck = [2, 3, 4, 5]

if args.dataset == 'imagenet':
    target_layer_list = imagenet_target_layer_list
elif args.dataset == 'cifar10':
    target_layer_list = cifar_target_laryer_list
elif args.dataset == 'tiny_imagenet' and args.resnet_type == 'basicblock':
    target_layer_list = tiny_imagenet_target_layer_list_basicblock
elif args.dataset == 'tiny_imagenet' and args.resnet_type == 'bottleneck':
    target_layer_list = tiny_imagenet_target_layer_list_bottleneck
else:
    print(f'Invalid dataset input {args.dataset}')

print(model)

for layer, (name, modules) in enumerate(model._modules.items()):
    if layer in target_layer_list:
        for sublayer, (name, submodule) in enumerate(modules._modules.items()):
            print(f'{layer}-{sublayer}: {name}; {type(modules)}-{type(submodule)}')
            current_block = modules[sublayer]
            block_clustering_info = copy.deepcopy(current_block.candidate_methods_list)
            block_kernel_pruning_info = copy.deepcopy(current_block.preserved_kernel_index)

            for subsublayer, (name, module) in enumerate(submodule._modules.items()):
                if isinstance(module, torch.nn.modules.conv.Conv2d):
                    conv_shape = module.weight.cuda().shape

                    if args.task == 'kernel':
                        try:
                            conv_kernel_di = block_kernel_pruning_info.pop(0)
                        except IndexError:
                            print(f'Skipped: Layer {layer}-{sublayer}-{subsublayer} due to such layer is not pruned')
                            continue

                    else:
                        try:
                            conv_permutation_matrix, conv_method, conv_candidate_methods = block_clustering_info.pop(0)
                        except IndexError:
                            print(f'Skipped: Layer {layer}-{sublayer}-{subsublayer} due to such layer is not pruned')
                            continue


                    if args.task == 'permutation':
                        output_list.append((conv_permutation_matrix.tolist(), conv_method))
                    elif args.task == 'method':
                        output_list.append(conv_method)
                    elif args.task == 'kernel':
                        output_list.append(conv_kernel_di)
                    else:
                        print(f'Invalid task {args.task} given.')
                        sys.exit()

                    if args.task == 'kernel':
                        print(f'Extracted: Layer {layer}-{sublayer}-{subsublayer}; Shape {conv_shape}; di: {conv_kernel_di}')
                    else:
                        print(f'Extracted: Layer {layer}-{sublayer}-{subsublayer}; Shape {conv_shape}; Method: {conv_method}')


with open(args.output, "w+") as output_f:
    json.dump(output_list, output_f, indent = 4)

print(f'Model {args.model} information extracted and saved as {args.output}')


