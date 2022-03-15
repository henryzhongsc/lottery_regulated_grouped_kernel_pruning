import warnings
# warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore')

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import os
import sys
import argparse
import copy
import json
import datetime
import logging
logger = logging.getLogger()
import numpy as np
np.seterr(all="ignore")

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import random
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)







start_time = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f%Z")


parser = argparse.ArgumentParser(description='PyTorch training/testing')
parser.add_argument('--exp_desc', default='', type=str, help='experriment description')
parser.add_argument('--setting_dir', type=str, help='path of setting JSON file')
parser.add_argument('--dataset', default='cifar10', type=str, help='decide which dataset to run on.')
parser.add_argument('--dataset_dir', default='../data', type=str, help='folder path of dataset.')
parser.add_argument('--model_dir', default='', type=str, help='path of pretrain model')
parser.add_argument('--model_state_dict_dir', default='', type=str, help='path of pretrain model')
parser.add_argument('--snapshot_folder_dir', default='', type=str, help='path of model training snapshots')
parser.add_argument('--output_folder_dir', default='', type=str, help='path of output model')

parser.add_argument('--multi_prune_flag', default=False, type=bool, help='prune all conv layes or just 3x3, False means just prune 3x3')
parser.add_argument('--resnet_type', default='basicblock', type=str, help='ResNet type, basicblock or bottleneck.')
parser.add_argument('--zero_mask', default=False, type=bool, help='zero mask pruned network')
parser.add_argument('--baseline', default=False, type=bool, help='test baseline before prune/group')
# parser.add_argument('--iterative_flag', default=False, type=bool, help='prune iteratively or as one-shot')
parser.add_argument('--task', default='finetune', type=str, help='Select from one of the following tasks: finetune, train, test')
parser.add_argument('--gpu', default='', help='gpu available')


args = parser.parse_args()




with open(args.setting_dir) as setting_f:
    global setting
    setting = json.load(setting_f)

if args.output_folder_dir is not '':
    setting['management']['output_folder_dir'] = args.output_folder_dir
    if args.output_folder_dir[-1] != '/':
        setting['management']['output_folder_dir']  += '/'
if not os.path.isdir(setting['management']['output_folder_dir']):
    os.makedirs(setting['management']['output_folder_dir'])

log_formatter = logging.Formatter("%(asctime)s | %(levelname)s : %(message)s")
logger = logging.getLogger()
file_handler = logging.FileHandler(setting['management']['output_folder_dir'] + 'experiment.log', mode = 'w')
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)



logger.info(f'Experiment {args.exp_desc} starts at {start_time}.')
logger.info(f'Parsed setting file from {args.setting_dir}...')
if args.gpu is not '':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

setting['results'] = dict()
if args.exp_desc is not '':
    setting['management']['exp_desc'] = args.exp_desc
else:
    setting['management']['exp_desc'] = None

if args.model_dir is not '':
    setting['management']['model_dir'] = args.model_dir
if args.snapshot_folder_dir is not '':
    setting['management']['snapshot_folder_dir'] = args.snapshot_folder_dir
    if args.snapshot_folder_dir[-1] != '/':
        setting['management']['snapshot_folder_dir'] += '/'

setting['management']['multi_prune_flag'] = args.multi_prune_flag
setting['management']['task'] = args.task
setting['management']['dataset'] = args.dataset
setting['management']['dataset_dir'] = args.dataset_dir


logger.info(f"==> Preparing dataset: {setting['management']['dataset']}")
if args.task != 'prune':
    if setting['management']['dataset'] == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

        trainset = torchvision.datasets.CIFAR10(root=setting['management']['dataset_dir'] , train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=setting['train_params']['train_batch_size'], shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root=setting['management']['dataset_dir'] , train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=setting['train_params']['test_batch_size'], shuffle=False, num_workers=2)

    elif setting['management']['dataset'] == 'imagenet':
        transform_train = transforms.Compose([
            transforms.Scale(256),
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4467), (0.2471, 0.2435, 0.2616)),
        ])

        transform_test = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4467), (0.2471, 0.2435, 0.2616)),
        ])

        trainset = torchvision.datasets.ImageFolder(root=setting['management']['dataset_dir']+'train', transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=setting['train_params']['train_batch_size'], shuffle=True, num_workers=4, pin_memory=True)

        testset = torchvision.datasets.ImageFolder(root=setting['management']['dataset_dir']+'val', transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=setting['train_params']['test_batch_size'], shuffle=False, num_workers=4, pin_memory=True)

        # pass

    elif setting['management']['dataset'] == 'tiny_imagenet':

        transform_train = transforms.Compose([
            # transforms.RandomSizedCrop(32),
            transforms.Scale(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
        ])

        transform_test = transforms.Compose([
            # transforms.Scale(32),
            transforms.Scale(64),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
        ])


        trainset = torchvision.datasets.ImageFolder(root=setting['management']['dataset_dir']+'train', transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=setting['train_params']['train_batch_size'], shuffle=True, num_workers=1)

        testset = torchvision.datasets.ImageFolder(root=setting['management']['dataset_dir']+'val', transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=setting['train_params']['test_batch_size'], shuffle=False, num_workers=1)

        # pass
    else:
        logger.error(f"Invalid dataset_dir: {setting['management']['dataset']}")



logger.info(f"==> Building baseline model: {setting['management']['model_dir']} (dataset: {setting['management']['dataset']})")


if setting['management']['dataset'] == 'cifar10':
    from group_cifar import *
    from prune_cifar import *
elif setting['management']['dataset'] == 'imagenet' and setting['management']['multi_prune_flag']:
    from group_imagenet_multi import *
    from prune_imagenet_multi import *
elif setting['management']['dataset'] == 'imagenet' and not setting['management']['multi_prune_flag']:
    from group_imagenet import *
    from prune_imagenet import *
elif setting['management']['dataset'] == 'tiny_imagenet' and not setting['management']['multi_prune_flag'] and args.resnet_type == 'bottleneck':
    from models.resnet_bottleneck import *
    from group_imagenet import *
    from prune_imagenet import *
elif setting['management']['dataset'] == 'tiny_imagenet' and not setting['management']['multi_prune_flag'] and args.resnet_type == 'basicblock':
    print('is basicblock')
    from group_cifar import *
    from prune_cifar import *
else:
    logger.error(f"Invalid input on task: {setting['management']['task']} or multi_prune_flag {setting['management']['multi_prune_flag']}")
    sys.exit()




net = torch.load(args.model_dir)
# net = torch.load(args.model_dir, map_location=torch.device('cpu'))
if args.model_state_dict_dir != '':
    temp_checkpoint = torch.load(args.model_state_dict_dir)
    net.load_state_dict(temp_checkpoint['model_state_dict'])
    setting['management']['model_state_dict_dir'] = args.model_state_dict_dir
    logger.info(f"Model state dict at {setting['management']['model_state_dict_dir']} loaded into {args.model_dir}.")
net = net.to(device)
criterion = nn.CrossEntropyLoss()

logger.info(f"Model {setting['management']['model_dir']} loaded ({type(net)})")


def test(epoch=None):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100.*correct/total
    return acc, test_loss

baseline_test_acc = -1
setting['results']['baseline'] = baseline_test_acc
if args.baseline and not isinstance(net, GroupResNet) and not isinstance(net, ModifiedResNet):
    baseline_test_acc, _ = test()
    baseline_net_output_path = setting['management']['output_folder_dir'] + 'baseline'
    setting['results']['baseline'] = baseline_test_acc
    torch.save(net, baseline_net_output_path)
    logger.info(f'Model {setting["management"]["model_dir"]} {type(net)} has a baseline of {baseline_test_acc}.')

if not isinstance(net, GroupResNet) and args.task == 'finetune':
    if not isinstance(net, ModifiedResNet):
        original_model = copy.deepcopy(net)
        net = ModifiedResNet(net, setting, original_model, dataset = args.dataset, pruning_rate = sum(setting['prune_params']['pruning_rate']), pruned_flag = False).cuda()
        pruned_net_output_path = setting['management']['output_folder_dir'] + 'pruned'
        torch.save(net, pruned_net_output_path)
        logger.info(f'Model {setting["management"]["model_dir"]} {type(net)} now pruned.')

    net = GroupResNet(net, dataset = args.dataset).cuda()
    grouped_net_output_path = setting['management']['output_folder_dir'] + 'grouped'
    torch.save(net, grouped_net_output_path)
    logger.info(f'Model {setting["management"]["model_dir"]} {type(net)} now grouped.')
    net.to(device)
else:
    logger.info(f'Model {setting["management"]["model_dir"]} {type(net)} is already grouped or task is test or train (task: {args.task} {args.zero_mask}).')

lr = setting['train_params']['lr']
lr_step_size = setting['train_params']['lr_step_size']
weight_decay = setting['train_params']['weight_decay']
momentum = setting['train_params']['momentum']
gamma = setting['train_params']['gamma']

optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=gamma)

logger.info(f"==> Starting task: {setting['management']['task']} for {sum(setting['train_params']['epoch_num'])} epochs; multi_prune_flag: {setting['management']['multi_prune_flag']})")

def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if args.zero_mask:
            zeroize_pruned_filters(net)

    acc = 100.*correct/total
    return acc, loss


train_acc_list = []
train_loss_list = []
test_acc_list = []
test_loss_list = []
def process_epoch_results(current_epoch_num, epoch_snapshot, epoch_eval_info, save_subdir = None, overwrite_save_flag = False):
    train_acc, train_loss, test_acc, test_loss, best_acc, best_epoch, current_lr = epoch_eval_info
    epoch_dict = {
            'model_state_dict': epoch_snapshot,
            'epoch': current_epoch_num,
            'train_acc': train_acc,
            'train_loss': train_loss,
            'test_acc': test_acc,
            'test_loss': test_loss,
            }
    train_acc_list.append(train_acc)
    train_loss_list.append(train_loss)
    test_acc_list.append(test_acc)
    test_loss_list.append(test_loss)

    if test_acc >= best_acc:
        best_acc = test_acc
        best_epoch = current_epoch_num

    logger.info(f'Epoch #{current_epoch_num}')
    logger.info(f'\t train_acc: {train_acc}; train_loss: {train_loss :.3f}; test_loss: {test_loss :.3f}; lr: {current_lr}')
    logger.info(f'\t test_acc: {test_acc} (best: {best_acc} from #{best_epoch})')

    if save_subdir is not None:
        snapshot_save_path = setting['management']['output_folder_dir'] + str(save_subdir) + '/'
        if not os.path.isdir(snapshot_save_path):
            os.mkdir(snapshot_save_path)

        if not overwrite_save_flag:
            torch.save(epoch_dict, snapshot_save_path + f'epoch_{current_epoch_num}.pt')
        else:
            epoch_dict['optimizer'] = optimizer
            epoch_dict['scheduler'] = scheduler
            epoch_dict['model'] = net

            if (current_epoch_num + 1) % 10 == 0:
                torch.save(epoch_dict, snapshot_save_path + f'epoch_{current_epoch_num}.pt')
            torch.save(epoch_dict, snapshot_save_path + 'most_recent_checkpoint.pt')


    return epoch_dict

def finetune_model(num_of_epoch, baseline_net_output_path, finetuned_net_output_path):
    baseline_acc = -1
    if args.baseline:
        test_acc, _ = test()
        baseline_acc = test_acc
        logger.info(f'Model {setting["management"]["model_dir"]} has a baseline of {test_acc}.')
    torch.save(net, baseline_net_output_path)

    best_acc = baseline_acc
    best_epoch = -1
    for current_epoch_num in range(num_of_epoch):
        train_acc, train_loss = train(current_epoch_num)
        test_acc, test_loss = test(current_epoch_num)

        epoch_snapshot = net.state_dict()
        current_lr = scheduler.get_last_lr()

        epoch_eval_info = (train_acc, train_loss, test_acc, test_loss, best_acc, best_epoch, current_lr)
        current_epoch_dict = process_epoch_results(current_epoch_num, epoch_snapshot, epoch_eval_info, save_subdir = setting['management']['save_subdir'], overwrite_save_flag = setting['management']['overwrite_save_flag'])

        scheduler.step()

        if test_acc >= best_acc:
            best_acc = test_acc
            best_epoch = current_epoch_num
            torch.save(net, finetuned_net_output_path)

    logger.info(f'Model {setting["management"]["model_dir"]} finetuned {baseline_acc} --> {best_acc}.')
    return baseline_acc, best_acc


if setting['management']['task'] == 'test':
    # torch.backends.cudnn.benchmark=True
    # torch.cuda.synchronize()
    test_start_time = datetime.datetime.utcnow()
    test_acc, _ = test()
    test_end_time = datetime.datetime.utcnow()
    # torch.cuda.synchronize()
    test_duration = test_end_time - test_start_time
    test_duration_ms = test_duration.total_seconds() * 1000
    logger.info(f'Model {setting["management"]["model_dir"]} has a baseline of {test_acc};\nTest time: {test_start_time.strftime("%Y-%m-%dT%H:%M:%S.%f%Z")} --> {test_end_time.strftime("%Y-%m-%dT%H:%M:%S.%f%Z")} (duration: {test_duration_ms} ms).')

    sys.exit(0)



elif setting['management']['task'] == 'train':

    num_of_epoch = sum(setting['train_params']['epoch_num'])
    baseline_net_output_path = setting['management']['output_folder_dir'] + 'trained_only_baseline'
    finetuned_net_output_path = setting['management']['output_folder_dir'] + 'best_trained_only'
    baseline_test_acc, finetuned_test_acc = finetune_model(num_of_epoch, baseline_net_output_path, finetuned_net_output_path)

    sys.exit(0)


elif setting['management']['task'] == 'finetune':
    num_of_epoch = sum(setting['train_params']['epoch_num'])

    grouped_test_acc = -1
    if args.baseline:
        grouped_test_acc, _ = test()
    logger.info(f'Model {setting["management"]["model_dir"]} (type: {type(net)}) pruned and grouped (grouped baseline: {grouped_test_acc}); now to finetune with {num_of_epoch} epochs.\n')


    grouped_net_output_path = setting['management']['output_folder_dir'] + 'grouped'
    finetuned_net_output_path = setting['management']['output_folder_dir'] + 'finetuned'
    grouped_test_acc, finetuned_test_acc = finetune_model(num_of_epoch, grouped_net_output_path, finetuned_net_output_path)
    setting["results"]["grouped_test_acc"] = grouped_test_acc
    setting["results"]["best_acc"] = finetuned_test_acc

    logger.info(f"Model {setting['management']['model_dir']} pruned, grouped, and finetuned: {setting['results']['baseline']} --> {grouped_test_acc} --> {finetuned_test_acc}.\n")


end_time = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f%Z")
setting['results']['start_time'] = start_time
setting['results']['end_time'] = end_time
setting['results']['train_acc_list'] = train_acc_list
setting['results']['train_loss_list'] = [i.item() for i in train_loss_list]
setting['results']['test_acc_list'] = test_acc_list
setting['results']['test_loss_list'] = [i for i in test_loss_list]




setting_output_path = setting['management']['output_folder_dir'] + 'setting.json'

logger.info('setting', setting)
with open(setting_output_path, 'w+') as out_setting_f:
    logger.info(f'Saving setting file to {setting_output_path}...')
    json.dump(setting, out_setting_f, indent = 4)


logger.info(f'Experiment {args.exp_desc} (task: {setting["management"]["task"]}) ends at {end_time}; result: {setting["results"]["baseline"]} --> {setting["results"]["best_acc"]}.')

