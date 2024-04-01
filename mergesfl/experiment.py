from fileinput import filename
import os
import argparse
import time
import numpy as np
import torch
import copy
import torch.nn.functional as F
import datasets, models
import torch.optim as optim
import logging
from training_utils import *
from torch.utils.tensorboard import SummaryWriter


#init parameters
parser = argparse.ArgumentParser(description='Distributed Client')
parser.add_argument('--dataset_type', type=str, default='CIFAR10')
parser.add_argument('--model_type', type=str, default='AlexNet')
parser.add_argument('--worker_num', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--data_pattern', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--decay_rate', type=float, default=0.993)
parser.add_argument('--min_lr', type=float, default=0.005)
parser.add_argument('--epoch', type=int, default=250)
parser.add_argument('--momentum', type=float, default=-1)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--data_path', type=str, default='./data')
parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
parser.add_argument('--expname', type=str, default='MergeSFL')


args = parser.parse_args()
device = torch.device(args.device)

RESULT_PATH = os.getcwd() + '/server/'

recorder: SummaryWriter = SummaryWriter(os.path.join('runs', args.expname))

if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH, exist_ok=True)

def non_iid_partition(ratio, train_class_num, worker_num):
    partition_sizes = np.ones((train_class_num, worker_num)) * ((1 - ratio) / (worker_num-1))

    for i in range(train_class_num):
        partition_sizes[i][i%worker_num]=ratio

    return partition_sizes

def dirichlet_partition(dataset_type: str, alpha: float, worker_num: int, nclasses: int):
    partition_sizes = []
    filepath = './data_partition/%s-part_dir%.1f.npy' % (dataset_type, alpha)
    if os.path.exists(filepath):
        partition_sizes = np.load(filepath)
    else:
        for _ in range(nclasses):
            partition_sizes.append(np.random.dirichlet([alpha] * worker_num))
        partition_sizes = np.array(partition_sizes)
        # np.save(filepath, partition_sizes)

    return partition_sizes


def partition_data(dataset_type, data_pattern, worker_num=10):
    train_dataset, test_dataset = datasets.load_datasets(dataset_type)
    labels = None
    if dataset_type == "CIFAR10" or dataset_type == "FashionMNIST":
        train_class_num = 10

    if data_pattern == 0:
        partition_sizes = np.ones((train_class_num, worker_num)) * (1.0 / worker_num)
    elif data_pattern == 1:
        non_iid_ratio = 0.2
        partition_sizes = non_iid_partition(non_iid_ratio, train_class_num, worker_num)
    elif data_pattern == 2:
        non_iid_ratio = 0.4
        partition_sizes = non_iid_partition(non_iid_ratio, train_class_num, worker_num)
    elif data_pattern == 3:
        non_iid_ratio = 0.6
        partition_sizes = non_iid_partition(non_iid_ratio, train_class_num, worker_num)
    elif data_pattern == 4:
        non_iid_ratio = 0.8
        partition_sizes = non_iid_partition(non_iid_ratio, train_class_num, worker_num)

    elif data_pattern == 5:  # dir-1.0
        print('Dirichlet partition 1.0')
        partition_sizes = dirichlet_partition(dataset_type, 1.0, worker_num, train_class_num)

    elif data_pattern == 6:  # dir-0.5
        print('Dirichlet partition 0.5')
        partition_sizes = dirichlet_partition(dataset_type, 0.5, worker_num, train_class_num)

    elif data_pattern == 7:  # dir-0.1
        print('Dirichlet partition 0.1')
        partition_sizes = dirichlet_partition(dataset_type, 0.1, worker_num, train_class_num)

    elif data_pattern == 8:  # dir-0.1
        print('Dirichlet partition 0.05')
        partition_sizes = dirichlet_partition(dataset_type, 0.01, worker_num, train_class_num)
    print(partition_sizes)
    train_data_partition = datasets.LabelwisePartitioner(train_dataset, partition_sizes=partition_sizes, class_num=train_class_num, labels=labels)
    return train_dataset, test_dataset, train_data_partition, labels


def main():
    worker_num = args.worker_num

    print(args.__dict__)
    

    client_global_model, server_global_model = models.create_model_instance_SL(args.dataset_type, args.model_type, 1)
    nets, _ = models.create_model_instance_SL(args.dataset_type, args.model_type, worker_num)

    #client_global_model.to(device)
    #server_global_model.to(device)
    client_global_model = client_global_model[0]
    global_model_par = client_global_model.state_dict()
    for net_id, net in nets.items():
        net.load_state_dict(global_model_par)

    # Create model instance
    train_dataset, test_dataset, train_data_partition, labels = partition_data(args.dataset_type, args.data_pattern, worker_num)

    if labels:
        test_loader = datasets.create_dataloaders(test_dataset, batch_size=64, shuffle=False, collate_fn=lambda x: datasets.collate_fn(x, labels))

    else:
        test_loader = datasets.create_dataloaders(test_dataset, batch_size=64, shuffle=False)

    # clients data loaders
    bsz_list = np.ones(worker_num, dtype=int) * args.batch_size
    client_train_loader = []
    for worker_idx in range(worker_num):
        if labels:
            client_train_loader.append(datasets.create_dataloaders(train_dataset, batch_size=int(bsz_list[worker_idx]), selected_idxs=train_data_partition.use(worker_idx), pin_memory=False, drop_last=True, collate_fn=lambda x: datasets.collate_fn(x, labels)))
        else:
            client_train_loader.append(datasets.create_dataloaders(train_dataset, batch_size=int(bsz_list[worker_idx]), selected_idxs=train_data_partition.use(worker_idx), pin_memory=False, drop_last=True))
        #break
    
    epoch_lr = args.lr
    print('start training')
    for epoch_idx in range(1, 1+args.epoch):
        print(f'In epoch {epoch_idx}')
        start_time = time.time()
        # learning rate
        if epoch_idx > 1:
            epoch_lr = max((args.decay_rate * epoch_lr, args.min_lr))

        # define optimizers    
        if args.momentum < 0:
            global_optim = optim.SGD(server_global_model.parameters(), lr=epoch_lr, weight_decay=args.weight_decay)
        else:
            global_optim = optim.SGD(server_global_model.parameters(), lr=epoch_lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay)

        clients_optimizers = []
        for worker_idx in range(worker_num):
            if args.momentum < 0:
                clients_optimizers.append(optim.SGD(nets[worker_idx].parameters(), lr=epoch_lr, weight_decay=args.weight_decay))
            else:
                clients_optimizers.append(optim.SGD(nets[worker_idx].parameters(), momentum=args.momentum, nesterov=True, lr=epoch_lr, weight_decay=args.weight_decay))

        server_global_model.train()
        local_steps = 42
        server_global_model.to(device)
        
        # client side
        for iter_idx in range(local_steps):
            clients_smash_data = []
            client_send_data = []
            client_send_targets = []
            for worker_idx in range(worker_num):
                inputs, targets = next(client_train_loader[worker_idx])

                inputs, targets = inputs.to(device), targets.to(device)

                nets[worker_idx].to(device)

                clients_smash_data.append(nets[worker_idx](inputs))

                send_smash = clients_smash_data[-1].detach()
                client_send_data.append(send_smash)
                client_send_targets.append(targets)
                #break
        
            m_data = torch.cat(client_send_data, dim=0)
            m_target = torch.cat(client_send_targets, dim=0)
            
            m_data.requires_grad_()

            # server side fp
            outputs = server_global_model(m_data)
            loss = F.cross_entropy(outputs, m_target.long())

            # server side bp
            global_optim.zero_grad()
            loss.backward()
            global_optim.step()
            
            # gradient dispatch
            sum_bsz = sum([bsz_list[i] for i in range(worker_num)])
            bsz_s = 0
            for worker_idx in range(worker_num):
                clients_grad = m_data.grad[bsz_s: bsz_s + bsz_list[worker_idx]] * sum_bsz / bsz_list[worker_idx]
                bsz_s += bsz_list[worker_idx]

                clients_optimizers[worker_idx].zero_grad()
                clients_smash_data[worker_idx].backward(clients_grad.to(device))
                clients_optimizers[worker_idx].step()
                #break
        with torch.no_grad():
            for worker_idx in range(worker_num):
                nets[worker_idx].to('cpu')
                net_para = nets[worker_idx].cpu().state_dict()
                if worker_idx == 0:
                    for key in net_para:
                        global_model_par[key] = net_para[key]/worker_num
                else:
                    for key in net_para:
                        global_model_par[key] += net_para[key]/worker_num
            client_global_model.load_state_dict(global_model_par)
       
        server_global_model.to('cpu')
        test_loss, acc = test(client_global_model, server_global_model, test_loader)
        print("Epoch: {}, accuracy: {}, test_loss: {}".format(epoch_idx, acc, test_loss))

        global_model_par = client_global_model.state_dict()
        for net_id, net in nets.items():
            net.load_state_dict(global_model_par)



    
if __name__ == '__main__':
    main()