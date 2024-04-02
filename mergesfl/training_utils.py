import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def merge_and_dispatch(args, global_model, global_optim, selected_ids, bsz_list, worker_list, device=torch.device('cpu')):
    """
    merge collected batches for server-side training

    Parameters:
        global_model: server-side model  
        global_optim: server-side optimizer
        bsz_list: list of batchsizes for all workers
    
    Return:
        average training loss
        comm cost
    """
    comm_cost = 0
    # merge smashed data
    m_data = []
    m_target = []
    for worker_id, worker in zip(selected_ids, worker_list):
        data_i, target_i = worker.config.neighbor_paras[0].to(device), worker.config.neighbor_paras[1].to(device)
        m_data.append(data_i)
        m_target.append(target_i)
        comm_cost += data_i.nelement() * 4 + target_i.nelement() * 2

    m_data = torch.cat(m_data, dim=0)
    m_target = torch.cat(m_target, dim=0)
    
    m_data.requires_grad_()

    # server side fp
    outputs = global_model(m_data)
    loss = F.cross_entropy(outputs, m_target.long())

    # server side bp
    global_optim.zero_grad()
    loss.backward()
    global_optim.step()
    
    # gradient dispatch
    sum_bsz = sum([bsz_list[i] for i in selected_ids])
    bsz_s = 0
    for worker_idx, worker in zip(selected_ids, worker_list):
        worker.config.grad_in = (m_data.grad[bsz_s: bsz_s + bsz_list[worker_idx]] * sum_bsz / bsz_list[worker_idx])
        bsz_s += bsz_list[worker_idx]
        comm_cost += worker.config.grad_in.nelement() * 4
          
    return loss, comm_cost


def control(args, active_num, worker_list):

    selected_ids = np.random.choice(range(len(worker_list)), size=active_num, replace=False)
    # bsz_list = np.random.randint(32, 64 + 1, len(worker_list))
    bsz_list = np.ones(len(worker_list), dtype=int) * args.batch_size
    
    return selected_ids, bsz_list


def test(model_fe, model_p, data_loader, device=torch.device("cpu"), two_split=False):
    model_p.eval()
    
    if two_split:
        model_fe[0].eval()
        model_fe[1].eval()
    else:
        model_fe.eval()
    
    data_loader = data_loader.loader
    test_loss = 0.0
    test_accuracy = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:

            data, target = data.to(device), target.to(device)
            
            if two_split:
                output1 = model_fe[0](data)
            else:
                output1 = model_fe(data)
            
            output = model_p(output1)

            if two_split:
                output = model_fe[1](output)
            
            # sum up batch loss
            loss_func = nn.CrossEntropyLoss(reduction='sum') 
            test_loss += loss_func(output, target.long()).item()

            pred = output.argmax(1, keepdim=True)
            batch_correct = pred.eq(target.view_as(pred)).sum().item()

            correct += batch_correct
            
    test_loss /= len(data_loader.dataset)
    test_accuracy = np.float(1.0 * correct / len(data_loader.dataset))
    return test_loss, test_accuracy

