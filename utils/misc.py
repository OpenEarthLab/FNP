import torch.distributed as dist
import torch
import os
import numpy as np
import random
from typing import Any
import re


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def get_ip(ip_list):
    if "," in ip_list:
        ip_list = ip_list.split(',')[0]
    if "[" in ip_list:
        ipbefore_4, ip4 = ip_list.split('[')
        ip4 = re.findall(r"\d+", ip4)[0]
        ip1, ip2, ip3 = ipbefore_4.split('-')[-4:-1]
    else:
        ip1,ip2,ip3,ip4 = ip_list.split('-')[-4:]
    ip_addr = "tcp://" + ".".join([ip1, ip2, ip3, ip4]) + ":"
    return ip_addr
    


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.local_rank = int(os.environ['SLURM_LOCALID'])
        args.world_size = int(os.environ['SLURM_NTASKS'])
        ip_addr = get_ip(os.environ['SLURM_STEP_NODELIST'])
        port = int(os.environ['SLURM_SRUN_COMM_PORT'])
        # args.init_method = ip_addr + str(port)
        args.init_method = ip_addr + args.init_method.split(":")[-1]
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.local_rank)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}, local_rank {}): {}'.format(
        args.rank, args.local_rank, args.init_method), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.init_method,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def DistributedParallel_Model(model, gpu_num, find_unused_parameters=False):
    if is_dist_avail_and_initialized():
        device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        if device == torch.device('cpu'):
            raise EnvironmentError('No GPUs, cannot initialize multigpu training.')
        model.to(device)
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # model, optimizer = amp.initialize(model, optimizer, opt_level="O0")
        ddp_sub_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu_num], find_unused_parameters=find_unused_parameters)
        model = ddp_sub_model
        
        # model.to(device)
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu_num])
        # model_without_ddp = model.module
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # model, optimizer = amp.initialize(model, optimizer, opt_level="O0")
        # for key in model.model:
        #     model.model[key].to(device)

    return model


class Dict(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]
    # __setattr__ = dict.__setitem__
    # __getattr__ = dict.__getitem__

def dictToObj(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    d = Dict()
    for k, v in dictObj.items():
        d[k] = dictToObj(v)
    return d


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True


def named_params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())

def check_ddp_consistency(module, ignore_regex=None):
    assert isinstance(module, torch.nn.Module)
    for name, tensor in named_params_and_buffers(module):
        fullname = type(module).__name__ + '.' + name
        if ignore_regex is not None and re.fullmatch(ignore_regex, fullname):
            continue
        tensor = tensor.detach()
        # if tensor.is_floating_point():
        #     tensor = nan_to_num(tensor)
        other = tensor.clone()
        torch.distributed.broadcast(tensor=other, src=0)
        # print(fullname, tensor.sum(), other.sum())
        assert (tensor == other).all(), fullname
