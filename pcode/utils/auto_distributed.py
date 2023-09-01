# -*- coding: UTF-8 -*-
"""
@author:aaaal
@file:auto_distributed.py
@time:2022/05/06
"""
import types
import warnings
from enum import Enum
from typing import List

import torch
import torch.distributed as dist

type_list = "torch.float32,torch.float,torch.float64,torch.double,torch.float16,torch.bfloat16,torch.half,torch.uint8,torch.int8,torch.int16,torch.short,torch.int32,torch.int,torch.int64,torch.long,torch.complex32,torch.complex64,torch.cfloat,torch.complex128,torch.cdouble,torch.quint8,torch.qint8,torch.qint32,torch.bool,torch.quint4x2,torch.quint2x4"
type_enum = Enum('dtype', type_list)
type_list = type_list.split(',')


def check_device(func, backend):
    def wrapped_function(*args, **kwargs):
        if 'tensor' in kwargs:
            device = kwargs['tensor'].device.type
        elif isinstance(args[0], torch.Tensor):
            device = args[0].device.type
        assert (backend == 'gloo' and device == 'cpu') or (backend == 'nccl' and device == 'cuda') or backend == 'mpi'
        result = func(*args, **kwargs)
        return result

    return wrapped_function


def auto_device(func, backend):
    def wrapped_function(*args, **kwargs):
        device = get_default_device(backend)
        _args, _kwargs = move_paras(device, *args, **kwargs)
        result = func(*_args, **_kwargs)
        move_back_params(device, args, _args, kwargs, _kwargs)
        return result

    def move_paras(device, *args, **kwargs):
        _args = [move_tensor(arg, device) for arg in args]
        _kwargs = {k: move_tensor(v, device) for k, v in kwargs.items()}
        return _args, _kwargs

    def move_tensor(tensor, device):
        if hasattr(tensor, "to"):
            return tensor.to(device)
        if isinstance(tensor, list):
            return [t.to(device) if hasattr(t, "to") else t for t in tensor]
        return tensor

    def move_back_params(device, args, _args, kwargs, _kwargs):
        for t, _t in zip(args, _args):
            move_back_tensor(t, _t, device)
        for t, _t in zip(kwargs.values(), _kwargs.values()):
            move_back_tensor(t, _t, device)

    def move_back_tensor(tensor, _tensor, device):
        if hasattr(_tensor, "to"):
            if hasattr(tensor, "device") and hasattr(_tensor, "device") and tensor.device != _tensor.device:
                warnings.warn(f"Tensor device mismatch,except {_tensor.device}, receive {tensor.device}")
            tensor.data = _tensor.to(device).data
        if isinstance(tensor, list):
            for i, (t, _t) in enumerate(zip(tensor, _tensor)):
                if hasattr(_t, "to"):
                    if hasattr(t, "device") and hasattr(_t, "device") and t.device != _t.device:
                        warnings.warn(
                            f"Tensor device mismatch,except {_t.device}, receive {None if t is None else t.device}")
                    tensor[i] = _tensor[i].to(device)
                tensor[i] = _tensor[i]

    def get_default_device(backend):
        device = None
        if backend == 'gloo':
            device = torch.device("cpu")
        elif backend == 'nccl':
            device = torch.device("cuda")
        return device

    return wrapped_function


for k, v in vars(dist).items():
    if isinstance(v, types.FunctionType):
        vars(dist)[k] = auto_device(v, "gloo")


def check_and_send(tensor: torch.Tensor, dst, group=None, tag=0):
    device = tensor.device
    info = torch.zeros(2, device=device)
    info[0] = tensor.numel()
    info[1] = type_enum[tensor.dtype.__str__()].value
    dist.send(info, dst, group, tag)
    dist.send(tensor, dst, group, tag)


def check_and_receive(tensor: torch.Tensor, src, group=None, tag=0):
    device = tensor.device
    info = torch.zeros(2, device=device)
    dist.recv(info, src, group, tag)
    length = tensor.numel()
    if info[0] != length:
        warnings.warn(f"Tensor length mismatch. Except {length}, but received {int(info[0])}.")
    if info[1] != type_enum[tensor.dtype.__str__()].value:
        warnings.warn(f"Tensor dtype mismatch. Except {tensor.dtype}, but received {type_enum(int(info[1])).value}.")
    dist.recv(tensor, src, group, tag)


def send_list(tensor_list: List[torch.Tensor], dst, group=None, tag=0, device=None):
    if dist.get_backend() == "gloo":
        device = torch.device("cpu")
    elif dist.get_backend() == "nccl":
        device = torch.device("cuda")

    info = []
    flatten_tensor = []
    for tensor in tensor_list:
        info += [0] if tensor.device == torch.device("cpu") else [1]
        tensor = tensor.to(device)
        info += [tensor.numel()]
        info += [tensor.dim()]
        info += list(tensor.shape)
        info += [type_enum[tensor.dtype.__str__()].value]
        flatten_tensor.append(torch.flatten(tensor))
    flatten_tensor = [torch.tensor(info, device=device)] + flatten_tensor
    flatten_tensor = torch.concat(flatten_tensor)
    dist.send(torch.IntTensor([len(info), len(flatten_tensor), type_enum[flatten_tensor.dtype.__str__()].value],
                              device=device), dst, group, tag)
    dist.send(flatten_tensor, dst, group, tag)


def recv_list(src, group=None, tag=0, trans_device=None, restore_device=False):
    if dist.get_backend() == "gloo":
        trans_device = torch.device("cpu")
    elif dist.get_backend() == "nccl":
        trans_device = torch.device("cuda")

    lens = torch.IntTensor([0, 0, 0], device=trans_device)
    dist.recv(lens, src, group, tag)
    flatten_tensor = torch.zeros(lens[1], dtype=eval(type_list[lens[2] - 1]), device=trans_device)
    dist.recv(flatten_tensor, src, group, tag)
    tensor_list = []
    start = int(lens[0])
    idx = 0
    while (idx < lens[0]):
        tensor_device = trans_device if restore_device is False else torch.device("cpu") if int(
            flatten_tensor[idx]) == 0 else torch.device("cuda")
        idx += 1
        numel = int(flatten_tensor[idx])
        idx += 1
        dim = int(flatten_tensor[idx])
        idx += 1
        shape = tuple(flatten_tensor[idx:idx + dim].numpy().astype('int'))
        idx += dim
        dtype = eval(type_list[int(flatten_tensor[idx]) - 1])
        idx += 1
        tensor_list.append(flatten_tensor[start:start + numel].to(dtype).to(tensor_device).reshape(shape))
        start += numel
    return tensor_list

def gather_objects(objects_list: object = None, dst: int = 0):
    output_list = [None] * dist.get_world_size()
    dist.gather_object(
        objects_list,
        output_list if dist.get_rank() == dst else None,
        dst=0
    )
    return output_list

def scatter_objects(scatter_list: List[object] = None, src: int = 0):
    output_list = [None]
    if dist.get_rank() == src:
        object_list = [None] + scatter_list
    else:
        object_list = [None] * dist.get_world_size()
    dist.scatter_object_list(output_list, object_list, src)
    return output_list

import os
import torch
import torch.multiprocessing as mp


def run(rank, size):
    list = [torch.tensor([1, 2, 3, 4]), torch.tensor([3, 4, 2, 1])]
    if rank == 0:
        dist.send(list[0], dst=1)
    else:
        # Receive tensor from process 0
        dist.recv(list[1], src=0)

    print('Rank ', rank, ' has data', list[rank])


def init_process(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    torch.cuda.set_device(rank)
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
