# -*- coding: utf-8 -*-
import torch


class TensorBuffer:
    """
    Packs multiple tensors into one flat buffer for efficient
    intra-worker communication.
    """

    def __init__(self, tensors, device=torch.device('cpu'), clone=True):
        indices = [0]
        end = 0
        self.buffer = []
        self._tensors_sizes = []
        for i, tensor in enumerate(tensors):
            end += tensor.nelement()
            indices.append(end)
            buffer = tensor.data.flatten().to(device)
            if clone:
                buffer = buffer.clone()
            self.buffer.append(buffer.clone())
            self._tensors_sizes.append(tensor.size())

        self._start_idx = indices[:-1]
        self._end_idx = indices[1:]
        self._tensors_len = i + 1

        self.buffer = torch.concat(self.buffer)  # copies

    def __getitem__(self, index):
        return self.buffer[self._start_idx[index]: self._end_idx[index]].view(
            self._tensors_sizes[index]
        )

    def __len__(self):
        return self._tensors_len

    def is_cuda(self):
        return self.buffer.is_cuda

    def nelement(self):
        return self.buffer.nelement()

    def unpack(self, tensors):
        for tensor, entry in zip(tensors, self):
            tensor.data = entry.clone().to(tensor.device)

    def unpack_grad(self, tensors):
        for tensor, entry in zip(tensors, self):
            tensor.grad = entry.clone()

    def __iadd__(self, other):
        assert isinstance(other, TensorBuffer)
        assert other._start_idx == self._start_idx
        assert other._end_idx == self._end_idx
        assert other._tensors_len == self._tensors_len
        assert other._tensors_sizes == self._tensors_sizes
        self.buffer += other.buffer
        return self
