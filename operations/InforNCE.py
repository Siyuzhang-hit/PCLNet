import torch
from torch import nn


class InforNCE(nn.Module):
    def __init__(self):
        super(InforNCE, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):

        labels = torch.zeros(x.shape[0], dtype=torch.long).cuda()
        loss = self.criterion(x, labels)
        return loss, labels


class MemoryBank(nn.Module):
    def __init__(self, inputSize, outputSize, K, T=0.4, use_softmax=False):
        super(MemoryBank, self).__init__()
        self.outputSize = outputSize
        self.inputSize = inputSize
        self.queueSize = K
        self.T = T
        self.index = 0
        self.use_softmax = use_softmax

        # create the queue
        self.register_buffer("queue", torch.randn(inputSize, self.queueSize))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queueSize % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queueSize  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, q, k):

        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        k = k.detach()

        # pos logit
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # neg logit
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        out = torch.cat((l_pos, l_neg), dim=1)
        # apply temperature
        out /= self.T

        # # update memory
        self._dequeue_and_enqueue(k)
        return out


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
