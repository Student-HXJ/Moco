import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.ResNet import ResNet50


class MoCo(nn.Module):

    def __init__(self, dim=128):
        super(MoCo, self).__init__()
        self.K = 65536
        self.encoder_q = ResNet50(num_classes=dim)
        self.encoder_k = ResNet50(num_classes=dim)

        dim_mlp = self.encoder_q.fc.weight.shape[1]
        self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
        self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, self.K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, k):
        # gather k before updating queue
        batch_size = k.shape[0]
        ptr = int(self.queue_ptr)
        # replace the k at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = k.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle(self, x):
        batch_size_all = x.shape[0]
        idx_shuffle = torch.randperm(batch_size_all).cuda()
        idx_unshuffle = torch.argsort(idx_shuffle)
        return x[idx_shuffle], idx_unshuffle

    def forward(self, im_q, im_k):
        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = F.normalize(q, dim=1)
        # compute key features
        with torch.no_grad():  # no gradient to keys
            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data = param_k.data * 0.999 + param_q.data * (1. - 0.999)
            # im_k, idunshuffle = self._batch_shuffle(im_k)
            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)
            # k = k[idunshuffle]
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= 0.07
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels
