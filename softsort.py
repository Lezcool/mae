import torch
from torch import Tensor


class SoftSort(torch.nn.Module):
    def __init__(self, tau=1.0, hard=False, pow=1.0):
        super(SoftSort, self).__init__()
        self.hard = hard
        self.tau = tau
        self.pow = pow

    def forward(self, scores: Tensor):
        """
        scores: elements to be sorted. Typical shape: batch_size x n
        """
        scores = scores.unsqueeze(-1)
        sorted = scores.sort(descending=True, dim=1)[0]
        pairwise_diff = (scores.transpose(1, 2) - sorted).abs().pow(self.pow).neg() / self.tau
        P_hat = pairwise_diff.softmax(-1)

        if self.hard: #这段代码的作用就是在P_hat上查找最大值，并将其位置置为1，其余位置置为0.
            P = torch.zeros_like(P_hat, device=P_hat.device)
            P.scatter_(-1, P_hat.topk(1, -1)[1], value=1) 
            #使用scatter_函数将P_hat.topk(1, -1)[1]位置的元素置为1。这里的topk函数是Pytorch中用来查找最大值的函数，topk(1, -1)表示在P_hat最后一维中查找最大值。
            P_hat = (P - P_hat).detach() + P_hat
        return P_hat
