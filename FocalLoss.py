import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.average = average

        if alpha is None:
            self.alpha = torch.ones(class_num, 1)
        else:
            if isinstance(alpha, torch.Tensor):
                self.alpha = alpha
            else:
                self.alpha = torch.Tensor(alpha)

    def forward(self, inputs, targets, device):
        N, C = inputs.size()
        P = F.softmax(inputs, dim=1)

        class_mask = inputs.new_zeros(N, C)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids, 1.)
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.to(device)
        alpha = self.alpha[ids.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()

        batch_loss = -alpha * torch.pow((1 - probs),self.gamma) * log_p

        if self.average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


if __name__ == "__main__":
    FL = FocalLoss(class_num=5, gamma=2)
    CE = nn.CrossEntropyLoss()
    N = 4
    C = 5
    inputs = torch.rand(N, C)
    targets = torch.LongTensor(N).random_(C)
    inputs_fl = torch.tensor(inputs, requires_grad=True)
    targets_fl = torch.tensor(targets)

    inputs_ce = torch.tensor(inputs, requires_grad=True)
    targets_ce = torch.tensor(targets)
    print('----inputs----')
    print(inputs)
    print('---target-----')
    print(targets)

    fl_loss = FL(inputs_fl, targets_fl, device = torch.device('cuda:0'))
    ce_loss = CE(inputs_ce, targets_ce)
    print('ce = {}, fl ={}'.format(ce_loss.item(), fl_loss.item()))
    fl_loss.backward()
    ce_loss.backward()
    #print(inputs_fl.grad.data)
    print(inputs_ce.grad.data)