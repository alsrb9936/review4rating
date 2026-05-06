import torch
import torch.nn as nn
import torch.nn.functional as F

class AbstractRec(nn.Module):
    def __init__(self):
        super().__init__()

    def init_weights(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError
    
    def cal_loss(self):
        raise NotImplementedError

    def predict_scores(self):
        raise NotImplementedError

def bpr_loss(user_emb, pos_emb, neg_emb):
    pos_scores = (user_emb * pos_emb).sum(dim=1)
    neg_scores = (user_emb * neg_emb).sum(dim=1)
    return F.softplus(neg_scores - pos_scores).mean()

def l2_reg_loss(user_ego, pos_ego, neg_ego):
    batch_size = user_ego.size(0)
    reg = (
        user_ego.pow(2).sum()
        + pos_ego.pow(2).sum()
        + neg_ego.pow(2).sum()
    ) / (2.0 * batch_size)
    return reg

def cal_infonce_loss(embeds1, embeds2, all_embeds2, temp=1.0):
    normed_embeds1 = embeds1 / torch.sqrt(1e-8 + embeds1.square().sum(-1, keepdim=True))
    normed_embeds2 = embeds2 / torch.sqrt(1e-8 + embeds2.square().sum(-1, keepdim=True))
    normed_all_embeds2 = all_embeds2 / torch.sqrt(1e-8 + all_embeds2.square().sum(-1, keepdim=True))
    nume_term = -(normed_embeds1 * normed_embeds2 / temp).sum(-1)
    deno_term = torch.log(torch.sum(torch.exp(normed_embeds1 @ normed_all_embeds2.T / temp), dim=-1))
    cl_loss = (nume_term + deno_term).sum()
    return cl_loss
