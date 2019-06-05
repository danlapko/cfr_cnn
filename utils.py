import numpy as np
import torch
from torch.nn import PairwiseDistance

cpu = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

l2_dist = PairwiseDistance(2)


def rank1(embeddings_anc, embeddings_pos, clf):
    n = len(embeddings_anc)
    n_good = 0

    A = conjagate_matrix(embeddings_anc, embeddings_pos, clf)
    for i, anc_base_dists in enumerate(A):
        j = np.argmin(anc_base_dists)
        if i == j:
            n_good += 1
        # print(i,j)
    return n_good / n


def roc_curve(embeddings_anc, embeddings_pos, clf):
    A = conjagate_matrix(embeddings_anc, embeddings_pos, clf)
    A = (A - A.min()) / (A.max() - A.min())
    trshs = []
    tprs = [0]
    fprs = [0]
    for th in np.sort(np.unique(A.ravel())):
        tpr, fpr = tpr_fpr(A, th)
        trshs.append(th)
        tprs.append(tpr)
        fprs.append(fpr)
    fprs.append(1)
    tprs.append(1)

    return trshs, fprs, tprs


def tpr_fpr(Conj, th):
    n, _ = Conj.shape
    B = Conj < th
    tpr = np.trace(B)

    fpr = np.sum(B) - tpr

    tpr = tpr / n
    fpr = fpr / n ** 2

    return tpr, fpr


def conjagate_matrix(embeddings_anc, embeddings_pos, clf):
    n = embeddings_anc.shape[0]
    A = np.zeros((n, n), dtype=np.float)
    for i, pos in enumerate(embeddings_pos):
        A[:, i] = clf(torch.cat((embeddings_anc,
                                 pos.unsqueeze(0).expand_as(embeddings_anc)), dim=-1)).detach().to(cpu).view(-1)
    return A


mask = torch.full((1, 3, 60, 48), 0.2).to(device)
mask[:, :, 10:30] = 1.0
mask[:, :, 10:, 8:40] = 1.0


def weighted_mse_loss(input, target):
    mas = mask.expand_as(target)
    return torch.mean(mas * (input - target) ** 2)
