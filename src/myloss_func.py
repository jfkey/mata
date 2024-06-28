import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from src.utils import *
from torch import Tensor
import numpy as np
from multiprocessing import Pool
import scipy.optimize as opt


# binary cross entropy， type = 1
class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()

    def forward(self, pred_mat: Tensor, gt_mat: Tensor, src_ns: Tensor, tgt_ns: Tensor, ) -> Tensor:
        if len(pred_mat.shape) == 2:
            pred_mat = pred_mat.unsqueeze(0)
            gt_mat = gt_mat.unsqueeze(0)

        batch_num = pred_mat.shape[0]

        pred_mat = pred_mat.to(dtype=torch.float32)
        gt_mat = gt_mat.to(dtype=torch.float32)

        try:
            assert torch.all((pred_mat >= 0) * (pred_mat <= 1))
            assert torch.all((pred_mat >= 0) * (pred_mat <= 1))
        except AssertionError as err:
            print(pred_mat)
            raise err

        loss = torch.tensor(0.).to(pred_mat.device)
        n_sum = torch.zeros_like(loss)
        for b in range(batch_num):
            batch_slice = [b, slice(src_ns[b]), slice(tgt_ns[b])]
            loss += F.binary_cross_entropy(pred_mat[batch_slice], gt_mat[batch_slice], reduction='sum')
            n_sum += src_ns[b].to(n_sum.dtype).to(pred_mat.device)
        return loss / n_sum



class WeightMultiMatchingLoss(nn.Module):
    def __init__(self):
        super(WeightMultiMatchingLoss, self).__init__()

    def forward(self, pred_mat: Tensor, gt_mat: Tensor, src_ns: Tensor, tgt_ns: Tensor, ) -> Tensor:
        if len(pred_mat.shape) == 2:
            pred_mat = pred_mat.unsqueeze(0)
            gt_mat = gt_mat.unsqueeze(0)

        batch_num = pred_mat.shape[0]

        pred_mat = pred_mat.to(dtype=torch.float32)
        gt_mat = gt_mat.to(dtype=torch.float32)

        try:
            assert torch.all((pred_mat >= 0) * (pred_mat <= 1))
            assert torch.all((pred_mat >= 0) * (pred_mat <= 1))
        except AssertionError as err:
            print(pred_mat)
            raise err

        items = torch.sum(torch.mul(pred_mat, gt_mat), dim=2)
        weight = torch.softmax(items, dim=-1).detach()
        one = torch.ones(items.shape[1]).to(pred_mat.device)
        loss = torch.tensor(0.).to(pred_mat.device)
        n_sum = torch.zeros_like(loss)
        for b in range(batch_num):
            loss += F.binary_cross_entropy(items[b, :], one, weight[b, :], reduction='sum')
            n_sum += src_ns[b].to(n_sum.dtype).to(pred_mat.device)
        return loss / n_sum

class MultiMatchingLoss(nn.Module):
    def __init__(self):
        super(MultiMatchingLoss, self).__init__()

    def forward(self, pred_mat: Tensor, gt_mat: Tensor, src_ns: Tensor, tgt_ns: Tensor, ) -> Tensor:
        if len(pred_mat.shape) == 2:
            pred_mat = pred_mat.unsqueeze(0)
            gt_mat = gt_mat.unsqueeze(0)

        batch_num = pred_mat.shape[0]

        pred_mat = pred_mat.to(dtype=torch.float32)
        gt_mat = gt_mat.to(dtype=torch.float32)

        try:
            assert torch.all((pred_mat >= 0) * (pred_mat <= 1))
            assert torch.all((pred_mat >= 0) * (pred_mat <= 1))
        except AssertionError as err:
            print(pred_mat)
            raise err
        items = torch.sum(torch.mul(pred_mat, gt_mat), dim=2)
        items = torch.where(items > 0, items, torch.tensor(1.0).to(pred_mat.device))
        one = torch.ones(items.shape[1]).to(pred_mat.device)
        loss = torch.tensor(0.).to(pred_mat.device)
        n_sum = torch.zeros_like(loss)
        for b in range(batch_num):
            loss += F.binary_cross_entropy(items[b, :], one, reduction='sum')
            n_sum += src_ns[b].to(n_sum.dtype).to(pred_mat.device)
        return loss / n_sum

class GEDLoss(nn.Module):
    def __init__(self, app_astar, nx_graphs, batch_graphs):
        super(GEDLoss,  self).__init__()
        self.app_astar = app_astar
        self.nx_graphs = nx_graphs
        self.batch_graphs = batch_graphs

    def forward(self, pred_mat: Tensor, gt_mat: Tensor, src_ns: Tensor, tgt_ns: Tensor, ) -> Tensor:
        if len(pred_mat.shape) == 2:
            pred_mat = pred_mat.unsqueeze(0)
            gt_mat = gt_mat.unsqueeze(0)

        batch_num = pred_mat.shape[0]

        pred_mat = pred_mat.to(dtype=torch.float32)
        gt_mat = gt_mat.to(dtype=torch.float32)

        try:
            assert torch.all((pred_mat >= 0) * (pred_mat <= 1))
            assert torch.all((pred_mat >= 0) * (pred_mat <= 1))
        except AssertionError as err:
            print(pred_mat)
            raise err
        items = torch.sum(torch.mul(pred_mat, gt_mat), dim=2)
        items = torch.where(items > 0, items, torch.tensor(1.0).to(pred_mat.device))
        one = torch.ones(items.shape[1]).to(pred_mat.device)
        loss = torch.tensor(0.).to(pred_mat.device)
        loss_map_ed = torch.tensor(0.).to(pred_mat.device)
        n_sum = torch.zeros_like(loss)
        np_pred_mat = pred_mat.detach().numpy()
        for b in range(batch_num):
            loss += F.binary_cross_entropy(items[b, :], one, reduction='sum')
            g1_id = self.batch_graphs['g1'][b]['i'].item()
            g2_id = self.batch_graphs['g2'][b]['i'].item()
            rows, cols = linear_sum_assignment(np_pred_mat[b, :src_ns[b], :tgt_ns[b]])
            astar_out = self.app_astar.mapping_ed(CT(self.nx_graphs[g1_id]), CT(self.nx_graphs[g2_id]), int1ArrayToPointer(rows), int1ArrayToPointer(cols))
            loss_map_ed += (2 * astar_out / ((src_ns[b].to(n_sum.dtype).to(pred_mat.device) + src_ns[b].to(n_sum.dtype).to(pred_mat.device) ) ))
            # loss = 2 * loss / (src_ns[b].to(n_sum.dtype).to(pred_mat.device) + src_ns[b].to(n_sum.dtype).to(pred_mat.device) )
        # return  loss_map_ed/batch_num
        return loss /batch_num


class PermutationLoss(nn.Module):
    def __init__(self):
        super(PermutationLoss, self).__init__()

    def forward(self, pred_dsmat: Tensor, node_matching_item, src_ns: Tensor, tgt_ns: Tensor, ) -> Tensor:

        gt_perm = torch.zeros(pred_dsmat.shape)
        for k, v in node_matching_item['map'].items():
            if k != '-1': gt_perm[int(k)][int(v)] = 1

        if len(pred_dsmat.shape) == 2:
            pred_dsmat = pred_dsmat.unsqueeze(0)
            gt_perm = gt_perm.unsqueeze(0)

        batch_num = pred_dsmat.shape[0]

        pred_dsmat = pred_dsmat.to(dtype=torch.float32)

        try:
            assert torch.all((pred_dsmat >= 0) * (pred_dsmat <= 1))
            assert torch.all((gt_perm >= 0) * (gt_perm <= 1))
        except AssertionError as err:
            print(pred_dsmat)
            raise err

        loss = torch.tensor(0.).to(pred_dsmat.device)
        n_sum = torch.zeros_like(loss)
        for b in range(batch_num):
            batch_slice = [b, slice(src_ns[b]), slice(tgt_ns[b])]
            loss += F.binary_cross_entropy(pred_dsmat[batch_slice], gt_perm[batch_slice], reduction='sum')
            n_sum += src_ns[b].to(n_sum.dtype).to(pred_dsmat.device)
        return loss / n_sum



def hungarian(s: Tensor, n1: Tensor=None, n2: Tensor=None, nproc: int=1) -> Tensor:
    r"""
    Solve optimal LAP permutation by hungarian algorithm. The time cost is :math:`O(n^3)`.
    :param s: :math:`(b\times n_1 \times n_2)` input 3d tensor. :math:`b`: batch size
    :param n1: :math:`(b)` number of objects in dim1
    :param n2: :math:`(b)` number of objects in dim2
    :param nproc: number of parallel processes (default: ``nproc=1`` for no parallel)
    :return: :math:`(b\times n_1 \times n_2)` optimal permutation matrix
    .. note::
        We support batched instances with different number of nodes, therefore ``n1`` and ``n2`` are
        required to specify the exact number of objects of each dimension in the batch. If not specified, we assume
        the batched matrices are not padded.
    """
    if len(s.shape) == 2:
        s = s.unsqueeze(0)
        matrix_input = True
    elif len(s.shape) == 3:
        matrix_input = False
    else:
        raise ValueError('input data shape not understood: {}'.format(s.shape))

    device = s.device
    batch_num = s.shape[0]

    perm_mat = s.cpu().detach().numpy() * -1
    if n1 is not None:
        n1 = n1.cpu().numpy()
    else:
        n1 = [None] * batch_num
    if n2 is not None:
        n2 = n2.cpu().numpy()
    else:
        n2 = [None] * batch_num

    if nproc > 1:
        with Pool(processes=nproc) as pool:
            mapresult = pool.starmap_async(_hung_kernel, zip(perm_mat, n1, n2))
            perm_mat = np.stack(mapresult.get())
    else:
        perm_mat = np.stack([_hung_kernel(perm_mat[b], n1[b], n2[b]) for b in range(batch_num)])

    perm_mat = torch.from_numpy(perm_mat).to(device)

    if matrix_input:
        perm_mat.squeeze_(0)

    return perm_mat

def _hung_kernel(s: torch.Tensor, n1=None, n2=None):
    if n1 is None:
        n1 = s.shape[0]
    if n2 is None:
        n2 = s.shape[1]
    row, col = opt.linear_sum_assignment(s[:n1, :n2])
    perm_mat = np.zeros_like(s)
    perm_mat[row, col] = 1
    return perm_mat



if __name__ == '__main__':

    sim_mat = torch.tensor(

    [[3.3329e-01, 9.2224e-31, 2.2443e-10, 2.2443e-10, 2.2443e-10, 2.2443e-10, 2.2443e-10, 3.3329e-01],
     [3.3329e-01, 8.7182e-31, 7.2120e-15, 7.2120e-15, 7.2120e-15, 7.2120e-15, 7.2120e-15, 3.3329e-01],
     [3.3328e-01, 9.2956e-21, 3.0264e-06, 3.0264e-06, 3.0264e-06, 3.0264e-06, 3.0264e-06, 3.3328e-01],
     [1.4560e-04, 3.5217e-10, 3.1458e-01, 3.1458e-01, 3.1458e-01, 3.1458e-01, 3.1458e-01, 1.4560e-04],
     [3.6051e-07, 5.0233e-09, 3.1472e-01, 3.1472e-01, 3.1472e-01, 3.1472e-01, 3.1472e-01, 3.6051e-07],
     [1.5101e-15, 1.8247e-03, 3.1367e-01, 3.1367e-01, 3.1367e-01, 3.1367e-01, 3.1367e-01, 1.5101e-15],
     [1.6941e-24, 5.4876e-01, 3.1059e-05, 3.1059e-05, 3.1059e-05, 3.1059e-05, 3.1059e-05, 1.6941e-24],
     [1.3628e-27, 4.4941e-01, 5.7001e-02, 5.7001e-02, 5.7001e-02, 5.7001e-02, 5.7001e-02, 1.3628e-27]])

    t = torch.zeros(1).fill_(sim_mat[0][0])

    loss = F.binary_cross_entropy(t, torch.zeros(1), reduction='sum')
    print(loss)


    m = np.array([[2,1], [0,1], [0,2]])
    m = m.transpose()
    z = np.random.rand(3,3)
    z[m[0], m[1]] = 0
    print(z)

