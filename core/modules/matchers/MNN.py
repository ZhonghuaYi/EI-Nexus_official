"""
Nearest neighbor matcher for normalized descriptors.
Optionally apply the mutual check and threshold the distance or ratio.
"""
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def find_nn(sim, ratio_thresh, distance_thresh):
    k = 2 if ratio_thresh else 1
    sim_nn, ind_nn = sim.topk(k, dim=-1, largest=True)
    dist_nn = 2 * (1 - sim_nn)
    mask = torch.ones(ind_nn.shape[:-1], dtype=torch.bool, device=sim.device)
    if ratio_thresh:
        mask = mask & (dist_nn[..., 0] <= (ratio_thresh**2) * dist_nn[..., 1])
    if distance_thresh:
        mask = mask & (dist_nn[..., 0] <= distance_thresh**2)
    matches = torch.where(mask, ind_nn[..., 0], ind_nn.new_tensor(-1))
    return matches


def mutual_check(m0, m1):
    inds0 = torch.arange(m0.shape[-1], device=m0.device)
    inds1 = torch.arange(m1.shape[-1], device=m1.device)
    loop0 = torch.gather(m1, -1, torch.where(m0 > -1, m0, m0.new_tensor(0)))
    loop1 = torch.gather(m0, -1, torch.where(m1 > -1, m1, m1.new_tensor(0)))
    m0_new = torch.where((m0 > -1) & (inds0 == loop0), m0, m0.new_tensor(-1))
    m1_new = torch.where((m1 > -1) & (inds1 == loop1), m1, m1.new_tensor(-1))
    return m0_new, m1_new


class NearestNeighborMatcher(nn.Module):

    def __init__(self, ratio_thresh=None, distance_thresh=None, mutual_check=True):
        super().__init__()
        self.ratio_thresh = ratio_thresh
        self.distance_thresh = distance_thresh
        self.mutual_check = mutual_check

    def forward(self, feats0: torch.Tensor, feats1: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
        Returns:
            dict[str, Tensor]: a dictionary of matching results, including:
                - matches0: (B, N) tensor of indices of matches in desc1
                - matches1: (B, M) tensor of indices of matches in desc0
                - matching_scores0: (B, N) tensor of matching scores
                - matching_scores1: (B, M) tensor of matching scores
                - matched_kpts0: (B, N, 3) tensor of matched keypoints in image 0
                - matched_kpts1: (B, M, 3) tensor of matched keypoints in image 1
                - similarity: (B, N, M) tensor of descriptor similarity
                - log_assignment: (B, N+1, M+1) tensor of log assignment matrix
        """
        desc0 = feats0['sparse_descriptors']
        desc1 = feats1['sparse_descriptors']
        
        kpts0 = feats0['sparse_positions']
        kpts1 = feats1['sparse_positions']
        
        if kpts0.numel() == 0 or kpts1.numel() == 0:
            print("No keypoints found in either image")
            if kpts0.shape[0] > 1:
                return {
                    "matches0": desc0.new_full((desc0.shape[0], desc0.shape[1]), -1),
                    "matches1": desc1.new_full((desc1.shape[0], desc1.shape[1]), -1),
                    "matching_scores0": desc0.new_zeros((desc0.shape[0], desc0.shape[1])),
                    "matching_scores1": desc1.new_zeros((desc1.shape[0], desc1.shape[1])),
                    "matched_kpts0": [kpts0.new_zeros((kpts0.shape[1], 3))] * kpts0.shape[0],
                    "matched_kpts1": [kpts1.new_zeros((kpts1.shape[1], 3))] * kpts1.shape[0],
                    "similarity": desc0.new_zeros((desc0.shape[0], desc0.shape[1], desc1.shape[1])),
                    "log_assignment": desc0.new_zeros((desc0.shape[0], desc0.shape[1] + 1, desc1.shape[1] + 1)),
                }
            else:
                return {
                    "matches0": desc0.new_full((desc0.shape[0], desc0.shape[1]), -1),
                    "matches1": desc1.new_full((desc1.shape[0], desc1.shape[1]), -1),
                    "matching_scores0": desc0.new_zeros((desc0.shape[0], desc0.shape[1])),
                    "matching_scores1": desc1.new_zeros((desc1.shape[0], desc1.shape[1])),
                    "matched_kpts0": kpts0.new_zeros((0, 3)),
                    "matched_kpts1": kpts1.new_zeros((0, 3)),
                    "similarity": desc0.new_zeros((desc0.shape[0], desc0.shape[1], desc1.shape[1])),
                    "log_assignment": desc0.new_zeros((desc0.shape[0], desc0.shape[1] + 1, desc1.shape[1] + 1)),
                }
        
        sim = torch.einsum("bnd,bmd->bnm", desc0, desc1)
        matches0 = find_nn(sim, self.ratio_thresh, self.distance_thresh)
        matches1 = find_nn(
            sim.transpose(1, 2), self.ratio_thresh, self.distance_thresh
        )
        if self.mutual_check:
            matches0, matches1 = mutual_check(matches0, matches1)
            assert (matches0 > -1).sum() == (matches1 > -1).sum()
        b, n, m = sim.shape
        la = sim.new_zeros(b, n + 1, m + 1)
        la[:, :-1, :-1] = F.log_softmax(sim, -1) + F.log_softmax(sim, -2)
        
        mscores0 = (matches0 > -1).float()
        mscores1 = (matches1 > -1).float()
        
        if b > 1:
            matched_kpts0 = [] 
            matched_kpts1 = []
            for i in range(b):
                matched_kpts0_i = []
                matched_kpts1_i = []
                for j in range(n):
                    if matches0[i][j] > -1:
                        matched_kpts0_i.append(kpts0[i][j])
                        matched_kpts1_i.append(kpts1[i][matches0[i][j]])
                matched_kpts0_i = torch.stack(matched_kpts0_i, dim=0)
                matched_kpts1_i = torch.stack(matched_kpts1_i, dim=0)
                matched_kpts0.append(matched_kpts0_i)
                matched_kpts1.append(matched_kpts1_i)
                # matched_kpts0.append(kpts0[i][matches0[i] > -1])
                # matched_kpts1.append(kpts1[i][matches1[i] > -1])
        else:
            matched_kpts0 = []
            matched_kpts1 = []
            for j in range(n):
                if matches0[0][j] > -1:
                    matched_kpts0.append(kpts0[0][j])
                    matched_kpts1.append(kpts1[0][matches0[0][j]])
            matched_kpts0 = torch.stack(matched_kpts0, dim=0)
            matched_kpts1 = torch.stack(matched_kpts1, dim=0)
            # matched_kpts0 = kpts0[matches0 > -1]
            # matched_kpts1 = kpts1[matches1 > -1]
        
        return {
            "matches0": matches0,
            "matches1": matches1,
            "matching_scores0": mscores0,
            "matching_scores1": mscores1,
            "matched_kpts0": matched_kpts0,
            "matched_kpts1": matched_kpts1,
            "similarity": sim,
            "log_assignment": la,
        }
