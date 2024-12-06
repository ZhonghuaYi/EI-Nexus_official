from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
from rich import print


class MNNLoss(nn.Module):
    """
    Loss for Mutual Nearest Neighbor (MNN) matcher. No learnable parameters.
    """

    def __init__(self, weight) -> None:
        super().__init__()
        self.weight = weight

    def forward(
        self, pred_match: torch.Tensor, gt_match: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            pred_match (dict): a dict of predicted matches.
            gt_match (dict): a dict of ground truth matches.

        Returns:
            loss (torch.Tensor): [B, 1], the loss.
            loss_info (dict): the loss info.
        """
        similarity = pred_match["similarity"]
        gt_assignment = gt_match["assignment"]

        sim = similarity

        if torch.any(sim > (1.0 + 1e-6)):
            logging.warning(f"Similarity larger than 1, max={sim.max()}")
            print(
                f"[bold yellow]Warning |[/bold yellow] Similarity larger than 1, max={sim.max()}"
            )

        scores = torch.sqrt(torch.clamp(2 * (1 - sim), min=1e-6))
        scores = 2 - scores

        assert not torch.any(torch.isnan(scores)), torch.any(torch.isnan(sim))

        prob0 = torch.nn.functional.log_softmax(scores, 2)
        prob1 = torch.nn.functional.log_softmax(scores, 1)

        assignment = gt_assignment.float()
        num = torch.max(assignment.sum((1, 2)), assignment.new_tensor(1))

        nll0 = (prob0 * assignment).sum((1, 2)) / num
        nll1 = (prob1 * assignment).sum((1, 2)) / num
        nll = -(nll0 + nll1) / 2
        loss = nll.mean() * self.weight

        loss_info = {
            "matcher_n_pair_nll": loss.detach().item(),
            "matcher_total": loss.detach().item(),
            "matcher_num_matchable": num.mean().detach().item(),
        }

        return loss, loss_info


class NLLLoss(nn.Module):

    def __init__(self, weight, nll_balancing=0.5):
        super().__init__()
        self.loss_fn = self.nll_loss
        self.weight = weight
        self.nll_balancing = nll_balancing

    @staticmethod
    def weight_loss(log_assignment, weights):
        b, m, n = log_assignment.shape
        m -= 1
        n -= 1

        loss_sc = log_assignment * weights

        num_neg0 = weights[:, :m, -1].sum(-1).clamp(min=1.0)
        num_neg1 = weights[:, -1, :n].sum(-1).clamp(min=1.0)
        num_pos = weights[:, :m, :n].sum((-1, -2)).clamp(min=1.0)

        nll_pos = -loss_sc[:, :m, :n].sum((-1, -2))
        nll_pos /= num_pos.clamp(min=1.0)

        nll_neg0 = -loss_sc[:, :m, -1].sum(-1)
        nll_neg1 = -loss_sc[:, -1, :n].sum(-1)

        nll_neg = (nll_neg0 + nll_neg1) / (num_neg0 + num_neg1)

        return nll_pos, nll_neg, num_pos, (num_neg0 + num_neg1) / 2.0

    def forward(
        self, log_assignment, gt_matches0, gt_matches1, gt_assignment, weights=None
    ):
        if weights is None:
            weights = self.loss_fn(
                log_assignment, gt_matches0, gt_matches1, gt_assignment
            )
        nll_pos, nll_neg, num_pos, num_neg = self.weight_loss(log_assignment, weights)
        nll = self.nll_balancing * nll_pos + (1 - self.nll_balancing) * nll_neg

        loss = nll.mean() * self.weight
        loss_info = {
            "matcher_n_pair_nll": loss.detach().item(),
            "matcher_nll_positive": nll_pos.mean().detach().item(),
            "matcher_nll_negtive": nll_neg.mean().detach().item(),
            "matcher_num_matchable": num_pos.detach().item(),
            "matcher_num_unmatchable": num_neg.detach().item(),
        }

        return loss, loss_info

    @staticmethod
    def nll_loss(log_assignment, gt_matches0, gt_matches1, gt_assignment):
        m, n = gt_matches0.size(-1), gt_matches1.size(-1)
        positive = gt_assignment.float()
        neg0 = (gt_matches0 == -1).float()
        neg1 = (gt_matches1 == -1).float()

        weights = torch.zeros_like(log_assignment)
        weights[:, :m, :n] = positive

        weights[:, :m, -1] = neg0
        weights[:, -1, :n] = neg1
        return weights
