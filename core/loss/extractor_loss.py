import torch
import torch.nn as nn
import torch.nn.functional as F


class ScoreLoss(nn.Module):
    def __init__(self, weight, mode, use_mask=True) -> None:
        super().__init__()
        self.mode = mode
        self.weight = weight
        self.use_mask = use_mask

    def forward(self, pred_feats, gt_feats, mask=None, padder=None):
        """
        Args:
            pred_feats (dict): the predicted features.
            gt_feats (dict): the ground truth features.

        Returns:
            loss (torch.Tensor): [B, 1], the loss.
            loss_info (dict): the loss info.
        """
        pred_keypoints_scores = pred_feats["score"]
        gt_keypoints_scores = gt_feats["score"]

        assert (
            pred_keypoints_scores.shape == gt_keypoints_scores.shape
        ), f"pred: {pred_keypoints_scores.shape}, gt: {gt_keypoints_scores.shape}"

        pred_keypoints_scores = pred_keypoints_scores.view(
            pred_keypoints_scores.shape[0], -1
        )
        gt_keypoints_scores = gt_keypoints_scores.view(gt_keypoints_scores.shape[0], -1)

        if not self.use_mask:
            mask = None

        if mask is not None:
            mask = mask.view(mask.shape[0], -1)

        if self.mode == "bce":
            gt_keypoints_scores = (gt_keypoints_scores > 0).float()
            loss = F.binary_cross_entropy(pred_keypoints_scores, gt_keypoints_scores)
        elif self.mode == "mse-whole":
            if mask is not None:
                gt_keypoints_scores[mask] = 0.0
            loss = F.mse_loss(pred_keypoints_scores, gt_keypoints_scores)
        elif self.mode == "mse":
            if mask is not None:
                loss = ((pred_keypoints_scores - gt_keypoints_scores) ** 2)[mask].mean()
            else:
                loss = ((pred_keypoints_scores - gt_keypoints_scores) ** 2).mean()
        elif self.mode == "mae":
            if mask is not None:
                loss = (
                    torch.abs(pred_keypoints_scores - gt_keypoints_scores) * mask
                ).sum() / mask.sum()
            else:
                loss = torch.abs(pred_keypoints_scores - gt_keypoints_scores).mean()

        else:
            raise NotImplementedError(f"Not implemented mode: {self.mode}")

        loss = loss * self.weight

        loss_info = {
            "extractor_keypoints_loss": loss.detach().item(),
        }
        return loss, loss_info


class LogitsLoss(nn.Module):
    def __init__(self, weight, mode, cell_size) -> None:
        super().__init__()
        self.mode = mode
        self.weight = weight
        self.cell_size = cell_size

    def forward(self, pred_feats, gt_feats, mask=None, padder=None):
        """
        Args:
            pred_feats (dict): the predicted features.
            gt_feats (dict): the ground truth features.

        Returns:
            loss (torch.Tensor): [B, 1], the loss.
            loss_info (dict): the loss info.
        """
        pred_keypoints_logits = pred_feats["logits"]
        gt_keypoints_logits = gt_feats["logits"]

        assert (
            pred_keypoints_logits.shape == gt_keypoints_logits.shape
        ), f"pred: {pred_keypoints_logits.shape}, gt: {gt_keypoints_logits.shape}"

        channel_dim = pred_keypoints_logits.shape[1]
        assert (
            channel_dim == self.cell_size * self.cell_size + 1
        ), f"channel_dim: {channel_dim}, cell_size: {self.cell_size}"

        if self.cell_size > 1:
            # remove the last (dustbin) cell from the list
            pred_keypoints_logits, _ = pred_keypoints_logits.split(
                self.cell_size * self.cell_size, dim=1
            )
            gt_keypoints_logits, _ = gt_keypoints_logits.split(
                self.cell_size * self.cell_size, dim=1
            )

            # change the dimensions to get an output shape of (batch_size, H, W)
            pred_keypoints_logits = F.pixel_shuffle(
                pred_keypoints_logits, self.cell_size
            )
            gt_keypoints_logits = F.pixel_shuffle(gt_keypoints_logits, self.cell_size)

        else:
            assert (
                channel_dim == 1
            ), f"channel_dim: {channel_dim}, cell_size: {self.cell_size}"

        if padder is not None:
            pred_keypoints_logits, gt_keypoints_logits = padder.unpad(
                pred_keypoints_logits, gt_keypoints_logits
            )

        pred_keypoints_logits = pred_keypoints_logits.view(
            pred_keypoints_logits.shape[0], -1
        )
        gt_keypoints_logits = gt_keypoints_logits.view(gt_keypoints_logits.shape[0], -1)
        if mask is not None:
            mask = mask.view(mask.shape[0], -1)

        loss = F.mse_loss(pred_keypoints_logits, gt_keypoints_logits, reduction="none")
        if mask is not None:
            loss = loss * mask
        loss = loss.mean()

        loss = loss * self.weight

        loss_info = {
            "extractor_keypoints_loss": loss.detach().item(),
        }
        return loss, loss_info


class DescriptorsLoss(nn.Module):
    def __init__(
        self, weight, desc_type="normalized", mode="mse", use_mask=True, **kargs
    ) -> None:
        super().__init__()
        self.weight = weight

        assert desc_type in ("normalized", "raw", "coarse")
        self.desc_type = desc_type
        self.mode = mode
        self.use_mask = use_mask
        self.kargs = kargs

    def contrastive_loss(self, pred_desc, gt_desc, mask, kernel_size=5):
        B, C, H, W = pred_desc.shape
        pred_desc = (
            pred_desc.permute(0, 2, 3, 1).view(B, H * W, C).contiguous()
        )  # [B, H*W, C]
        gt_desc = (
            gt_desc.permute(0, 2, 3, 1).view(B, H * W, C).contiguous()
        )  # [B, H*W, C]
        mask = mask.view(B, H * W).contiguous()  # [B, H*W]

    def dual_softmax_loss(self, pred_desc, gt_desc, mask=None):
        B, C, H, W = pred_desc.shape
        pred_desc = (
            pred_desc.permute(0, 2, 3, 1).view(B, H * W, C).contiguous()
        )  # [B, H*W, C]
        gt_desc = (
            gt_desc.permute(0, 2, 3, 1).view(B, H * W, C).contiguous()
        )  # [B, H*W, C]
        if mask is None:
            mask = torch.ones(
                (B, H * W), dtype=pred_desc.dtype, device=pred_desc.device
            )
        mask = mask.view(B, H * W).contiguous()  # [B, H*W]

        loss = 0.0
        sim_matrix = pred_desc @ gt_desc.transpose(-1, -2)  # [H*W, H*W]
        conf_matrix = F.softmax(sim_matrix, -1) * F.softmax(
            sim_matrix, -2
        )  # apply dual softmax

        mask_i = mask.float()
        conf_gt = mask_i.unsqueeze(-1) @ mask_i.unsqueeze(-2)  # [H*W, H*W]
        loss = -torch.log(conf_matrix[conf_gt > 0] + 1e-8).mean()

        loss = loss / B

        return loss

    def triplet_loss(self, pred_desc, gt_desc, mask=None):
        """use coarse descriptors to calculate the triplet loss."""
        margin = 0.2
        B, C, H, W = pred_desc.shape
        pred_desc = (
            pred_desc.permute(0, 2, 3, 1).view(B, H * W, C).contiguous()
        )  # [B, H*W, C]
        gt_desc = (
            gt_desc.permute(0, 2, 3, 1).view(B, H * W, C).contiguous()
        )  # [B, H*W, C]

        if mask is None:
            mask = torch.ones(
                (B, H * W), dtype=pred_desc.dtype, device=pred_desc.device
            )
        else:
            assert mask.dim() == 4

            if mask.shape[1] != 1:
                mask = mask[:, 0, :, :].unsqueeze(1).contiguous()

            if mask.shape[-2] != H or mask.shape[-1] != W:
                mask = F.interpolate(mask.float(), size=(H, W), mode="bilinear") > 0.5

        mask = mask.view(B, H * W, 1).float().contiguous()  # [B, H*W]
        mask = mask @ mask.transpose(-1, -2)  # [B, H*W, H*W]

        # triplet
        distance = torch.cdist(pred_desc, gt_desc, p=2)  # [B, H*W, H*W]

        diag = (
            torch.eye(H * W, dtype=pred_desc.dtype, device=pred_desc.device)
            .unsqueeze(0)
            .repeat(B, 1, 1)
        )  # [B, H*W, H*W]
        d_pos = (
            distance[diag > 0]
            .view(B, H * W, 1)
            .repeat(1, 1, H * W)
            .view(B, H * W, H * W)
        )  # [B, H*W, H*W]
        loss_map = torch.clamp(d_pos - distance + margin, min=0.0)  # [B, H*W, H*W]

        loss_map[diag > 0] = 0.0
        loss_map[mask <= 0] = 0.0

        loss = loss_map.mean()

        return loss

    def forward(self, pred_feats, gt_feats, mask: torch.Tensor = None):
        """
        Args:
            pred_feats (dict): the predicted features.
            gt_feats (dict): the ground truth features.

        Returns:
            loss (torch.Tensor): [B, 1], the loss.
            loss_info (dict): the loss info.
        """

        if self.desc_type == "normalized":
            pred_descriptors = pred_feats["normalized_descriptors"]
            gt_descriptors = gt_feats["normalized_descriptors"]
        elif self.desc_type == "raw":
            pred_descriptors = pred_feats["raw_descriptors"]
            gt_descriptors = gt_feats["raw_descriptors"]
        elif self.desc_type == "coarse":
            pred_descriptors = pred_feats["coarse_descriptors"]
            gt_descriptors = gt_feats["coarse_descriptors"]

        if not self.use_mask:
            mask = None

        if mask is not None and mask.shape[1] == 1:
            mask = mask.repeat(1, pred_descriptors.shape[1], 1, 1)

        assert pred_descriptors.shape == gt_descriptors.shape

        if self.mode == "mse":
            # loss = (torch.norm(pred_descriptors - gt_descriptors, dim=1) * mask).sum() / mask.sum()
            if mask is not None:
                if mask.sum() == 0:
                    loss = torch.tensor(
                        0.0,
                        dtype=pred_descriptors.dtype,
                        device=pred_descriptors.device,
                    )
                loss = (
                    ((pred_descriptors - gt_descriptors) ** 2) * mask
                ).sum() / mask.sum()
            else:
                loss = F.mse_loss(pred_descriptors - gt_descriptors)
        elif self.mode == "mae":
            if mask is not None:
                if mask.sum() == 0:
                    loss = torch.tensor(
                        0.0,
                        dtype=pred_descriptors.dtype,
                        device=pred_descriptors.device,
                    )
                loss = (
                    torch.abs(pred_descriptors - gt_descriptors) * mask
                ).sum() / mask.sum()
            else:
                loss = F.l1_loss(pred_descriptors, gt_descriptors)
        elif self.mode == "cosine_similarity":
            if mask is not None:
                if mask.sum() == 0:
                    loss = torch.tensor(
                        0.0,
                        dtype=pred_descriptors.dtype,
                        device=pred_descriptors.device,
                    )
                cosine_sim = F.cosine_similarity(
                    pred_descriptors, gt_descriptors, dim=1
                )
                loss = 1 - cosine_sim.view(-1)[mask.view(-1)].mean()
            else:
                cosine_sim = F.cosine_similarity(
                    pred_descriptors, gt_descriptors, dim=1
                )
                loss = 1 - cosine_sim.mean()
        elif self.mode == "dual-softmax":
            loss = self.dual_softmax_loss(pred_descriptors, gt_descriptors, mask=mask)
        elif self.mode == "triplet":
            loss = self.triplet_loss(pred_descriptors, gt_descriptors, mask=mask)
        elif self.mode == "mae+triplet":
            mae_weight = self.kargs["mae+triplet"]["mae_weight"]
            triplet_weight = self.kargs["mae+triplet"]["triplet_weight"]
            # mae
            pred_descriptors = pred_feats["normalized_descriptors"]
            gt_descriptors = gt_feats["normalized_descriptors"]
            if mask is not None:
                if mask.sum() == 0:
                    mae_loss = torch.tensor(
                        0.0,
                        dtype=pred_descriptors.dtype,
                        device=pred_descriptors.device,
                    )
                mae_loss = (
                    torch.abs(pred_descriptors - gt_descriptors) * mask
                ).sum() / mask.sum()
            else:
                mae_loss = F.l1_loss(pred_descriptors, gt_descriptors)
            # triplet, use raw descriptors
            pred_descriptors = pred_feats["raw_descriptors"]
            gt_descriptors = pred_feats["raw_descriptors"]
            triplet_loss = self.triplet_loss(pred_descriptors, gt_descriptors, mask)
            loss = mae_weight * mae_loss + triplet_weight * triplet_loss
        else:
            raise NotImplementedError(f"Not implemented mode: {self.mode}")
        loss = loss * self.weight

        loss_info = {
            "extractor_descriptor_loss": loss.detach().item(),
        }
        return loss, loss_info


class FeatureLoss(nn.Module):
    def __init__(self, weight, mode, **kargs) -> None:
        super().__init__()
        self.weight = weight
        self.mode = mode

    def forward(self, pred_feats, gt_feats):
        pred_features = pred_feats["backbone_feats"]
        gt_features = gt_feats["backbone_feats"]

        assert (
            pred_features.shape == gt_features.shape
        ), f"pred: {pred_features.shape}, gt: {gt_features.shape}"

        if self.mode == "mse":
            loss = F.mse_loss(pred_features, gt_features)
        elif self.mode == "mae":
            loss = F.l1_loss(pred_features, gt_features)
        else:
            raise NotImplementedError(f"Not implemented mode: {self.mode}")

        loss = loss * self.weight

        loss_info = {
            "feature_loss": loss.detach().item(),
        }
        return loss, loss_info
