from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def parse_degree_bins(value):
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return [float(v.strip()) for v in str(value).split(",") if v.strip()]


def build_prefix_mean_context(h1: Tensor) -> Tensor:
    h1_norm = F.normalize(h1)
    prefix_sum = h1_norm.cumsum(dim=0)
    counts = torch.arange(1, h1.size(0) + 1, dtype=h1.dtype, device=h1.device).view(-1, 1, 1)
    return prefix_sum / counts


def build_cl_node_weight(
    graph_stats: Optional[Dict],
    time_idx: int,
    num_nodes: int,
    device,
    degree_bins,
    eta: float,
    w_min: float,
    w_max: float,
) -> Optional[Tensor]:
    if graph_stats is None or "degrees" not in graph_stats:
        return None

    idx = min(time_idx, len(graph_stats["degrees"]) - 1)
    deg = graph_stats["degrees"][idx].to(device).float()
    if deg.numel() != num_nodes:
        deg = deg[:num_nodes]

    bins = torch.tensor(degree_bins, dtype=deg.dtype, device=device)
    bucket_id = torch.bucketize(deg, bins)
    num_bins = bins.numel() + 1
    bucket_freq = torch.bincount(bucket_id, minlength=num_bins).float()
    bucket_freq = bucket_freq / bucket_freq.sum().clamp_min(1e-12)
    occupied = bucket_freq > 0
    bucket_weight = torch.ones(num_bins, dtype=deg.dtype, device=device)
    bucket_weight[occupied] = 1.0 / (bucket_freq[occupied] + 1e-12).pow(eta)

    sample_weight = bucket_weight[bucket_id]
    sample_weight = sample_weight / sample_weight.mean().clamp_min(1e-12)
    return sample_weight.clamp(w_min, w_max)


def build_time_disagreement(
    scale_logits: Optional[Tensor],
    edge_time_ids: Optional[Tensor],
    T: int,
    normalize: bool = True,
) -> Optional[Tensor]:
    if scale_logits is None or edge_time_ids is None:
        return None
    if scale_logits.numel() == 0 or scale_logits.dim() != 2:
        return None

    device = scale_logits.device
    t_ids = edge_time_ids.to(device).long().clamp(min=0, max=T - 1)
    edge_disagreement = scale_logits.detach().float().var(dim=-1, unbiased=False)

    time_sum = torch.zeros(T, dtype=edge_disagreement.dtype, device=device)
    time_count = torch.zeros(T, dtype=edge_disagreement.dtype, device=device)
    time_sum.index_add_(0, t_ids, edge_disagreement)
    time_count.index_add_(0, t_ids, torch.ones_like(edge_disagreement))

    occupied = time_count > 0
    time_disagreement = torch.zeros(T, dtype=edge_disagreement.dtype, device=device)
    time_disagreement[occupied] = time_sum[occupied] / time_count[occupied].clamp_min(1e-12)
    if normalize and occupied.any():
        time_disagreement = time_disagreement / time_disagreement[occupied].mean().clamp_min(1e-12)
    return time_disagreement


def apply_disagreement_factor(
    sample_weight: Tensor,
    time_disagreement: Optional[Tensor],
    time_idx: int,
    alpha: float,
    factor_min: float,
    factor_max: float,
) -> Tensor:
    if time_disagreement is None:
        return sample_weight
    idx = min(time_idx, time_disagreement.numel() - 1)
    factor = 1.0 - alpha * time_disagreement[idx]
    return sample_weight * torch.clamp(factor, min=factor_min, max=factor_max)


def temporal_cl_loss(
    h1: Tensor,
    h2: Tensor,
    tau: float,
    graph_stats: Optional[Dict],
    time_disagreement: Optional[Tensor],
    cl_degree_bins,
    cl_powerlaw_eta: float,
    cl_weight_min: float,
    cl_weight_max: float,
    cl_disagreement_alpha: float,
    cl_disagreement_min: float,
    cl_disagreement_max: float,
) -> Tensor:
    context = build_prefix_mean_context(h1)
    losses = []
    for i in range(h1.shape[0]):
        u, v = F.normalize(h1[i]), F.normalize(h2[i])
        s = context[i]

        pos_similarity = torch.sum(u * s, dim=1) / tau
        neg_similarity = torch.sum(v * s, dim=1) / tau
        similarity = torch.cat((pos_similarity, neg_similarity))
        labels = torch.cat((torch.ones_like(pos_similarity), torch.zeros_like(neg_similarity)))

        loss_each = F.binary_cross_entropy_with_logits(similarity, labels, reduction="none")
        node_weight = build_cl_node_weight(
            graph_stats=graph_stats,
            time_idx=i,
            num_nodes=h1.size(1),
            device=h1.device,
            degree_bins=cl_degree_bins,
            eta=cl_powerlaw_eta,
            w_min=cl_weight_min,
            w_max=cl_weight_max,
        )
        sample_weight = torch.cat([node_weight, node_weight], dim=0) if node_weight is not None else torch.ones_like(loss_each)
        sample_weight = apply_disagreement_factor(
            sample_weight,
            time_disagreement,
            i,
            cl_disagreement_alpha,
            cl_disagreement_min,
            cl_disagreement_max,
        )
        losses.append((sample_weight * loss_each).mean())

    return torch.stack(losses, dim=0)


class MultiScaleLoss(nn.Module):
    """Loss for M1 prior-conditioned LoRA PL-DisGLSL."""

    def __init__(
        self,
        lam: float = 0.0005,
        tau: float = 0.1,
        cl_powerlaw_eta: float = 0.5,
        cl_weight_min: float = 0.5,
        cl_weight_max: float = 2.0,
        cl_degree_bins: str = "0,1,2,4,8,16,32,64,128",
        cl_disagreement_alpha: float = 0.5,
        cl_disagreement_min: float = 0.2,
        cl_disagreement_max: float = 1.0,
        cl_disagreement_normalize: bool = True,
    ):
        super().__init__()
        self.lam = lam
        self.tau = tau
        self.cl_powerlaw_eta = cl_powerlaw_eta
        self.cl_weight_min = cl_weight_min
        self.cl_weight_max = cl_weight_max
        self.cl_degree_bins = parse_degree_bins(cl_degree_bins)
        self.cl_disagreement_alpha = cl_disagreement_alpha
        self.cl_disagreement_min = cl_disagreement_min
        self.cl_disagreement_max = cl_disagreement_max
        self.cl_disagreement_normalize = cl_disagreement_normalize
        self.bce_logits = nn.BCEWithLogitsLoss()

    def forward(
        self,
        final_logits: Tensor,
        target: Tensor,
        feature_reg_loss: Tensor,
        cl_h1: Tensor,
        cl_h2: Tensor,
        scale_logits: Tensor,
        graph_stats: Dict,
        cl_edge_time_ids: Tensor,
    ) -> Dict[str, Tensor]:
        losses = {}
        losses["edge"] = self.bce_logits(final_logits, target)

        time_disagreement = build_time_disagreement(
            scale_logits=scale_logits,
            edge_time_ids=cl_edge_time_ids,
            T=cl_h1.shape[0],
            normalize=self.cl_disagreement_normalize,
        )
        losses["cl_disagreement"] = (
            time_disagreement.mean()
            if time_disagreement is not None
            else torch.tensor(0.0, device=final_logits.device)
        )
        losses["contrastive"] = temporal_cl_loss(
            h1=cl_h1,
            h2=cl_h2,
            tau=self.tau,
            graph_stats=graph_stats,
            time_disagreement=time_disagreement,
            cl_degree_bins=self.cl_degree_bins,
            cl_powerlaw_eta=self.cl_powerlaw_eta,
            cl_weight_min=self.cl_weight_min,
            cl_weight_max=self.cl_weight_max,
            cl_disagreement_alpha=self.cl_disagreement_alpha,
            cl_disagreement_min=self.cl_disagreement_min,
            cl_disagreement_max=self.cl_disagreement_max,
        ).mean()

        losses["x"] = self.lam * feature_reg_loss

        losses["total"] = (
            losses["edge"]
            + losses["contrastive"]
            + losses["x"]
        )
        return losses
