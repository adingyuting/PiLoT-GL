from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


SCALE_NAMES = ("short", "mid", "long")


class EdgeScorer(nn.Module):
    """Shared edge scorer for all three scales."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, edge_repr: Tensor) -> Tensor:
        return self.mlp(edge_repr).squeeze(-1)


class ObservabilityFeatureBuilder(nn.Module):
    """Build edge-level observability features from split-specific graph stats."""

    def forward(self, edge_index: Tensor, edge_time_ids: Tensor, graph_stats: Dict) -> Tensor:
        device = edge_index.device
        t_idx = edge_time_ids.long()
        u = edge_index[0]
        v = edge_index[1]

        degrees = torch.stack(graph_stats["degrees"], dim=0)
        gaps = torch.stack(graph_stats["gaps"], dim=0)
        recent_counts = torch.stack(graph_stats["recent_counts"], dim=0)
        num_edges_t = torch.tensor(graph_stats["num_edges"], dtype=torch.float, device=device)
        active_ratios = torch.tensor(graph_stats["active_ratios"], dtype=torch.float, device=device)

        cnt_u = recent_counts[t_idx, u]
        cnt_v = recent_counts[t_idx, v]
        cnt_uv = torch.sqrt(cnt_u * cnt_v + 1e-8)
        deg_u = degrees[t_idx, u]
        deg_v = degrees[t_idx, v]
        gap_u = gaps[t_idx, u]
        gap_v = gaps[t_idx, v]
        global_edges = num_edges_t[t_idx]
        miss_rate = 1.0 - active_ratios[t_idx]

        return torch.stack(
            [
                torch.log1p(cnt_u),
                torch.log1p(cnt_v),
                torch.log1p(cnt_uv),
                torch.log1p(deg_u),
                torch.log1p(deg_v),
                torch.log1p(gap_u),
                torch.log1p(gap_v),
                torch.log1p(global_edges),
                miss_rate,
            ],
            dim=-1,
        )


class MultiScaleCDPSS(nn.Module):
    """
    Single-model CDPSS for M1-prior-conditioned-lora-PL-DisGLSL.

    Fixed design:
    - input-level short/mid/long node embeddings are provided by train.py
    - shared edge scorer plus scale-specific LoRA residuals
    - LoRA residuals are gated by time-slot prior features
    - scale logits are fused by learned scale attention
    """

    def __init__(
        self,
        emb_dim: int,
        num_nodes: int,
        lora_rank: int = 8,
        attention_temperature: float = 1.5,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.attention_temperature = attention_temperature
        self.num_scales = len(SCALE_NAMES)
        self.scale_names = SCALE_NAMES

        edge_dim = emb_dim * 4
        obs_dim = 9
        evidence_dim = edge_dim + 1 + obs_dim

        self.edge_scorer = EdgeScorer(edge_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.lora_down = nn.ModuleList(
            nn.Linear(edge_dim, lora_rank, bias=False)
            for _ in range(self.num_scales)
        )
        self.lora_up = nn.ModuleList(
            nn.Linear(lora_rank, 1, bias=False)
            for _ in range(self.num_scales)
        )
        for up in self.lora_up:
            nn.init.zeros_(up.weight)

        self.lora_prior_mlp = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_scales),
        )
        self.gamma_g = nn.Parameter(torch.zeros(1))

        self.obs_builder = ObservabilityFeatureBuilder()
        self.attention_mlp = nn.Sequential(
            nn.Linear(evidence_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _decode_time_from_edges(self, edge_src_nodes: Tensor) -> Tensor:
        return edge_src_nodes // self.num_nodes

    @staticmethod
    def _edge_repr(h_u: Tensor, h_v: Tensor) -> Tensor:
        return torch.cat([h_u, h_v, h_u * h_v, torch.abs(h_u - h_v)], dim=-1)

    def _build_time_prior_feat(self, edge_time_ids: Tensor, graph_stats: Dict) -> Tensor:
        device = edge_time_ids.device
        t_idx = edge_time_ids.long()

        degrees = torch.stack(graph_stats["degrees"], dim=0)
        num_edges_t = torch.tensor(graph_stats["num_edges"], dtype=torch.float, device=device)
        active_ratios = torch.tensor(graph_stats["active_ratios"], dtype=torch.float, device=device)

        edge_count = num_edges_t[t_idx]
        prev_t = torch.clamp(t_idx - 1, min=0)
        prev_edge_count = num_edges_t[prev_t]
        density = torch.log1p(edge_count)
        edge_burst = torch.log1p(torch.relu(edge_count - prev_edge_count))
        activity = active_ratios[t_idx]
        sparsity = 1.0 - activity

        deg_prob = degrees / (degrees.sum(dim=-1, keepdim=True) + 1e-8)
        entropy_by_time = -(deg_prob * torch.log(deg_prob + 1e-8)).sum(dim=-1)
        entropy_by_time = entropy_by_time / torch.log(torch.tensor(float(degrees.size(1)), device=device))
        degree_entropy = entropy_by_time[t_idx]

        return torch.stack([density, edge_burst, sparsity, activity, degree_entropy], dim=-1)

    def _gather_pairs(self, node_emb_by_scale: Dict[str, Tensor], edge_index: Tensor, t_ids: Tensor) -> Dict[str, Tuple[Tensor, Tensor]]:
        u_flat = t_ids.long() * self.num_nodes + edge_index[0]
        v_flat = t_ids.long() * self.num_nodes + edge_index[1]
        pairs = {}
        for scale in self.scale_names:
            H = node_emb_by_scale[scale]
            H_flat = H.reshape(-1, H.size(-1))
            pairs[scale] = (H_flat[u_flat], H_flat[v_flat])
        return pairs

    @staticmethod
    def _fuse_scale_logits(scale_logits: Tensor, alpha: Tensor) -> Tensor:
        return (alpha * scale_logits).sum(dim=-1)

    def forward(
        self,
        node_emb_by_scale: Dict[str, Tensor],
        edge_index: Tensor,
        edge_src_nodes: Tensor,
        edge_trg_nodes: Tensor,
        graph_stats: Dict,
    ) -> Dict[str, Tensor]:
        del edge_trg_nodes
        t_ids = self._decode_time_from_edges(edge_src_nodes)
        scale_pairs = self._gather_pairs(node_emb_by_scale, edge_index, t_ids)

        prior_feat = self._build_time_prior_feat(t_ids, graph_stats)
        lora_gate = 1.0 + self.gamma_g * torch.tanh(self.lora_prior_mlp(prior_feat))

        edge_reprs = []
        scale_logits_cols = []
        for scale_id, scale in enumerate(self.scale_names):
            h_u, h_v = scale_pairs[scale]
            edge_repr = self._edge_repr(h_u, h_v)
            edge_reprs.append(edge_repr)

            scale_logit = self.edge_scorer(edge_repr)
            lora_residual = self.lora_up[scale_id](self.lora_down[scale_id](edge_repr)).squeeze(-1)
            scale_logit = scale_logit + lora_gate[:, scale_id] * lora_residual
            scale_logits_cols.append(scale_logit)

        scale_logits = torch.stack(scale_logits_cols, dim=-1)
        obs_feat = self.obs_builder(edge_index, t_ids, graph_stats)

        alpha_logits = []
        for scale_id, edge_repr in enumerate(edge_reprs):
            evidence = torch.cat([edge_repr, scale_logits[:, scale_id:scale_id + 1], obs_feat], dim=-1)
            alpha_logits.append(self.attention_mlp(evidence).squeeze(-1))
        alpha_logits = torch.stack(alpha_logits, dim=-1)
        alpha = F.softmax(alpha_logits / self.attention_temperature, dim=-1)
        final_logit = self._fuse_scale_logits(scale_logits, alpha)

        return {
            "final_logits": final_logit,
            "scale_logits": scale_logits,
            "alpha": alpha,
            "attention_logits": alpha_logits,
            "obs_feat": obs_feat,
            "lora_gate": lora_gate,
        }
