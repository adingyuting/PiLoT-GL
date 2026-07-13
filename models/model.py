import torch
import torch.nn as nn

from models.layers import FFTLayer, TensorGraphConvolution


class SGDDyG(nn.Module):
    """Temporal encoder used by M1 prior-conditioned LoRA PL-DisGLSL."""

    def __init__(
        self,
        time_slices,
        num_nodes,
        hidden_features,
        num_feature,
        bandwidth,
        tgc_dropout=0.6,
        fft_dropout=0.6,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.layers = len(hidden_features)
        self.feature_dims = [num_feature] + hidden_features
        self.time_slices = time_slices
        self.X = nn.Parameter(torch.empty(time_slices, num_nodes, num_feature))

        self.fft_layer = FFTLayer(time_slices, num_feature, num_feature, fft_dropout)
        self.tgcs = nn.ModuleList(
            TensorGraphConvolution(
                time_slices,
                self.feature_dims[layer],
                self.feature_dims[layer + 1],
                bandwidth,
                tgc_dropout,
            )
            for layer in range(self.layers)
        )
        self.activation = nn.ReLU()
        self.init_weight()

    def forward(self, A, edges_nodes, M, cl=True, return_embeddings=False, scale_id=None):
        del edges_nodes, return_embeddings, scale_id
        H = self.X if cl else self.fft_layer(self.X)

        for layer, tgc in enumerate(self.tgcs):
            H = tgc(A, H, M)
            if layer != self.layers - 1:
                H = self.activation(H)

        return {
            "node_emb_seq": H,
            "feature_reg_loss": torch.norm(self.X, p=2),
        }

    def init_weight(self):
        nn.init.xavier_normal_(self.X)
