import torch
from torch import nn
from torch.nn import Module, Parameter


class TensorGraphConvolution(Module):
    """Band-limited temporal tensor graph convolution."""

    def __init__(self, time_slices, in_features, out_features, band_width, tgc_dropout=0.6):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.time_slices = time_slices
        self.band_width = band_width
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.dropout = nn.Dropout(tgc_dropout)
        self.reset_parameters()

    def compute_AtXt(self, A, X, M):
        At = self.func_MProduct(A, M, self.time_slices)
        Xt = torch.matmul(M, X.reshape(self.time_slices, -1)).reshape(X.size())
        num_nodes = X.size()[1]
        AtXt = torch.zeros(self.time_slices, num_nodes, X.size()[-1], device=M.device)
        for k in range(self.time_slices):
            AtXt[k] = torch.sparse.mm(At[k], Xt[k])
        return AtXt

    def forward(self, adj, input, M):
        h = self.compute_AtXt(adj, input, M)
        return torch.matmul(self.dropout(h), self.weight)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def func_MProduct(self, input, M, time_slices):
        len_M = self.band_width
        device = M.device
        num_nodes = input[0].size()[-1]
        sz = torch.Size([num_nodes, num_nodes])
        res = []
        for tm in range(time_slices):
            temp = torch.sparse_coo_tensor(sz, device=device)
            if tm < len_M:
                m = M[tm, 0:tm + 1]
                for i in range(tm + 1):
                    A = input[i]
                    val = m[i] * A._values()
                    temp = temp + torch.sparse_coo_tensor(A._indices(), val, sz, device=device)
            else:
                m = M[tm, tm - len_M + 1: tm + 1]
                for i in range(len_M):
                    A = input[tm - len_M + i]
                    val = m[i] * A._values()
                    temp = temp + torch.sparse_coo_tensor(A._indices(), val, sz, device=device)
            res.append(temp.coalesce())
        return res


class FFTLayer(Module):
    """Frequency-domain temporal feature mixer."""

    def __init__(self, time_slices, in_features, out_features, fft_dropout):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.time_slices = time_slices
        self.weight = Parameter(torch.empty(in_features, out_features, dtype=torch.complex64))
        self.fft_dropout = nn.Dropout(fft_dropout)
        self.activation = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, tensor):
        tensor = self.fft_dropout(tensor)
        tensor_fft = torch.fft.rfft(tensor, dim=1)
        tensor_fft = torch.matmul(tensor_fft, self.weight)
        tensor_fft_real = self.activation(tensor_fft.real)
        tensor_fft_imag = self.activation(tensor_fft.imag)
        tensor_fft = torch.complex(tensor_fft_real, tensor_fft_imag)
        return torch.fft.irfft(tensor_fft, tensor.shape[1], dim=1)
