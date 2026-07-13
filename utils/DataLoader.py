import random

import numpy as np
import scipy.io as sio
import torch

from constant import *
from utils import util


def func_get_floatTensor(saved_content, tensor_name, size, device):
    indices = torch.tensor(np.array(saved_content[tensor_name + '_idx'], dtype=int), dtype=torch.long, device=device)
    values = torch.tensor(saved_content[tensor_name + '_vals'], dtype=torch.float, device=device)
    return torch.sparse_coo_tensor(indices, torch.squeeze(values), size, dtype=torch.float, device=device)


def split_sparse_tensor_by_time(tensor, num_steps, num_nodes, device):
    tensor = tensor.coalesce()
    indices = tensor._indices()
    values = tensor._values()
    if indices.numel() == 0:
        empty = torch.empty((2, 0), dtype=torch.long, device=device)
        empty_values = torch.empty((0,), dtype=values.dtype, device=device)
        return [
            torch.sparse_coo_tensor(empty, empty_values, (num_nodes, num_nodes), device=device).coalesce()
            for _ in range(num_steps)
        ]

    order = torch.argsort(indices[0])
    times = indices[0, order]
    spatial_indices = indices[1:3, order]
    sorted_values = values[order]
    boundaries = torch.searchsorted(
        times,
        torch.arange(num_steps + 1, dtype=times.dtype, device=device),
        right=False,
    )

    slices = []
    for t in range(num_steps):
        start = int(boundaries[t].item())
        end = int(boundaries[t + 1].item())
        slices.append(
            torch.sparse_coo_tensor(
                spatial_indices[:, start:end],
                sorted_values[start:end],
                (num_nodes, num_nodes),
                device=device,
            ).coalesce()
        )
    return slices


def load_data(args, device):
    util.set_seed(args.seed)
    dataset_name = args.dataset_name
    if dataset_name == DatasetType.BITCOIN_ALPHA.name.lower():
        mat_path = os.path.join(BITCOIN_ALPHA_PATH, BITCOIN_ALPHA_NAME)
        TS = BITCOIN_ALPHA_TS
    elif dataset_name == DatasetType.BITCOIN_OTC.name.lower():
        mat_path = os.path.join(BITCOIN_OTC_PATH, BITCOIN_OTC_NAME)
        TS = BITCOIN_OTC_TS
    elif dataset_name == DatasetType.WIKI_GL.name.lower():
        mat_path = os.path.join(WIKI_GL_PATH, WIKI_GL_NAME)
        TS = WIKI_GL_TS
    elif dataset_name == DatasetType.DIGG.name.lower():
        mat_path = os.path.join(DIGG_PATH, DIGG_NAME)
        TS = DIGG_TS
    elif dataset_name == DatasetType.WIKI_EO.name.lower():
        mat_path = os.path.join(WIKI_EO_PATH, WIKI_EO_NAME)
        TS = WIKI_EO_TS
    elif dataset_name == DatasetType.PPIN.name.lower():
        mat_path = os.path.join(PPIN_PATH, PPIN_NAME)
        TS = PPIN_TS
    elif dataset_name == DatasetType.DIGG.name.lower():
        mat_path = os.path.join(DIGG_PATH, DIGG_NAME)
        TS = DIGG_TS
    elif dataset_name == DatasetType.LAST_FM.name.lower():
        mat_path = os.path.join(LAST_FM_PATH, LAST_FM_NAME)
        TS = LAST_FM_TS
    elif dataset_name == DatasetType.INTERNET.name.lower():
        mat_path = os.path.join(INTERNET_PATH, INTERNET_NAME)
        TS = INTERNET_TS
    elif dataset_name == DatasetType.IA_REALITY_CALL.name.lower():
        mat_path = os.path.join(IA_REALITY_CALL_PATH, IA_REALITY_CALL_NAME)
        TS = IA_REALITY_CALL_TS
    elif dataset_name == DatasetType.ENRON.name.lower():
        mat_path = os.path.join(ENRON_PATH, ENRON_NAME)
        TS = ENRON_TS
    elif dataset_name == DatasetType.WIKI.name.lower():
        mat_path = os.path.join(WIKI_PATH, WIKI_NAME)
        TS = WIKI_TS
    elif dataset_name == DatasetType.FB_MESSAGES.name.lower():
        mat_path = os.path.join(FB_MESSAGES_PATH, FB_MESSAGES_NAME)
        TS = FB_MESSAGES_TS
    elif dataset_name == DatasetType.DBLP.name.lower():
        mat_path = os.path.join(DBLP_PATH, DBLP_NAME)
        TS = DBLP_TS
    elif dataset_name == DatasetType.MATH.name.lower():
        mat_path = os.path.join(MATH_PATH, MATH_NAME)
        TS = MATH_TS
    elif dataset_name == DatasetType.CHESS.name.lower():
        mat_path = os.path.join(CHESS_PATH, CHESS_NAME)
        TS = CHESS_TS
    
    else:
        raise Exception('Invalid DataSet')

    val_steps = int(TS * VAL_RATE)
    test_steps = int(TS * TEST_RATE)
    train_steps = TS - val_steps - test_steps

    # Load stuff from mat file
    saved_content = sio.loadmat(mat_path)
    T = np.max(saved_content["tensor_idx"][:, 0]) + 1
    N1 = np.max(saved_content["tensor_idx"][:, 1]) + 1
    N2 = np.max(saved_content["tensor_idx"][:, 2]) + 1
    N = max(N1, N2)
    A_sz = torch.Size([T, N, N])
    train_sz = torch.Size([train_steps, N, N])

    A = func_get_floatTensor(saved_content, 'A', A_sz, device)
    labels, neg_adj = negative_sample(A, BETA, device)

    train = func_get_floatTensor(saved_content, 'train', train_sz, device)
    val = func_get_floatTensor(saved_content, 'val', train_sz, device)
    test = func_get_floatTensor(saved_content, 'test', train_sz, device)

    A_train = split_sparse_tensor_by_time(train, train_steps, N, device)
    A_val = split_sparse_tensor_by_time(val, train_steps, N, device)
    A_test = split_sparse_tensor_by_time(test, train_steps, N, device)

    graph_stats_train = compute_graph_stats(A_train, N, device)
    graph_stats_val = compute_graph_stats(A_val, N, device)
    graph_stats_test = compute_graph_stats(A_test, N, device)

    return TS, labels, A_train, A_val, A_test, N, graph_stats_train, graph_stats_val, graph_stats_test


def compute_graph_stats(A_list, N, device, L_window=5):
    """Precompute graph statistics used by observability features and LoRA prior gates."""
    T = len(A_list)
    degrees = []
    gaps = []
    num_edges = []
    active_ratios = []

    last_seen = torch.full((N,), -1, dtype=torch.long, device=device)

    for t in range(T):
        A_t = A_list[t]
        idx = A_t._indices()

        # degree
        deg = torch.zeros(N, dtype=torch.float, device=device)
        if idx.numel() > 0:
            src = idx[0]
            deg.index_add_(0, src, torch.ones(src.size(0), dtype=torch.float, device=device))
        degrees.append(deg)

        # gap
        gap = torch.full((N,), float(t + 1), dtype=torch.float, device=device)
        if idx.numel() > 0:
            active_nodes = src.unique()
            prev_seen = last_seen[active_nodes]
            gap[active_nodes] = torch.where(
                prev_seen >= 0,
                (t - prev_seen).to(torch.float),
                torch.full_like(prev_seen, float(t + 1), dtype=torch.float),
            )
            last_seen[active_nodes] = t
        gaps.append(gap)

        # num_edges
        num_edges.append(int(A_t._nnz()))

        # active_ratio
        if idx.numel() > 0:
            active_nodes = src.unique().numel()
        else:
            active_nodes = 0
        active_ratios.append(active_nodes / N)

    degrees_tensor = torch.stack(degrees, dim=0)
    prefix = torch.cat([torch.zeros_like(degrees_tensor[:1]), degrees_tensor.cumsum(dim=0)], dim=0)
    end = torch.arange(1, T + 1, dtype=torch.long, device=device)
    start = torch.clamp(end - L_window, min=0)
    recent_counts = list((prefix[end] - prefix[start]).unbind(0))

    return {
        "degrees": degrees,
        "gaps": gaps,
        "num_edges": num_edges,
        "active_ratios": active_ratios,
        "recent_counts": recent_counts,
    }


def negative_sample(A, beta, device):
    tensor_idx = torch.zeros([3, A._nnz() + A._nnz() * beta], dtype=torch.long, device=device)
    tensor_val = torch.zeros([A._nnz() + A._nnz() * beta], dtype=torch.float, device=device)
    next_idx = 0

    T, N, _ = A.size()
    neg_adj = []
    for i in range(T):
        num_samples = int(A[i]._nnz() * beta)
        if num_samples == 0:
            continue
        edge_index = A[i]._indices().to(device)
        negative_samples = generate_negative_samples(N, edge_index, num_samples)
        negative_samples = torch.tensor(negative_samples, dtype=torch.long, device=device).t()
        values = torch.zeros(num_samples, dtype=torch.float, device=device)
        neg_A = torch.sparse_coo_tensor(
            negative_samples,
            values,
            torch.Size([N, N]),
            dtype=torch.double
        ).to(device)
        neg_adj.append(neg_A)

        combined_A = A[i].to(device) + neg_A
        num = combined_A._nnz()
        tensor_idx[1:3, next_idx:next_idx + num] = combined_A._indices()
        tensor_idx[0, next_idx:next_idx + num] = i
        tensor_val[next_idx:next_idx + num] = combined_A._values()

        next_idx += num

    return torch.sparse_coo_tensor(tensor_idx, tensor_val, torch.Size([T, N, N]), device=device), neg_adj


def generate_negative_samples(num_nodes, edge_index, num_samples=None):
    existing_edges = set((edge_index[0][i].item(), edge_index[1][i].item()) for i in range(edge_index.shape[1]))

    if num_samples is None:
        num_samples = edge_index.shape[1]

    negative_samples = set()
    while len(negative_samples) < num_samples:
        u, v = random.sample(range(num_nodes), 2)
        if (u, v) not in existing_edges and (v, u) not in existing_edges and u != v:
            negative_samples.add((u, v))

    return list(negative_samples)
