import os  # 标准库：与操作系统交互（路径拼接、文件操作等）

import numpy as np  # 数值计算：数组/矩阵
import scipy.io as sio  # .mat 文件读写（MATLAB 格式）
import torch  # PyTorch：张量与稀疏张量运算
from sklearn.utils import shuffle  # sklearn：用于随机打乱序列

EDGE_LIFE = False  # 是否启用“边生命周期（edge life）”机制：把历史窗口内的边带到当前时刻
EDGE_LIFE_WINDOW = 10  # edge life 的时间窗口长度（例如最近10个时间片）
MAKE_SYMMETRIC = False  # 是否将每个时间片的邻接矩阵强制对称化（A := A + A^T）


def func_make_symmetric(sparse_tensor, N, TS):  # 将三维稀疏张量按时间片做对称化（TS个[N,N]邻接矩阵）
    count = 0  # 统计每个时间片对称化后的非零数量（仅统计，不影响后续逻辑）
    tensor_idx = torch.LongTensor([])  # 用于累积输出稀疏张量的 indices（形状最终是[3, nnz]）
    tensor_val = torch.DoubleTensor([]).unsqueeze(1)  # 用于累积输出稀疏张量的 values（先做成列向量便于cat）
    A_idx = sparse_tensor._indices()  # 输入稀疏张量的坐标索引，形状[3, nnz]：第0行=time，第1行=i，第2行=j
    A_val = sparse_tensor._values()  # 输入稀疏张量的取值，形状[nnz]
    for i in range(TS):  # 遍历每个时间片 i
        """make each slice symmetric"""  # 说明：对每个时间片的[N,N]邻接矩阵做对称化
        idx = A_idx[0] == i  # 选择当前时间片 i 的所有非零条目（布尔掩码）
        mat = torch.sparse.DoubleTensor(
            A_idx[1:3, idx],          # 取出空间索引(i,j)，形状[2, nnz_i]
            A_val[idx],               # 取出对应取值，形状[nnz_i]
            torch.Size([N, N])        # 当前时间片邻接矩阵大小
        )  # 构造当前时间片的二维稀疏邻接矩阵 mat
        mat_t = mat.transpose(1, 0)  # 计算转置矩阵 A^T
        sym_mat = (mat + mat_t)  # 对称化：A + A^T（注意：若原本有(i,j)与(j,i)，可能叠加）
        count = count + sym_mat._nnz()  # 累加对称化后的非零元素数量
        vertices = sym_mat._indices().clone().detach()  # 取对称矩阵的空间坐标索引[2, nnz_sym]，并复制为独立张量
        time = torch.ones(sym_mat._nnz(), dtype=torch.long) * i  # 为每条边生成对应的时间索引（全是i），长度=nnz_sym
        time = time.unsqueeze(0)  # 变成[1, nnz_sym]，便于与空间索引拼接
        full = torch.cat((time, vertices), 0)  # 拼出三维索引[3, nnz_sym]：第0行=time，第1行=row，第2行=col
        tensor_idx = torch.cat((tensor_idx, full), 1)  # 把当前时间片的索引拼到全局索引后面
        tensor_val = torch.cat((tensor_val, sym_mat._values().unsqueeze(1)), 0)  # 把当前时间片取值拼到全局取值后面（列向量）
    tensor_val.squeeze_(1)  # 把[nnz,1]压成[nnz]
    A = torch.sparse.DoubleTensor(
        tensor_idx,                  # 形状[3, nnz_total]
        tensor_val,                  # 形状[nnz_total]
        torch.Size([TS, N, N])       # 三维稀疏张量大小：时间×节点×节点
    ).coalesce()  # coalesce：合并重复索引并排序（很重要，否则重复边会导致后续运算异常/低效）
    return A  # 返回对称化后的三维稀疏张量


def func_edge_life(A, N, TS, edge_life_window):  # edge life：把过去窗口内的边“带到”当前时间片
    """carry edges from previous slices into current slice"""  # 意思：将最近window内出现过的边，在当前时刻也视为存在
    A_new = A.clone()  # 克隆一个与A同shape的稀疏张量（结构先复制）
    A_new._values()[:] = 0  # 把克隆张量的所有values置0（准备重新累加）
    for t in range(TS):  # 遍历每个时间片 t
        idx = (
            (A._indices()[0] >= max(0, t - edge_life_window + 1))  # 时间索引 >= 窗口起点
            & (A._indices()[0] <= t)                               # 时间索引 <= 当前时刻
        )  # 选出“窗口内”的所有边条目
        block = torch.sparse_coo_tensor(
            A._indices()[0:3, idx],         # 取出窗口内边的(时间,row,col)索引
            A._values()[idx],               # 对应取值
            torch.Size([TS, N, N]),         # 仍然构造为全尺寸三维张量
            dtype=torch.double,             # double精度
        )  # 构造一个仅在窗口内有非零的稀疏张量 block
        block._indices()[0] = t  # 把这些边的时间索引全部改成当前时刻t（相当于把历史边搬运到当前切片）
        A_new = A_new + block  # 累加到输出张量（会把历史窗口内所有边叠加到t时刻）
    return A_new.coalesce()  # 合并重复索引、排序后返回


def func_laplacian_transformation(B, N, TS):  # 对每个时间片做“带自环 + 度归一化”的邻接变换（常用于GCN归一化）
    vertices = torch.LongTensor([range(N), range(N)])  # 构造二维单位矩阵对角线索引：[2, N]，即(0,0),(1,1),...,(N-1,N-1)
    tensor_idx = torch.LongTensor([])  # 用于累积三维单位张量I的索引
    tensor_val = torch.DoubleTensor([]).unsqueeze(1)  # 用于累积三维单位张量I的取值（列向量）
    for i in range(TS):  # 为每个时间片都加上单位矩阵（自环）
        time = torch.ones(N, dtype=torch.long) * i  # 当前时间片i对应的时间索引（长度N）
        time = time.unsqueeze(0)  # [1, N]
        full = torch.cat((time, vertices), 0)  # [3, N]：拼成(time, row, col)三维索引
        tensor_idx = torch.cat((tensor_idx, full), 1)  # 累加索引
        val = torch.ones(N, dtype=torch.double)  # 单位矩阵的对角元素取值全为1
        tensor_val = torch.cat((tensor_val, val.unsqueeze(1)), 0)  # 累加取值（列向量）
    tensor_val.squeeze_(1)  # [TS*N]
    I = torch.sparse_coo_tensor(
        tensor_idx, torch.tensor(tensor_val), torch.Size([TS, N, N]), dtype=torch.double
    )  # 构造三维单位张量I：每个时间片一个[N,N]单位矩阵
    C = B + I  # 加自环：C = B + I（B是输入邻接，C是带自环的邻接）

    tensor_idx = torch.LongTensor([])  # 重置，用于累积归一化后的C
    tensor_val = torch.DoubleTensor([]).unsqueeze(1)  # 重置，用于累积归一化后的values
    for k in range(TS):  # 逐时间片做归一化
        idx = C._indices()[0] == k  # 选出时间片k的所有边
        mat = torch.sparse_coo_tensor(
            C._indices()[1:3, idx],    # 空间索引(row,col)
            C._values()[idx],          # 对应权重
            torch.Size([N, N]),
            dtype=torch.double,
        )  # 得到二维稀疏邻接矩阵 mat（时间片k）
        vec = torch.ones([N, 1], dtype=torch.double)  # 全1向量，用于计算度
        degree = 1 / torch.sqrt(torch.sparse.mm(mat, vec))  # degree = (D^{-1/2})，其中 D = mat * 1
        index = torch.LongTensor(C._indices()[0:3, idx].size())  # 为该时间片的三维索引创建缓冲区，形状[3, nnz_k]
        index[0] = k  # 第0行是时间索引k
        index[1:3] = mat._indices()  # 第1/2行是二维mat的(row,col)索引
        values = mat._values()  # 取出该时间片的边权
        count = 0  # 遍历values的游标
        for i, j in index[1:3].transpose(1, 0):  # 逐条边遍历 (row=i, col=j)
            values[count] = values[count] * degree[i] * degree[j]  # 归一化：A_ij := D^{-1/2}_i * A_ij * D^{-1/2}_j
            count = count + 1  # 游标+1
        tensor_idx = torch.cat((tensor_idx, index), 1)  # 累加该时间片归一化后的索引
        tensor_val = torch.cat((tensor_val, values.unsqueeze(1)), 0)  # 累加该时间片归一化后的values（列向量）
    tensor_val.squeeze_(1)  # 压成一维
    C = torch.sparse_coo_tensor(
        tensor_idx, tensor_val, torch.Size([TS, N, N]), dtype=torch.double
    )  # 重新构造三维稀疏张量C（归一化后）
    return C.coalesce()  # 合并重复索引并排序，返回


def get_random_idx(num_all, num_ones, random_state=2024):  # 生成长度=num_all的随机布尔掩码，其中True数量=num_ones
    no_true = np.ones(num_ones) == 1  # 生成num_ones个True
    no_false = np.zeros(num_all - num_ones) == 1  # 生成剩余个False
    temp = list(no_true) + list(no_false)  # 拼成一个True/False列表
    idx = shuffle(torch.tensor(temp), random_state=random_state)  # 打乱顺序得到随机掩码（可复现）
    return idx  # 返回布尔张量（mask）


def get_dataset(A, idx):  # 按掩码idx将稀疏张量A拆成两部分：remain(未选中)与sub(选中)
    sz = A.size()  # 保存原张量尺寸（TS,N,N）
    not_idx = idx == False  # 取反掩码：未选中部分

    index = torch.LongTensor(A._indices()[0:3, idx].size())  # 为“选中部分”的索引分配空间（形状[3, nnz_sub]）
    index[0:3] = A._indices()[0:3, idx]  # 拷贝选中条目的三维索引
    values = A._values()[idx]  # 拷贝选中条目的取值
    sub = torch.sparse_coo_tensor(index, values, sz)  # 构造选中子张量 sub（同shape）

    remain_index = torch.LongTensor(A._indices()[0:3, not_idx].size())  # 为“剩余部分”的索引分配空间
    remain_index[0:3] = A._indices()[0:3, not_idx]  # 拷贝剩余条目的索引
    remain_values = A._values()[not_idx]  # 拷贝剩余条目的取值
    remain = torch.sparse_coo_tensor(remain_index, remain_values, sz)  # 构造剩余子张量 remain（同shape）

    return remain.coalesce(), sub.coalesce()  # coalesce后返回（去重/排序）


def split_data(A, N, T, start, end):  # 从三维稀疏张量A按时间区间[start,end)截取子张量，并把时间轴平移到[0,T)
    assert (end - start) == T  # 保证截取长度与T一致
    idx = (A._indices()[0] >= start) & (A._indices()[0] < end)  # 选出时间索引在区间内的条目
    index = torch.LongTensor(A._indices()[0:3, idx].size())  # 分配索引缓冲区
    index[0:3] = A._indices()[0:3, idx]  # 拷贝索引
    index[0] = index[0] - start  # 时间维度减start，使得新时间从0开始
    values = A._values()[idx]  # 拷贝对应取值
    sub = torch.sparse_coo_tensor(index, values, torch.Size([T, N, N]), dtype=torch.double)  # 构造截取后的三维张量
    return sub.coalesce()  # 返回整理后的子张量


def get_node_to_index(edge):  # 将原始节点ID（可能不连续）映射到[0, num_nodes-1]的连续索引
    edge = edge.tolist()  # 可能是tensor/ndarray，转成Python list便于处理
    unique_nodes = list(set([src for src, _ in edge] + [dst for _, dst in edge]))  # 收集所有出现过的节点（src+dst去重）
    num_nodes = len(unique_nodes)  # 节点数
    node_to_index = {node: idx for idx, node in enumerate(unique_nodes)}  # 生成映射：原ID -> 连续索引
    return node_to_index, num_nodes  # 返回映射字典与节点总数


def get_adj_idx(node_to_index, edge):  # 将边列表(原始ID)转换为(连续索引)的边对
    edge = edge.tolist()  # 转list
    rows, cols = [], []  # 分别保存源点与终点的连续索引
    for src, dst in edge:  # 遍历每条边
        rows.append(node_to_index[src])  # 源点映射
        cols.append(node_to_index[dst])  # 终点映射
    rows_tensor = torch.tensor(rows, dtype=torch.long)  # 转为tensor
    cols_tensor = torch.tensor(cols, dtype=torch.long)  # 转为tensor
    return torch.stack((rows_tensor, cols_tensor), dim=1)  # 形状[E,2]，每行是(row, col)


def save_file(tensor_idx, tensor_labels, A, A_train, A_val, A_test, train, val, test, path, file_name):  # 保存为.mat便于复现/与MATLAB交互
    A_idx = A._indices()  # 原始全量邻接的索引
    A_vals = A._values()  # 原始全量邻接的取值

    A_train_idx = A_train._indices()  # 原始训练切片索引
    A_train_vals = A_train._values()  # 原始训练切片取值

    A_val_idx = A_val._indices()  # 原始验证切片索引
    A_val_vals = A_val._values()  # 原始验证切片取值

    A_test_idx = A_test._indices()  # 原始测试切片索引
    A_test_vals = A_test._values()  # 原始测试切片取值

    train_idx = train._indices()  # 预处理后的训练张量索引（对称/edge_life/laplacian等之后）
    train_vals = train._values()  # 预处理后的训练张量取值

    val_idx = val._indices()  # 预处理后的验证索引
    val_vals = val._values()  # 预处理后的验证取值

    test_idx = test._indices()  # 预处理后的测试索引
    test_vals = test._values()  # 预处理后的测试取值

    sio.savemat(os.path.join(path, file_name), {  # 保存为mat：键名->numpy数组
        'tensor_idx': np.array(tensor_idx),  # 原始样本索引（外部传入，可能是边/样本索引）
        'tensor_labels': np.array(tensor_labels),  # 原始标签（外部传入）

        'A_idx': np.array(A_idx),  # 全量邻接索引（time,row,col）
        'A_vals': np.array(A_vals),  # 全量邻接取值

        'A_train_idx': np.array(A_train_idx),  # train切片索引
        'A_train_vals': np.array(A_train_vals),  # train切片取值

        'A_val_idx': np.array(A_val_idx),  # val切片索引
        'A_val_vals': np.array(A_val_vals),  # val切片取值

        'A_test_idx': np.array(A_test_idx),  # test切片索引
        'A_test_vals': np.array(A_test_vals),  # test切片取值

        'train_idx': np.array(train_idx),  # 预处理后的train索引
        'train_vals': np.array(train_vals),  # 预处理后的train取值

        'test_idx': np.array(test_idx),  # 预处理后的test索引
        'test_vals': np.array(test_vals),  # 预处理后的test取值

        'val_idx': np.array(val_idx),  # 预处理后的val索引
        'val_vals': np.array(val_vals),  # 预处理后的val取值
    })


def split_data_link(TS, val_rate, test_rate, A, N):  # 按时间维度比例划分训练/验证/测试（切片划分）
    val_samples = int(TS * val_rate)  # 验证集时间片数量
    test_samples = int(TS * test_rate)  # 测试集时间片数量
    T = TS - val_samples - test_samples  # 训练集时间片数量（剩余的）

    # 注意：这里的“起止时间”写法决定了切片范围；逻辑是：train=[0,T)
    A_train = split_data(A, N, T, 0, T)  # 训练：从0到T
    A_val = split_data(A, N, T, val_samples, T + val_samples)  # 验证：按给定区间切
    A_test = split_data(A, N, T, val_samples + test_samples, TS)  # 测试：最后一段

    return A_train, A_val, A_test, T  # 返回切分后的张量与训练长度T


def pre_process(A_train, A_val, A_test, N, T):  # 数据预处理流水线：可选对称化、可选edge_life、必做拉普拉斯归一化
    print('make_sym...')  # 日志：对称化步骤
    if MAKE_SYMMETRIC:  # 若开启对称化
        A_train_sym = func_make_symmetric(A_train, N, T)  # train对称化
        A_val_sym = func_make_symmetric(A_val, N, T)  # val对称化
        A_test_sym = func_make_symmetric(A_test, N, T)  # test对称化
    else:  # 不对称化则直接透传
        A_train_sym = A_train
        A_val_sym = A_val
        A_test_sym = A_test

    print('edge_life...')  # 日志：edge_life步骤
    if EDGE_LIFE:  # 若开启edge_life
        A_train_sym_life = func_edge_life(A_train_sym, N, T, EDGE_LIFE_WINDOW)  # train做edge_life
        A_val_sym_life = func_edge_life(A_val_sym, N, T, EDGE_LIFE_WINDOW)  # val做edge_life
        A_test_sym_life = func_edge_life(A_test_sym, N, T, EDGE_LIFE_WINDOW)  # test做edge_life
    else:  # 不开启则直接透传
        A_train_sym_life = A_train_sym
        A_val_sym_life = A_val_sym
        A_test_sym_life = A_test_sym

    print('func_laplacian_trans...')  # 日志：拉普拉斯/归一化步骤
    A_train_sym_life_la = func_laplacian_transformation(A_train_sym_life, N, T)  # train做归一化
    A_val_sym_life_la = func_laplacian_transformation(A_val_sym_life, N, T)  # val做归一化
    A_test_sym_life_la = func_laplacian_transformation(A_test_sym_life, N, T)  # test做归一化

    train = A_train_sym_life_la  # 输出train
    val = A_val_sym_life_la  # 输出val
    test = A_test_sym_life_la  # 输出test

    return train, val, test  # 返回预处理后的三份张量


def process_tab(filename, save_filename):  # 工具函数：把文本文件中的tab替换为单个空格
    with open(filename, 'r') as file:  # 以读模式打开原文件
        content = file.read()  # 读取全部内容为字符串

    processed_content = content.replace('\t', ' ')  # 将所有制表符\t替换为空格
    print(processed_content)  # 打印替换后的内容（调试/检查）

    with open(save_filename, 'w') as file:  # 以写模式打开输出文件
        file.write(processed_content)  # 写入替换后的内容
