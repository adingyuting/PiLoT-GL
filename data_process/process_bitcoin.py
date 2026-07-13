import math  # 标准库：数学工具（这里用于 floor 等）

from constant import VAL_RATE, TEST_RATE, BITCOIN_OTC_TS, BITCOIN_ALPHA_TS  # 导入常量：验证/测试比例、时间片数等
from process import *  # 导入数据处理工具函数（如 split_data / func_make_symmetric / func_edge_life / func_laplacian_transformation / save_file）

# =========================
# 参数设置区
# =========================
dataset = 'bitcoin_alpha'  # 数据集选择器：'bitcoin_alpha' 或 'bitcoin_otc'
print(dataset)  # 打印当前选择的数据集名称

DATA_DIR_PATH = os.path.join(os.path.dirname(__file__), '../data')  # 数据根目录：当前脚本目录/../data

# =========================
# bitcoin alpha 数据集路径设置
# =========================
ALPHA_PATH = os.path.join(DATA_DIR_PATH, 'bitcoin_alpha')  # alpha 数据目录
ALPHA_NAME = 'bitcoin_alpha.mat'  # alpha 输出的 mat 文件名

# =========================
# bitcoin otc 数据集路径设置
# =========================
OTC_PATH = os.path.join(DATA_DIR_PATH, 'bitcoin_otc')  # otc 数据目录
OTC_NAME = 'bitcoin_otc.mat'  # otc 输出的 mat 文件名
TIME_DIM = 3  # CSV 中时间戳所在列的索引（从0开始）：第4列为时间戳

# =========================
# 图预处理开关
# =========================
EDGE_LIFE = False  # 是否启用“边生命周期”机制：把历史窗口内出现的边延续到当前时间片
EDGE_LIFE_WINDOW = 10  # 边生命周期窗口长度（单位：时间片）
MAKE_SYMMETRIC = False  # 是否对每个时间片的邻接矩阵做对称化（A := A + A^T）


def preprocess_data():  # 主预处理流程：读取CSV→按时间离散→构造稀疏张量→切分→归一化→保存
    # -------------------------
    # 1) 根据 dataset 选择数据源与输出文件名
    # -------------------------
    if dataset == 'bitcoin_alpha':  # 如果选择 alpha 数据集
        data = np.loadtxt(
            os.path.join(ALPHA_PATH, 'soc-sign-bitcoinalpha.csv'),
            delimiter=','
        )  # 读取 alpha 的 CSV（每行一般是 src, dst, label, time）
        save_file_location = ALPHA_PATH  # 输出目录：alpha 目录
        save_file_name = ALPHA_NAME  # 输出文件名：bitcoin_alpha.mat
        TS = BITCOIN_ALPHA_TS  # 时间片数量（预定义常量），这个是从baseline中获得的
    elif dataset == 'bitcoin_otc':  # 如果选择 otc 数据集
        data = np.loadtxt(
            os.path.join(OTC_PATH, 'soc-sign-bitcoinotc.csv'),
            delimiter=','
        )  # 读取 otc 的 CSV
        save_file_location = OTC_PATH  # 输出目录：otc 目录
        save_file_name = OTC_NAME  # 输出文件名：bitcoin_otc.mat
        TS = BITCOIN_OTC_TS  # 时间片数量（预定义常量）
    else:  # 其他字符串都视为非法
        raise Exception('Invalid dataset')  # 直接报错终止

    # -------------------------
    # 2) 计算时间离散化参数：最小/最大时间戳、每个时间片宽度
    # -------------------------
    max_time = max(data[:, TIME_DIM])  # 取时间戳列的最大值
    min_time = min(data[:, TIME_DIM])  # 取时间戳列的最小值
    time_delta = math.floor((max_time - min_time) / TS)  # 用总跨度/时间片数得到每个时间片宽度（向下取整）

    # -------------------------
    # 3) 转 torch，并确定节点数 N
    # -------------------------
    data = torch.tensor(data)  # numpy → torch 张量（后续用 torch 处理）
    N = int(max(max(data[:, 0]), max(data[:, 1])))  # 节点数：取src列与dst列最大ID（这里默认节点ID从1开始）

    # -------------------------
    # 4) 过滤：只保留落在 TS 个时间片覆盖范围内的边
    # -------------------------
    data_idx = data[:, TIME_DIM] <= min_time + time_delta * TS  # 只取时间戳不超过“最后一个bin上界”的样本
    data = data[data_idx]  # 应用过滤

    # -------------------------
    # 5) 为构造三维稀疏张量准备 index/value/label 容器
    #    tensor_idx: 每条边的 (t, src, dst)
    #    tensor_val: 邻接取值（这里全设为1，表示有边）
    #    tensor_labels: 边的符号/评分标签（来自CSV第三列）
    # -------------------------
    tensor_idx = torch.zeros([data.size()[0], 3], dtype=torch.long)  # 初始化三维索引矩阵，形状[E,3]，保存所有张量所有元素的索引
    tensor_val = torch.ones([data.size()[0]], dtype=torch.double)  # 初始化邻接值为1，形状[E]，保存所有张量所有元素的值
    tensor_labels = torch.zeros([data.size()[0]], dtype=torch.double)  # 初始化标签容器，形状[E]，

    # -------------------------
    # 6) 将连续时间戳映射到离散时间片 t=0..TS-1 
    # -------------------------
    start = min_time  # 当前时间片的左边界（初始为最小时间戳）
    for t in range(TS):  # 遍历每个时间片
        end = start + time_delta  # 当前时间片的右边界
        if t == TS - 1:  # 最后一个时间片：包含右边界，避免丢数据
            idx = (data[:, TIME_DIM] >= start) & (data[:, TIME_DIM] <= end)  # 最后一段用 <=
        else:  # 非最后时间片：右边界开区间，避免重复归属
            idx = (data[:, TIME_DIM] >= start) & (data[:, TIME_DIM] < end)  # 常规段用 <
        start = end  # 时间窗口右移到下一个时间片

        # 将该时间片内的边写入 tensor_idx：
        # data[:,0:2] 是 src,dst，原始是 1-based，因此减1变成 0-based
        tensor_idx[idx, 1:3] = (data[idx, 0:2] - 1).type('torch.LongTensor')  # 写入(src,dst)并做0基化
        tensor_idx[idx, 0] = t  # 写入时间片编号 t
        tensor_labels[idx] = data[idx, 2].type('torch.DoubleTensor')  # 写入边标签（第三列：例如信任评分/正负符号）

    # -------------------------
    # 7) 构造三维稀疏邻接张量 A：shape = [TS, N, N]
    #    indices 需要是 [3, E]，所以 transpose(1,0)
    # -------------------------
    A = torch.sparse.DoubleTensor(
        tensor_idx.transpose(1, 0),  # [3, E]：time, src, dst
        tensor_val,                  # [E]：全1
        torch.Size([TS, N, N])       # 三维尺寸
    ).coalesce()  # 合并重复索引并排序

    # -------------------------
    # 8) 二值化邻接：即使原来有重复边/多次出现，也将边权统一置为1
    # -------------------------
    A = torch.sparse.DoubleTensor(
        A._indices(),                           # 保持 indices 不变
        torch.ones(A._values().shape),          # values 全改为 1
        torch.Size([TS, N, N])                  # 尺寸不变
    )  # 得到二值邻接张量（注意：这里未显式coalesce，但indices来自coalesce后的A，一般是规整的）

    # -------------------------
    # 9) 构造标签张量 labels_weight：shape=[TS,N,N]，在对应边位置存放标签值
    # -------------------------
    labels_weight = torch.sparse.DoubleTensor(
        tensor_idx.transpose(1, 0),  # 标签对应同样的 (t, src, dst) 索引
        tensor_labels,               # 标签取值（可能为正负/评分）
        torch.Size([TS, N, N]),      # 与A同形状
    ).coalesce()  # 合并重复索引并排序（重要）

    # -------------------------
    # 10) 按时间片划分 train/val/test（只在时间维切）
    # -------------------------
    val_samples = int(TS * VAL_RATE)  # 验证集时间片数量
    test_samples = int(TS * TEST_RATE)  # 测试集时间片数量
    T = TS - val_samples - test_samples  # 训练集时间片数量

    # -------------------------
    # 11) 切分邻接张量：分别取不同时间段
    # -------------------------
    A_train = split_data(A, N, T, 0, T)  # 训练：时间片[0, T)
    A_val = split_data(A, N, T, val_samples, T + val_samples)  # 验证：按给定区间取一段
    A_test = split_data(A, N, T, val_samples + test_samples, TS)  # 测试：最后一段

    # -------------------------
    # 12) （可选）对称化：让每个时间片的邻接矩阵 A := A + A^T
    # -------------------------
    print('make_sym...')  # 打印当前步骤
    if MAKE_SYMMETRIC:  # 若开启对称化
        A_train_sym = func_make_symmetric(A_train, N, T)  # 训练集对称化
        A_val_sym = func_make_symmetric(A_val, N, T)  # 验证集对称化
        A_test_sym = func_make_symmetric(A_test, N, T)  # 测试集对称化
    else:  # 不做对称化则透传
        A_train_sym = A_train
        A_val_sym = A_val
        A_test_sym = A_test

    # -------------------------
    # 13) （可选）edge life：把过去窗口内出现过的边延续到当前时间片
    # -------------------------
    print('edge_life...')  # 打印当前步骤
    if EDGE_LIFE:  # 若开启 edge life
        A_train_sym_life = func_edge_life(A_train_sym, N, T, EDGE_LIFE_WINDOW)  # train应用edge life
        A_val_sym_life = func_edge_life(A_val_sym, N, T, EDGE_LIFE_WINDOW)  # val应用edge life
        A_test_sym_life = func_edge_life(A_test_sym, N, T, EDGE_LIFE_WINDOW)  # test应用edge life
    else:  # 不做 edge life 则透传
        A_train_sym_life = A_train_sym
        A_val_sym_life = A_val_sym
        A_test_sym_life = A_test_sym

    # -------------------------
    # 14) 拉普拉斯/度归一化：常见GCN预处理（加自环 + D^{-1/2} A D^{-1/2}）
    # -------------------------
    print('func_laplacian_trans...')  # 打印当前步骤
    A_train_sym_life_la = func_laplacian_transformation(A_train_sym_life, N, T)  # train归一化
    A_val_sym_life_la = func_laplacian_transformation(A_val_sym_life, N, T)  # val归一化
    A_test_sym_life_la = func_laplacian_transformation(A_test_sym_life, N, T)  # test归一化

    # -------------------------
    # 15) 赋值输出：最终用于模型训练/验证/测试的图张量
    # -------------------------
    train = A_train_sym_life_la  # 最终训练用图
    val = A_val_sym_life_la  # 最终验证用图
    test = A_test_sym_life_la  # 最终测试用图

    # -------------------------
    # 16) 保存：把原始索引/标签、切分张量、预处理张量一并存入mat文件
    # -------------------------
    print('store...')  # 打印保存步骤
    save_file(
        tensor_idx, tensor_labels,      # 原始边索引与边标签（按边列表）
        A, A_train, A_val, A_test,      # 原始邻接与切分后的邻接
        train, val, test,               # 预处理后的 train/val/test 图张量
        save_file_location, save_file_name  # 输出路径与文件名
    )  # 执行保存


preprocess_data()  # 运行预处理（脚本入口）
