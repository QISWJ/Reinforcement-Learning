import numpy as np

from basic import *
import collections

from itertools import combinations

def build_edge_as_node_edge_index(mesh: om.PolyMesh):
    """
    输入：
      - mesh: OpenMesh PolyMesh（全四边形网格也适用）
      - edge_feature_list: 二维 list/array，edge_feature_list[eid] 是第 eid 条边的特征向量

    输出：
      - edge_index: numpy.ndarray, shape = (2, M)
        新图：节点=原网格边；若两条原边共享一个原顶点，则新图中连接。
        返回的是“无向图的双向表示”（a<->b 两条有向边都给出）。

    说明：
      - 新图的节点编号 = 原网格 edge idx（即 eh.idx()）
      - 边特征 -> 节点特征（你已有 edge_feature_list，即节点特征矩阵）
    """

    # 用集合去重存无向边 (min,max)
    undirected = set()

    # 遍历每个原顶点，把其 incident edges 两两相连
    for vh in mesh.vertices():
        # OpenMesh Python 通常支持 mesh.ve(vh) 来遍历与顶点相邻的边
        incident = [eh.idx() for eh in mesh.ve(vh)]

        # 顶点度<2不产生连接
        if len(incident) < 2:
            continue

        # 任意两条共享该顶点的边，在新图里都相连（形成 clique）
        for a, b in combinations(incident, 2):
            if a == b:
                continue
            if a < b:
                undirected.add((a, b))
            else:
                undirected.add((b, a))

    # 转成 edge_index（双向）
    src = []
    dst = []
    for a, b in undirected:
        src.append(a); dst.append(b)
        src.append(b); dst.append(a)

    edge_index = np.asarray([src, dst], dtype=np.int64)
    return edge_index




# 获得指定边的周围拓扑特征
def calculate_edge_score(mesh, eh, ver_valence, ver_valence_diff):
    if mesh.is_boundary(eh):

        heh = mesh.halfedge_handle(eh, 0)
        heh_ = mesh.opposite_halfedge_handle(heh)
        if not mesh.is_boundary(heh):
            heh1 = heh
        else:
            heh1 = heh_
        v_1 = mesh.to_vertex_handle(heh1)
        heh2 = mesh.opposite_halfedge_handle(heh1)
        heh3 = mesh.next_halfedge_handle(heh1)
        v_2 = mesh.to_vertex_handle(heh3)
        heh4 = mesh.next_halfedge_handle(heh3)
        v_3 = mesh.to_vertex_handle(heh4)
        heh5 = mesh.next_halfedge_handle(heh4)
        v_4 = mesh.to_vertex_handle(heh5)
        vh_1_valence = ver_valence[v_1.idx()]
        vh_2_valence = ver_valence[v_2.idx()]
        vh_3_valence = ver_valence[v_3.idx()]
        vh_4_valence = ver_valence[v_4.idx()]

        vh_1_valence_diff = ver_valence_diff[v_1.idx()]
        vh_2_valence_diff = ver_valence_diff[v_2.idx()]
        vh_3_valence_diff = ver_valence_diff[v_3.idx()]
        vh_4_valence_diff = ver_valence_diff[v_4.idx()]
        vh_7_idx = halfedge_vh_score_(mesh, heh1, bool_=True)
        if vh_7_idx == -1:
            vh_7_valence = 4
            vh_7_valence_diff = 0
        elif vh_7_idx == -2:
            vh_7_valence = 0
            vh_7_valence_diff = 0
        else:
            vh_7_valence = ver_valence[vh_7_idx]
            vh_7_valence_diff = ver_valence_diff[vh_7_idx]
        vh_8_idx = halfedge_vh_score_(mesh, heh2, bool_=True)
        if vh_8_idx == -1:
            vh_8_valence = 4
            vh_8_valence_diff = 0
        elif vh_8_idx == -2:
            vh_8_valence = 0
            vh_8_valence_diff = 0
        else:
            vh_8_valence = ver_valence[vh_8_idx]
            vh_8_valence_diff = ver_valence_diff[vh_8_idx]

        if heh1 == heh:
            tuopu_list = [vh_1_valence, vh_2_valence, vh_3_valence, vh_4_valence, 0, 0,
                          vh_7_valence, vh_8_valence, vh_1_valence_diff, vh_2_valence_diff,
                          vh_3_valence_diff, vh_4_valence_diff, 0, 0,
                          vh_7_valence_diff, vh_8_valence_diff]
        else:
            tuopu_list = [vh_4_valence, 0, 0, vh_1_valence, vh_2_valence, vh_3_valence,
                          vh_8_valence, vh_7_valence, vh_4_valence_diff, 0,
                          0, vh_1_valence_diff, vh_2_valence_diff, vh_3_valence_diff,
                          vh_8_valence_diff, vh_7_valence_diff]

    else:

        heh1 = mesh.halfedge_handle(eh, 0)
        heh2 = mesh.opposite_halfedge_handle(heh1)

        v_1 = mesh.to_vertex_handle(heh1)
        heh3 = mesh.next_halfedge_handle(heh1)
        v_2 = mesh.to_vertex_handle(heh3)
        heh4 = mesh.next_halfedge_handle(heh3)
        v_3 = mesh.to_vertex_handle(heh4)
        heh5 = mesh.next_halfedge_handle(heh4)
        v_4 = mesh.to_vertex_handle(heh5)
        heh6 = mesh.next_halfedge_handle(heh2)
        v_5 = mesh.to_vertex_handle(heh6)
        heh7 = mesh.next_halfedge_handle(heh6)
        v_6 = mesh.to_vertex_handle(heh7)

        vh_1_valence = ver_valence[v_1.idx()]
        vh_2_valence = ver_valence[v_2.idx()]
        vh_3_valence = ver_valence[v_3.idx()]
        vh_4_valence = ver_valence[v_4.idx()]
        vh_5_valence = ver_valence[v_5.idx()]
        vh_6_valence = ver_valence[v_6.idx()]

        vh_1_valence_diff = ver_valence_diff[v_1.idx()]
        vh_2_valence_diff = ver_valence_diff[v_2.idx()]
        vh_3_valence_diff = ver_valence_diff[v_3.idx()]
        vh_4_valence_diff = ver_valence_diff[v_4.idx()]
        vh_5_valence_diff = ver_valence_diff[v_5.idx()]
        vh_6_valence_diff = ver_valence_diff[v_6.idx()]

        vh_7_idx = halfedge_vh_score_(mesh, heh1, bool_=True)
        if vh_7_idx == -1:
            vh_7_valence = 4
            vh_7_valence_diff = 0
        elif vh_7_idx == -2:
            vh_7_valence = 0
            vh_7_valence_diff = 0
        else:
            vh_7_valence = ver_valence[vh_7_idx]
            vh_7_valence_diff = ver_valence_diff[vh_7_idx]
        vh_8_idx = halfedge_vh_score_(mesh, heh2, bool_=True)
        if vh_8_idx == -1:
            vh_8_valence = 4
            vh_8_valence_diff = 0
        elif vh_8_idx == -2:
            vh_8_valence = 0
            vh_8_valence_diff = 0
        else:
            vh_8_valence = ver_valence[vh_8_idx]
            vh_8_valence_diff = ver_valence_diff[vh_8_idx]

        tuopu_list = [vh_1_valence, vh_2_valence, vh_3_valence, vh_4_valence, vh_5_valence, vh_6_valence,
                      vh_7_valence, vh_8_valence, vh_1_valence_diff, vh_2_valence_diff,
                      vh_3_valence_diff, vh_4_valence_diff, vh_5_valence_diff, vh_6_valence_diff,
                      vh_7_valence_diff, vh_8_valence_diff]

    return tuopu_list




#获得边的特征向量（23维）
def get_edge_feature(mesh, ver_valence, ver_valence_diff):
    edge_feature_list = []
    for i in range(mesh.n_edges()):
        eh = mesh.edge_handle(i)

        topu_list = calculate_edge_score(mesh, eh, ver_valence, ver_valence_diff)
        a_1, a_2, a_3, a_4 = Calculate_side_length_ratio(mesh, eh)
        if not mesh.is_boundary(eh):

            heh1 = mesh.halfedge_handle(eh, 0)
            fh1 = mesh.face_handle(heh1)
            heh2 = mesh.halfedge_handle(eh, 1)
            fh2 = mesh.face_handle(heh2)

            v_1 = mesh.to_vertex_handle(heh1)
            heh3 = mesh.next_halfedge_handle(heh1)
            v_2 = mesh.to_vertex_handle(heh3)
            heh4 = mesh.next_halfedge_handle(heh3)
            v_3 = mesh.to_vertex_handle(heh4)
            heh5 = mesh.next_halfedge_handle(heh4)
            v_4 = mesh.to_vertex_handle(heh5)
            heh6 = mesh.next_halfedge_handle(heh2)
            v_5 = mesh.to_vertex_handle(heh6)
            heh7 = mesh.next_halfedge_handle(heh6)
            v_6 = mesh.to_vertex_handle(heh7)

            a_5_1 = calculate_dihedral_angle(mesh, v_2, v_1, v_4, v_6)
            a_5_2 = calculate_dihedral_angle(mesh, v_3, v_1, v_4, v_5)
            a_5 = (a_5_1+a_5_2)/2



            ang1 = compute_face_vertex_inner_angle(mesh,fh1, v_4)
            ang2 = compute_face_vertex_inner_angle(mesh, fh1, v_1)
            ang3 = compute_face_vertex_inner_angle(mesh, fh1, v_2)
            ang4 = compute_face_vertex_inner_angle(mesh, fh1, v_3)
            ang5 = compute_face_vertex_inner_angle(mesh, fh2, v_4)
            ang6 = compute_face_vertex_inner_angle(mesh, fh2, v_1)
            ang7 = compute_face_vertex_inner_angle(mesh, fh2, v_6)
            ang8 = compute_face_vertex_inner_angle(mesh, fh2, v_5)
            if (ang1 == 0) or (ang2 == 0) or (ang3 == 0) or (ang4 == 0) :
                a_6 = 0
            else:
                a_6 = (ang3 + ang4) / (ang1 + ang2)

            if (ang5 == 0) or (ang6 == 0) or (ang7 == 0) or (ang8 == 0):
                a_7 = 0
            else:

                a_7 = (ang7 + ang8) / (ang5 + ang6)

        else:
            heh = mesh.halfedge_handle(eh, 0)
            heh_ = mesh.opposite_halfedge_handle(heh)
            if not mesh.is_boundary(heh):
                heh1 = heh
            else:
                heh1 = heh_

            v_1 = mesh.to_vertex_handle(heh1)
            fh1 = mesh.face_handle(heh1)
            heh3 = mesh.next_halfedge_handle(heh1)
            v_2 = mesh.to_vertex_handle(heh3)
            heh4 = mesh.next_halfedge_handle(heh3)
            v_3 = mesh.to_vertex_handle(heh4)
            heh5 = mesh.next_halfedge_handle(heh4)
            v_4 = mesh.to_vertex_handle(heh5)
            a_5 = 0



            ang1 = compute_face_vertex_inner_angle(mesh, fh1, v_4)
            ang2 = compute_face_vertex_inner_angle(mesh, fh1, v_1)
            ang3 = compute_face_vertex_inner_angle(mesh, fh1, v_2)
            ang4 = compute_face_vertex_inner_angle(mesh, fh1, v_3)


            if (ang1 == 0) or (ang2 == 0) or (ang3 == 0) or (ang4 == 0) :
                a_6 = 0
                a_7 = 0
            else:

                if heh1 == heh:
                    a_6 = (ang3 + ang4) / (ang1 + ang2)
                    a_7 = 0
                else:
                    a_6 = 0
                    a_7 = (ang3 + ang4) / (ang1 + ang2)

        jihe_list = [a_1, a_2, a_3, a_4, a_5, a_6, a_7]
        #yc_list = Prediction_edge_ring_node(mesh, eh)

        eh_arr = topu_list + jihe_list

        edge_feature_list.append(eh_arr)



    return edge_feature_list


# 执行单元塌缩总代码
def quad_collapse_act(mesh, eh, features_vertex, a):
    heh1 = mesh.halfedge_handle(eh, 0)
    heh2 = mesh.opposite_halfedge_handle(heh1)
    if a == 0:
        mesh, v5_new = quad_collapse_diagonal(mesh, heh1, features_vertex)
    else:
        mesh, v5_new = quad_collapse_diagonal(mesh, heh2, features_vertex)

    mesh = Preprocessing_act(mesh)
    ph = get_two_ring_neighbors_exclude_features(mesh, v5_new, features_vertex)
    mesh = laplacian_smooth_selected_vertices(mesh, ph)

    return mesh



#特征数据做标准化处理
def zscore_standardize_feature_list(feat_list, eps=1e-8, clip=None):
    """
    对二维特征 list 做 Z-score 标准化：x' = (x - mean) / std
    输入:
      feat_list: List[List[float]]，形状 [N, F]，feat_list[a] 是边 a 的特征向量
      eps: 防止 std=0
      clip: 可选，比如 clip=5.0 把结果裁剪到 [-5, 5]（抑制极端值）
    输出:
      standardized_list: 同形状 [N, F] 的二维 list
    """
    if not feat_list:
        return []

    N = len(feat_list)
    F = len(feat_list[0])

    # 检查每行维度一致
    for row in feat_list:
        if len(row) != F:
            raise ValueError("feat_list 每一行的特征维度必须一致")

    # 逐列计算 mean 和 std
    means = [0.0] * F
    for j in range(F):
        s = 0.0
        for i in range(N):
            s += feat_list[i][j]
        means[j] = s / N

    stds = [0.0] * F
    for j in range(F):
        var = 0.0
        mu = means[j]
        for i in range(N):
            d = feat_list[i][j] - mu
            var += d * d
        var /= N  # population variance（更常用于ML预处理）
        stds[j] = math.sqrt(var)

    # 标准化
    out = []
    for i in range(N):
        row_out = []
        for j in range(F):
            denom = stds[j] if stds[j] > eps else 1.0
            v = (feat_list[i][j] - means[j]) / denom
            if clip is not None:
                v = max(-clip, min(clip, v))
            row_out.append(v)
        out.append(row_out)

    return out


import math

def zscore_standardize_feature_list_1(feat_list, normalize_cols, eps=1e-8, clip=None):
    """
    对二维特征 list 的指定列做 Z-score 标准化：x' = (x - mean) / std
    未在 normalize_cols 中的列保持原值不变。

    参数:
      feat_list: List[List[float]]
          二维特征数组，形状 [N, F]
      normalize_cols: List[int]
          需要做标准化的列索引，如 [0, 1, 5, 8]
      eps: float
          防止 std=0
      clip: float or None
          可选，若设置如 clip=5.0，则把标准化结果裁剪到 [-5, 5]

    返回:
      out: List[List[float]]
          同形状 [N, F] 的二维 list，仅指定列被标准化
    """
    if not feat_list:
        return []

    N = len(feat_list)
    F = len(feat_list[0])

    # 检查每行维度一致
    for row in feat_list:
        if len(row) != F:
            raise ValueError("feat_list 每一行的特征维度必须一致")

    # 检查 normalize_cols 合法性，并去重排序
    if normalize_cols is None:
        normalize_cols = []

    normalize_cols = sorted(set(normalize_cols))

    for col in normalize_cols:
        if col < 0 or col >= F:
            raise ValueError(f"normalize_cols 中存在非法列索引: {col}, 合法范围应为 [0, {F-1}]")

    # 先复制原数据，未标准化列保持原值
    out = [row[:] for row in feat_list]

    # 仅对指定列计算 mean/std 并标准化
    for j in normalize_cols:
        # 计算均值
        s = 0.0
        for i in range(N):
            s += feat_list[i][j]
        mu = s / N

        # 计算标准差
        var = 0.0
        for i in range(N):
            d = feat_list[i][j] - mu
            var += d * d
        var /= N
        std = math.sqrt(var)

        denom = std if std > eps else 1.0

        # 标准化该列
        for i in range(N):
            v = (feat_list[i][j] - mu) / denom
            if clip is not None:
                v = max(-clip, min(clip, v))
            out[i][j] = v

    return out







#缓冲池
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, edge_index, edge_feature_list, select_num, valid_actions,
            action, reward, next_edge_index, next_edge_feature_list,
            next_select_num, next_valid_actions, done):

        self.buffer.append((edge_index, edge_feature_list, select_num,
                            valid_actions, action, reward,
                            next_edge_index, next_edge_feature_list,
                            next_select_num, next_valid_actions, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)  # 返回 List[transition]

    def size(self):
        return len(self.buffer)

    # ✅ 保存缓冲池
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                "capacity": self.capacity,
                "buffer": list(self.buffer)
            }, f)
        print(f"ReplayBuffer saved to {path}")

    # ✅ 加载缓冲池
    def load(self, path):
        if not os.path.exists(path):
            print("ReplayBuffer file not found.")
            return
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.capacity = data["capacity"]
        self.buffer = collections.deque(data["buffer"], maxlen=self.capacity)
        print(f"ReplayBuffer loaded from {path}")

def average_normals(n1: np.ndarray, n2: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    输入两个法向量（最好是单位法向），返回它们的平均单位法向。
    会先将 n2 调整到与 n1 同向（若 dot<0 则翻转 n2）。
    """
    n1 = np.asarray(n1, dtype=np.float64)
    n2 = np.asarray(n2, dtype=np.float64)

    # 同向化：避免 n1 和 n2 近似相反导致相加抵消
    if np.dot(n1, n2) < 0.0:
        n2 = -n2

    n = n1 + n2
    norm = np.linalg.norm(n)
    if norm < eps:
        raise ValueError("两法向几乎完全相反，平均后长度接近0，无法得到稳定平均法向。")

    return n / norm

#计算四边形的平均法向量
def Calculation_fh_average_normal_vector(mesh, fh):
    heh1 = mesh.halfedge_handle(fh)
    heh2 = mesh.next_halfedge_handle(heh1)
    heh3 = mesh.next_halfedge_handle(heh2)
    heh4 = mesh.next_halfedge_handle(heh3)

    vh1 = mesh.to_vertex_handle(heh1)
    vh2 = mesh.to_vertex_handle(heh2)
    vh3 = mesh.to_vertex_handle(heh3)
    vh4 = mesh.to_vertex_handle(heh4)

    normal_1 = triangle_unit_normal(mesh, vh1, vh2, vh3)
    normal_2 = triangle_unit_normal(mesh, vh3, vh4, vh1)



    aver_norm = average_normals(normal_1, normal_2)
    return list(aver_norm)





#用于翻转的法向量翻转检测函数
def flip_collect_neighbor_quads(mesh: om.PolyMesh, heh: om.HalfedgeHandle, a):
    """
    输入：
      - mesh: openmesh.PolyMesh（全四边形网格）
      - heh : 半边句柄

    过程：
      - fh1 = face_handle(heh)
      - fh2 = face_handle(opposite_halfedge_handle(heh))
      - 取 fh1 与 fh2 顶点并集（通常为6个顶点）
      - 遍历这6个顶点 incident 的所有面（vf），排除 fh1/fh2
      - 将每个面中的4个顶点索引作为一组，加入 list

    返回：
      - groups: List[List[int]]，每个元素是一个四边形面的4个顶点索引（按面环顺序）
    """
    # --------- 基本有效性检查 ----------
    if not hasattr(heh, "is_valid") or not heh.is_valid():
        raise ValueError("输入 heh 无效。")

    fh1 = mesh.face_handle(heh)
    if not fh1.is_valid():
        raise ValueError("heh 不属于有效面（可能是边界半边或网格不完整）。")

    heh_op = mesh.opposite_halfedge_handle(heh)
    fh2 = mesh.face_handle(heh_op)  # 可能是无效（边界）

    exclude_face_ids = {fh1.idx()}
    if fh2.is_valid():
        exclude_face_ids.add(fh2.idx())

    # --------- 收集 fh1 + fh2 的顶点并集 ----------
    vertex_ids = set(vh.idx() for vh in mesh.fv(fh1))
    if fh2.is_valid():
        vertex_ids |= set(vh.idx() for vh in mesh.fv(fh2))

    # --------- 遍历这些顶点的所有邻接面，收集四边形顶点组 ----------
    vh_idx_groups = []
    fh_norm = []

    seen_face_ids = set()  # 去重：同一面可能被多个顶点访问到

    for vid in vertex_ids:
        vh = om.VertexHandle(vid)
        for f in mesh.vf(vh):
            if not f.is_valid():
                continue
            fid = f.idx()

            # 排除 fh1 / fh2
            if fid in exclude_face_ids:
                continue

            # 去重
            if fid in seen_face_ids:
                continue
            seen_face_ids.add(fid)

            # 取该面的4个顶点索引
            heh1 = mesh.halfedge_handle(f)
            heh2 = mesh.next_halfedge_handle(heh1)
            heh3 = mesh.next_halfedge_handle(heh2)
            heh4 = mesh.next_halfedge_handle(heh3)

            vh1 = mesh.to_vertex_handle(heh1)
            vh2 = mesh.to_vertex_handle(heh2)
            vh3 = mesh.to_vertex_handle(heh3)
            vh4 = mesh.to_vertex_handle(heh4)
            vh_idx_groups.append([vh1.idx(), vh2.idx(), vh3.idx(), vh4.idx()])
            normal_1 = triangle_unit_normal(mesh, vh1, vh2, vh3)
            normal_2 = triangle_unit_normal(mesh, vh3, vh4, vh1)
            aver_norm = pd_norm(normal_1, normal_2)
            fh_norm.append(list(aver_norm))

    f_vh1 = mesh.to_vertex_handle(heh)
    f_heh1 = mesh.next_halfedge_handle(heh)
    f_heh2 = mesh.next_halfedge_handle(f_heh1)
    f_heh3 = mesh.next_halfedge_handle(f_heh2)
    f_vh2 = mesh.to_vertex_handle(f_heh1)
    f_vh3 = mesh.to_vertex_handle(f_heh2)
    f_vh4 = mesh.to_vertex_handle(f_heh3)
    f_heh4 = mesh.next_halfedge_handle(heh_op)
    f_heh5 = mesh.next_halfedge_handle(f_heh4)
    f_vh5 = mesh.to_vertex_handle(f_heh4)
    f_vh6 = mesh.to_vertex_handle(f_heh5)

    f_normal_1 = triangle_unit_normal(mesh, f_vh1, f_vh2, f_vh3)
    f_normal_2 = triangle_unit_normal(mesh, f_vh3, f_vh4, f_vh1)
    f_aver_norm_1 = pd_norm(f_normal_1, f_normal_2)


    f_normal_3 = triangle_unit_normal(mesh, f_vh4, f_vh5, f_vh6)
    f_normal_4 = triangle_unit_normal(mesh, f_vh6, f_vh1, f_vh4)
    f_aver_norm_2 = pd_norm(f_normal_3, f_normal_4)


    fh_norm.append(list(f_aver_norm_1))
    fh_norm.append(list(f_aver_norm_2))
    if a==0:
        vh_idx_groups.append([f_vh5.idx(), f_vh2.idx(), f_vh3.idx(), f_vh4.idx()])
        vh_idx_groups.append([f_vh2.idx(), f_vh5.idx(), f_vh6.idx(), f_vh1.idx()])
    else:
        vh_idx_groups.append([f_vh3.idx(), f_vh6.idx(), f_vh1.idx(), f_vh2.idx()])
        vh_idx_groups.append([f_vh3.idx(), f_vh4.idx(), f_vh5.idx(), f_vh6.idx()])



    return vh_idx_groups, fh_norm


def fh_4_vh(mesh, heh1):
    heh2 = mesh.next_halfedge_handle(heh1)
    heh3 = mesh.next_halfedge_handle(heh2)
    heh4 = mesh.next_halfedge_handle(heh3)

    vh1 = mesh.to_vertex_handle(heh1)
    vh2 = mesh.to_vertex_handle(heh2)
    vh3 = mesh.to_vertex_handle(heh3)
    vh4 = mesh.to_vertex_handle(heh4)

    return vh1, vh2, vh3,vh4

def split_collect_neighbor_quads(mesh: om.PolyMesh, heh: om.HalfedgeHandle):
    heh_op = mesh.opposite_halfedge_handle(heh)
    heh1_idx = halfedge_vh_score_(mesh, heh, bool_=False)
    heh1_ = mesh.halfedge_handle(heh1_idx)
    heh1_op = mesh.opposite_halfedge_handle(heh1_)
    fh1 = mesh.face_handle(heh)
    fh2 = mesh.face_handle(heh_op)
    fh3 = mesh.face_handle(heh1_)
    fh4 = mesh.face_handle(heh1_op)

    exclude_face_ids = {fh1.idx(), fh2.idx(), fh3.idx(), fh4.idx()}

    vertex_ids = set(vh.idx() for vh in mesh.fv(fh1))
    vertex_ids |= set(vh.idx() for vh in mesh.fv(fh2))
    vertex_ids |= set(vh.idx() for vh in mesh.fv(fh3))
    vertex_ids |= set(vh.idx() for vh in mesh.fv(fh4))

    # --------- 遍历这些顶点的所有邻接面，收集四边形顶点组 ----------
    vh_idx_groups = []
    fh_norm = []

    seen_face_ids = set()  # 去重：同一面可能被多个顶点访问到

    for vid in vertex_ids:
        vh = om.VertexHandle(vid)
        for f in mesh.vf(vh):
            if not f.is_valid():
                continue
            fid = f.idx()

            # 排除 fh1 / fh2
            if fid in exclude_face_ids:
                continue

            # 去重
            if fid in seen_face_ids:
                continue
            seen_face_ids.add(fid)

            # 取该面的4个顶点索引
            heh1 = mesh.halfedge_handle(f)
            heh2 = mesh.next_halfedge_handle(heh1)
            heh3 = mesh.next_halfedge_handle(heh2)
            heh4 = mesh.next_halfedge_handle(heh3)

            vh1 = mesh.to_vertex_handle(heh1)
            vh2 = mesh.to_vertex_handle(heh2)
            vh3 = mesh.to_vertex_handle(heh3)
            vh4 = mesh.to_vertex_handle(heh4)
            vh_idx_groups.append([vh1.idx(), vh2.idx(), vh3.idx(), vh4.idx()])
            normal_1 = triangle_unit_normal(mesh, vh1, vh2, vh3)
            normal_2 = triangle_unit_normal(mesh, vh3, vh4, vh1)
            aver_norm = pd_norm(normal_1, normal_2)

            fh_norm.append(list(aver_norm))

    vh1, vh2, vh3, vh4 = fh_4_vh(mesh,heh)
    vh5, vh6, vh7, vh8 = fh_4_vh(mesh, heh_op)
    vh9, vh10, vh11, vh12 = fh_4_vh(mesh, heh1_)
    vh13, vh14, vh15, vh16 = fh_4_vh(mesh, heh1_op)

    a = mesh.n_vertices()

    normal_1 = triangle_unit_normal(mesh, vh1, vh2, vh3)
    normal_2 = triangle_unit_normal(mesh, vh3, vh4, vh1)
    aver_norm_1 = pd_norm(normal_1, normal_2)


    normal_3 = triangle_unit_normal(mesh, vh5, vh6, vh7)
    normal_4 = triangle_unit_normal(mesh, vh7, vh8, vh5)
    aver_norm_2 = pd_norm(normal_3, normal_4)

    normal_5 = triangle_unit_normal(mesh, vh9, vh10, vh11)
    normal_6 = triangle_unit_normal(mesh, vh11, vh12, vh9)
    aver_norm_3 = pd_norm(normal_5, normal_6)

    normal_7 = triangle_unit_normal(mesh, vh13, vh14, vh15)
    normal_8 = triangle_unit_normal(mesh, vh15, vh16, vh13)
    aver_norm_4 = pd_norm(normal_7, normal_8)

    aver_norm_5 = average_normals(average_normals(aver_norm_1, aver_norm_2), average_normals(aver_norm_3, aver_norm_4))
    fh_norm.append(list(aver_norm_1))
    fh_norm.append(list(aver_norm_2))
    fh_norm.append(list(aver_norm_3))
    fh_norm.append(list(aver_norm_4))
    fh_norm.append(list(aver_norm_5))

    vh_idx_groups.append([a, vh2.idx(), vh3.idx(), vh4.idx()])
    vh_idx_groups.append([vh5.idx(), vh6.idx(), vh7.idx(), vh8.idx()])
    vh_idx_groups.append([vh9.idx(), vh10.idx(), vh11.idx(), vh12.idx()])
    vh_idx_groups.append([vh13.idx(), vh14.idx(), vh15.idx(), a])
    vh_idx_groups.append([a, vh4.idx(), vh1.idx(), vh12.idx()])

    return vh_idx_groups, fh_norm




def split_ph_ver(mesh, heh, features_vertex):
    heh_op = mesh.opposite_halfedge_handle(heh)
    heh1_idx = halfedge_vh_score_(mesh, heh, bool_=False)
    heh1_ = mesh.halfedge_handle(heh1_idx)
    heh1_op = mesh.opposite_halfedge_handle(heh1_)
    fh1 = mesh.face_handle(heh)
    fh2 = mesh.face_handle(heh_op)
    fh3 = mesh.face_handle(heh1_)
    fh4 = mesh.face_handle(heh1_op)
    fh_idx_set = {fh1.idx(), fh2.idx(), fh3.idx(), fh4.idx()}
    fh_idx_set.discard(-1)
    vertex_indices = []
    seen_ = set()
    for i in fh_idx_set:
        fh = mesh.face_handle(i)
        for vh in mesh.fv(fh):

            if vh.idx() in seen_:
                continue
            seen_.add(vh.idx())
            vertex_indices.append(vh.idx())
    a = mesh.n_vertices()
    ph_ver = get_multi_vertices_two_ring_neighbors_exclude_features(mesh, vertex_indices, features_vertex)
    ph_ver.append(a)
    return ph_ver




def collapse_collect_neighbor_quads(mesh, heh):
    vh_point_groups = []
    fh_norm = []
    f = mesh.face_handle(heh)
    heh1 = mesh.next_halfedge_handle(heh)
    heh2 = mesh.next_halfedge_handle(heh1)
    heh3 = mesh.next_halfedge_handle(heh2)

    vh1 = mesh.to_vertex_handle(heh)
    vh2 = mesh.to_vertex_handle(heh1)
    vh3 = mesh.to_vertex_handle(heh2)
    vh4 = mesh.to_vertex_handle(heh3)

    vh_seen = {vh1.idx(), vh2.idx(), vh3.idx(), vh4.idx()}
    fh_seen = set()
    for i in vh_seen:
        for fh in mesh.vf(mesh.vertex_handle(i)):

            if fh == f:
                continue
            if fh.idx() in fh_seen:
                continue
            heh_1 = mesh.halfedge_handle(fh)
            heh_2 = mesh.next_halfedge_handle(heh_1)
            heh_3 = mesh.next_halfedge_handle(heh_2)
            heh_4 = mesh.next_halfedge_handle(heh_3)

            vh_1 = mesh.to_vertex_handle(heh_1)
            vh_2 = mesh.to_vertex_handle(heh_2)
            vh_3 = mesh.to_vertex_handle(heh_3)
            vh_4 = mesh.to_vertex_handle(heh_4)

            p_1 = mesh.point(vh_1)
            p_2 = mesh.point(vh_2)
            p_3 = mesh.point(vh_3)
            p_4 = mesh.point(vh_4)

            if vh_1 == vh1 or vh_1 == vh3:
                p_1 = []
            if vh_2 == vh1 or vh_2 == vh3:
                p_2 = []
            if vh_3 == vh1 or vh_3 == vh3:
                p_3 = []
            if vh_4 == vh1 or vh_4 == vh3:
                p_4 = []

            normal_1 = triangle_unit_normal(mesh, vh1, vh2, vh3)
            normal_2 = triangle_unit_normal(mesh, vh3, vh4, vh1)
            aver_norm = pd_norm(normal_1, normal_2)


            fh_norm.append(list(aver_norm))
            vh_point_groups.append([list(p_1), list(p_2), list(p_3), list(p_4)])

    return fh_norm, vh_point_groups

def _find_vertex_idx_by_coord_scan(mesh, coord, tol=1e-9):
    """O(n) 扫描查找：coord=[x,y,z] -> vh.idx()，找不到返回 -1"""
    c = np.asarray(coord, dtype=float).reshape(-1)
    if c.size != 3:
        raise ValueError(f"coord must be length-3, got {c.size}")

    t = float(tol)
    if t == 0.0:
        for vh in mesh.vertices():
            p = np.asarray(mesh.point(vh), dtype=float)
            if (p[0] == c[0]) and (p[1] == c[1]) and (p[2] == c[2]):
                return vh.idx()
        return -1
    else:
        for vh in mesh.vertices():
            p = np.asarray(mesh.point(vh), dtype=float)
            if np.max(np.abs(p - c)) <= t:
                return vh.idx()
        return -1


def coords_nested_to_vidx(mesh, coords_nested, a: int, *, tol=1e-9, notfound_default=-1):
    """
    把多维 list 里的三维坐标映射成 mesh 顶点索引，返回同形状多维索引数组。

    输入:
      mesh: openmesh.TriMesh / PolyMesh
      coords_nested: 多维 list，叶子元素要么是 [x,y,z]，要么是 []/None
      a: int。若叶子元素为空数组 []（或 None），返回该默认顶点索引 a
      tol: 坐标匹配容差（推荐 1e-9 ~ 1e-6 视数据尺度）
      notfound_default: 坐标找不到对应顶点时返回值（默认 -1）

    返回:
      与 coords_nested 同形状的多维 list（叶子为 int 顶点索引）
    """
    if not isinstance(a, int):
        raise TypeError(f"a must be int, got {type(a)}")

    def is_empty_slot(x):
        return x is None or (isinstance(x, (list, tuple)) and len(x) == 0)

    def is_coord3(x):
        if isinstance(x, np.ndarray):
            return x.ndim == 1 and x.size == 3
        if isinstance(x, (list, tuple)):
            return len(x) == 3 and not isinstance(x[0], (list, tuple, np.ndarray))
        return False

    def recur(obj):
        # 空槽：[] 或 None
        if is_empty_slot(obj):
            return a

        # 3D 坐标叶子
        if is_coord3(obj):
            idx = _find_vertex_idx_by_coord_scan(mesh, obj, tol=tol)
            return idx if idx != -1 else notfound_default

        # 更高层 list：递归保持形状
        if isinstance(obj, list):
            return [recur(x) for x in obj]

        raise TypeError(f"Unsupported element type in nested list: {type(obj)} -> {obj}")

    return recur(coords_nested)













#对初始网格每一个面的包围盒信息存储到list1数组中，以及三角面各顶点坐标
def build_face_aabb_list(mesh):
    list1 = []
    list2 = []
    for i in range(mesh.n_faces()):
        fh = mesh.face_handle(i)
        heh1 = mesh.halfedge_handle(fh)
        heh2 = mesh.next_halfedge_handle(heh1)
        heh3 = mesh.next_halfedge_handle(heh2)
        heh4 = mesh.next_halfedge_handle(heh3)

        vh1 = mesh.to_vertex_handle(heh1)
        vh2 = mesh.to_vertex_handle(heh2)
        vh3 = mesh.to_vertex_handle(heh3)
        vh4 = mesh.to_vertex_handle(heh4)

        xmin1 = ymin1 = zmin1 = math.inf
        xmax1 = ymax1 = zmax1 = -math.inf
        xmin2 = ymin2 = zmin2 = math.inf
        xmax2 = ymax2 = zmax2 = -math.inf

        p1 = mesh.point(vh1)
        p2 = mesh.point(vh2)
        p3 = mesh.point(vh3)
        p4 = mesh.point(vh4)

        x_1, y_1, z_1 = float(p1[0]), float(p1[1]), float(p1[2])
        x_2, y_2, z_2 = float(p2[0]), float(p2[1]), float(p2[2])
        x_3, y_3, z_3 = float(p3[0]), float(p3[1]), float(p3[2])
        x_4, y_4, z_4 = float(p4[0]), float(p4[1]), float(p4[2])

        if x_1 < xmin1: xmin1 = x_1
        if y_1 < ymin1: ymin1 = y_1
        if z_1 < zmin1: zmin1 = z_1
        if x_1 > xmax1: xmax1 = x_1
        if y_1 > ymax1: ymax1 = y_1
        if z_1 > zmax1: zmax1 = z_1

        if x_2 < xmin1: xmin1 = x_2
        if y_2 < ymin1: ymin1 = y_2
        if z_2 < zmin1: zmin1 = z_2
        if x_2 > xmax1: xmax1 = x_2
        if y_2 > ymax1: ymax1 = y_2
        if z_2 > zmax1: zmax1 = z_2

        if x_3 < xmin1: xmin1 = x_3
        if y_3 < ymin1: ymin1 = y_3
        if z_3 < zmin1: zmin1 = z_3
        if x_3 > xmax1: xmax1 = x_3
        if y_3 > ymax1: ymax1 = y_3
        if z_3 > zmax1: zmax1 = z_3

        if x_1 < xmin2: xmin2 = x_1
        if y_1 < ymin2: ymin2 = y_1
        if z_1 < zmin2: zmin2 = z_1
        if x_1 > xmax2: xmax2 = x_1
        if y_1 > ymax2: ymax2 = y_1
        if z_1 > zmax2: zmax2 = z_1

        if x_3 < xmin2: xmin2 = x_3
        if y_3 < ymin2: ymin2 = y_3
        if z_3 < zmin2: zmin2 = z_3
        if x_3 > xmax2: xmax2 = x_3
        if y_3 > ymax2: ymax2 = y_3
        if z_3 > zmax2: zmax2 = z_3

        if x_4 < xmin2: xmin2 = x_4
        if y_4 < ymin2: ymin2 = y_4
        if z_4 < zmin2: zmin2 = z_4
        if x_4 > xmax2: xmax2 = x_4
        if y_4 > ymax2: ymax2 = y_4
        if z_4 > zmax2: zmax2 = z_4

        list1.append([xmin1, ymin1, zmin1, xmax1, ymax1, zmax1])
        list1.append([xmin2, ymin2, zmin2, xmax2, ymax2, zmax2])
        list2.append([list(p1),list(p2),list(p3)])
        list2.append([list(p1), list(p3), list(p4)])


    return list1, list2



def _point_in_aabb(p: np.ndarray, aabb: np.ndarray, eps: float = 1e-12) -> bool:
    xmin, ymin, zmin, xmax, ymax, zmax = aabb
    return (p[0] >= xmin - eps and p[0] <= xmax + eps and
            p[1] >= ymin - eps and p[1] <= ymax + eps and
            p[2] >= zmin - eps and p[2] <= zmax + eps)


def _sq_dist_point_aabb(p: np.ndarray, aabb: np.ndarray) -> float:
    """点到AABB的平方距离（p在盒内则为0）"""
    xmin, ymin, zmin, xmax, ymax, zmax = aabb
    dx = 0.0
    if p[0] < xmin: dx = xmin - p[0]
    elif p[0] > xmax: dx = p[0] - xmax

    dy = 0.0
    if p[1] < ymin: dy = ymin - p[1]
    elif p[1] > ymax: dy = p[1] - ymax

    dz = 0.0
    if p[2] < zmin: dz = zmin - p[2]
    elif p[2] > zmax: dz = p[2] - zmax

    return dx*dx + dy*dy + dz*dz


def _as_tri(tri_like) -> np.ndarray:
    """把 list2[a] 规范成 shape=(3,3) 的 np.float64 三角形顶点数组"""
    t = np.asarray(tri_like, dtype=np.float64)
    if t.shape == (3, 3):
        return t
    if t.ndim == 1 and t.size == 9:
        return t.reshape(3, 3)
    raise ValueError(f"tri 格式不对：期望(3,3)或长度9的一维数组，得到 {t.shape}")


def _closest_point_on_triangle(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    返回点 p 在三角形 (a,b,c) 上的最近点（Real-Time Collision Detection 经典算法）
    """
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a

    bp = p - b
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return b

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + v * ab

    cp = p - c
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return c

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + w * ac

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b)

    # inside face region
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return a + ab * v + ac * w

#投影函数
def project_vertex_by_aabb_and_tris(
    mesh: om.PolyMesh,
    vh: om.VertexHandle,
    list1,
    list2,
    eps_in_aabb: float = 1e-12,
):
    """
    输入：
      mesh, vh
      list1[a] = [xmin,ymin,zmin,xmax,ymax,zmax]
      list2[a] = 三角形三个顶点坐标（(3,3) 或 9个数）
    输出：
      投影后的顶点坐标 np.ndarray(3,)
    """
    if not vh.is_valid():
        raise ValueError("vh 无效。")
    if len(list1) != len(list2):
        raise ValueError(f"list1长度({len(list1)}) != list2长度({len(list2)})")

    p = np.asarray(mesh.point(vh), dtype=np.float64)

    # 1) 找出包含该点的所有AABB
    inside_ids = []
    for i, aabb_like in enumerate(list1):
        aabb = np.asarray(aabb_like, dtype=np.float64).reshape(6)
        if _point_in_aabb(p, aabb, eps=eps_in_aabb):
            inside_ids.append(i)

    # 2) 如果没有落在任何AABB内：先找最近的AABB
    if not inside_ids:
        best_i = None
        best_sq = float("inf")
        for i, aabb_like in enumerate(list1):
            aabb = np.asarray(aabb_like, dtype=np.float64).reshape(6)
            sq = _sq_dist_point_aabb(p, aabb)
            if sq < best_sq:
                best_sq = sq
                best_i = i
        inside_ids = [best_i]

    # 3) 在候选面片里算 点->三角形 最近距离，取最近的投影点
    best_proj = None
    best_dist2 = float("inf")

    for i in inside_ids:
        tri = _as_tri(list2[i])
        a, b, c = tri[0], tri[1], tri[2]
        q = _closest_point_on_triangle(p, a, b, c)   # 投影/最近点
        d2 = float(np.dot(p - q, p - q))
        if d2 < best_dist2:
            best_dist2 = d2
            best_proj = q

    return best_proj

#对做过平滑的顶点数组都做投影
def project_vertex_aabb(mesh, ph_ver, list1, list2):
    for i in range(len(ph_ver)):
        vh = mesh.vertex_handle(ph_ver[i])
        best_proj = project_vertex_by_aabb_and_tris(mesh, vh, list1, list2)
        mesh.set_point(vh, best_proj)

    return mesh


def flip_act_norm(mesh, vh_idx_groups):
    act_norm = []
    for sub in vh_idx_groups:
        vh1 = mesh.vertex_handle(sub[0])
        vh2 = mesh.vertex_handle(sub[1])
        vh3 = mesh.vertex_handle(sub[2])
        vh4 = mesh.vertex_handle(sub[3])
        normal_1 = triangle_unit_normal(mesh, vh1, vh2, vh3)
        normal_2 = triangle_unit_normal(mesh, vh3, vh4, vh1)
        aver_norm =  pd_norm(normal_1, normal_2)

        act_norm.append(aver_norm)

    return act_norm

def flip_ph_ver(mesh, heh, features_vertex):


    heh_ = mesh.opposite_halfedge_handle(heh)

    v_1 = mesh.to_vertex_handle(heh)
    heh1 = mesh.next_halfedge_handle(heh)
    v_2 = mesh.to_vertex_handle(heh1)
    heh2 = mesh.next_halfedge_handle(heh1)
    v_3 = mesh.to_vertex_handle(heh2)
    heh3 = mesh.next_halfedge_handle(heh2)
    v_4 = mesh.to_vertex_handle(heh3)
    heh4 = mesh.next_halfedge_handle(heh_)
    v_5 = mesh.to_vertex_handle(heh4)
    heh5 = mesh.next_halfedge_handle(heh4)
    v_6 = mesh.to_vertex_handle(heh5)
    vertex_indices = [v_1.idx(), v_2.idx(), v_3.idx(), v_4.idx(), v_5.idx(), v_6.idx()]
    ph_ver = get_multi_vertices_two_ring_neighbors_exclude_features(mesh, vertex_indices, features_vertex)


    return ph_ver



def Execute_act_sample(mesh, action, select_num, features_vertex, list1, list2, i):
    act = action % 6
    act_edge_idx = select_num[action // 6]
    print(f'动作为：{act}')
    print(f'操作的边索引为{act_edge_idx}')
    eh = mesh.edge_handle(act_edge_idx)
    heh1 = mesh.halfedge_handle(eh,0)
    heh2 = mesh.opposite_halfedge_handle(heh1)
    if act == 0:
        #左翻转
        vh_idx_groups, fh_norm = flip_collect_neighbor_quads(mesh, heh1, 0)
        ph_ver = flip_ph_ver(mesh, heh1, features_vertex)
        #执行拓扑操作
        mesh = flip_quad_halfedge(mesh, heh1, 0)
        #执行平滑投影操作
        mesh = laplacian_smooth_selected_vertices(mesh, ph_ver)
        mesh = project_vertex_aabb(mesh, ph_ver, list1, list2)

        act_norm = flip_act_norm(mesh, vh_idx_groups)
        #计算操作前后法向量翻转角度
        act_ang = compute_max_angle(fh_norm, act_norm)

    elif act == 1:
        #右翻转
        vh_idx_groups, fh_norm = flip_collect_neighbor_quads(mesh, heh1, 1)
        ph_ver = flip_ph_ver(mesh, heh1, features_vertex)
        # 执行拓扑操作
        mesh = flip_quad_halfedge(mesh, heh1, 1)
        # 执行平滑投影操作
        mesh = laplacian_smooth_selected_vertices(mesh, ph_ver)
        mesh = project_vertex_aabb(mesh, ph_ver, list1, list2)

        act_norm = flip_act_norm(mesh, vh_idx_groups)
        # 计算操作前后法向量翻转角度
        act_ang = compute_max_angle(fh_norm, act_norm)

    elif act == 2:
        #边分裂heh1
        vh_idx_groups, fh_norm = split_collect_neighbor_quads(mesh, heh1)
        ph_ver = split_ph_ver(mesh, heh1, features_vertex)
        # 执行拓扑操作
        mesh = split_edge_base_heh(mesh, heh1)
        # 执行平滑投影操作
        mesh = laplacian_smooth_selected_vertices(mesh, ph_ver)
        mesh = project_vertex_aabb(mesh, ph_ver, list1, list2)


        act_norm = flip_act_norm(mesh, vh_idx_groups)
        # 计算操作前后法向量翻转角度
        act_ang = compute_max_angle(fh_norm, act_norm)

    elif act == 3:
        # 边分裂heh2
        vh_idx_groups, fh_norm = split_collect_neighbor_quads(mesh, heh2)
        ph_ver = split_ph_ver(mesh, heh2, features_vertex)
        # 执行拓扑操作
        mesh = split_edge_base_heh(mesh, heh2)
        # 执行平滑投影操作
        mesh = laplacian_smooth_selected_vertices(mesh, ph_ver)
        mesh = project_vertex_aabb(mesh, ph_ver, list1, list2)

        act_norm = flip_act_norm(mesh, vh_idx_groups)
        # 计算操作前后法向量翻转角度
        act_ang = compute_max_angle(fh_norm, act_norm)

    elif act == 4:
        # heh1单元塌缩
        fh_norm, vh_point_groups = collapse_collect_neighbor_quads(mesh, heh1)

        mesh, vh5 = quad_collapse_diagonal(mesh, heh1, features_vertex)
        vh_idx_group = coords_nested_to_vidx(mesh, vh_point_groups, vh5.idx())
        features_vertex, features_edge = find_feature_edges_and_vertices(mesh)
        ph_ver = get_two_ring_neighbors_exclude_features(mesh, vh5, features_vertex)

        # 执行平滑投影操作
        mesh = laplacian_smooth_selected_vertices(mesh, ph_ver)
        mesh = project_vertex_aabb(mesh, ph_ver, list1, list2)

        act_norm = flip_act_norm(mesh, vh_idx_group)
        # 计算操作前后法向量翻转角度
        act_ang = compute_max_angle(fh_norm, act_norm)

    else:
        # heh2单元塌缩
        fh_norm, vh_point_groups = collapse_collect_neighbor_quads(mesh, heh2)

        mesh, vh5 = quad_collapse_diagonal(mesh, heh2, features_vertex)
        vh_idx_group = coords_nested_to_vidx(mesh, vh_point_groups, vh5.idx())
        features_vertex, features_edge = find_feature_edges_and_vertices(mesh)
        ph_ver = get_two_ring_neighbors_exclude_features(mesh, vh5, features_vertex)

        # 执行平滑投影操作
        mesh = laplacian_smooth_selected_vertices(mesh, ph_ver)
        mesh = project_vertex_aabb(mesh, ph_ver, list1, list2)

        act_norm = flip_act_norm(mesh, vh_idx_group)
        # 计算操作前后法向量翻转角度
        act_ang = compute_max_angle(fh_norm, act_norm)

    if act_ang>15 or (contains_zero_row(act_norm)):
        ill_done = 1
    else:
        ill_done = 0
    print(f'操作后翻转角度为{act_ang}')
    #om.write_mesh(rf'E:\MeshStudy\RL_Mesh\quad\model\test_model_1\1\1\{i}_seg_000.obj', mesh)

    return mesh, ill_done





#将无效拓扑操作进行掩码操作，（1）对度数为3的进行边分裂操作（2）操作后产生了度数大于6的奇异点

def mask_invalid_actions(logits, valid_actions):
    """
    对actor输出的一维策略值（logits）进行mask，将不合法动作的策略值置为极小值

    参数:
        logits: 一维torch.Tensor，actor输出的各动作原始策略值（未做softmax的logits）
                例：tensor([1.2, 3.4, 0.8, 2.1])
        valid_actions: 一维bool列表，与logits长度一致，False表示对应动作不合法，True表示合法
                例：[True, False, True, False]

    返回:
        masked_logits: 一维torch.Tensor，mask后的策略值，不合法动作对应位置为-1e9（可微分）

    异常:
        ValueError: 若logits不是一维张量，或valid_actions与logits长度不一致时抛出
    """
    # 1. 参数校验：确保logits是一维张量
    if logits.dim() != 1:
        raise ValueError(f"logits必须是一维张量，当前维度为: {logits.dim()}")

    # 2. 参数校验：确保mask长度与logits一致
    if len(valid_actions) != len(logits):
        raise ValueError(
            f"valid_actions长度({len(valid_actions)})必须与logits长度({len(logits)})一致"
        )

    # 3. 将bool列表转换为与logits同设备、同类型的torch布尔张量
    mask = torch.tensor(valid_actions, dtype=torch.bool, device=logits.device)
    invalid_mask = ~mask  # True表示不合法动作

    # 4. 打印合法动作个数
    valid_count = int(mask.sum().item())
    print(f"当前合法动作个数: {valid_count}")

    # 5. 对不合法动作的logits置为-1e9
    masked_logits = logits.masked_fill(invalid_mask, -1e9)

    return masked_logits


def split_safe_eh(mesh,heh):
    vh_idx = halfedge_vh_score_(mesh, heh, bool_=False)
    p = mesh.valence(mesh.vertex_handle(vh_idx))
    vh_=mesh.from_vertex_handle(heh)
    p2 = mesh.valence(vh_)
    if (p>5) or (p2>5):
        return False
    else:
        return True


#生成mask掩码数组
def validate_invalid_topology_operations(mesh, select_num, ver_valence):
    act_bool_list = []
    for i in select_num:
        if i == -1:
            bool_1 = False
            bool_2 = False
            bool_3 = False
            bool_4 = False
            bool_5 = False
            bool_6 = False
        else:

            act_eh = mesh.edge_handle(i)
            heh1 = mesh.halfedge_handle(act_eh,0)
            vh_1 = mesh.to_vertex_handle(heh1)
            heh2 = mesh.opposite_halfedge_handle(heh1)
            vh_2 = mesh.to_vertex_handle(heh2)
            heh3_idx = halfedge_vh_score_(mesh, heh1, bool_=False)
            heh4_idx = halfedge_vh_score_(mesh, heh2, bool_=False)
            heh_3 = mesh.prev_halfedge_handle(heh1)
            vh_3 = mesh.from_vertex_handle(heh_3)
            heh_4 = mesh.prev_halfedge_handle(heh2)
            vh_4 = mesh.from_vertex_handle(heh_4)
            if heh3_idx < 0:
                bool_3 = False
            elif not split_safe_eh(mesh,heh1):
                bool_3 = False

            else:
                heh3 = mesh.halfedge_handle(heh3_idx)
                # 动作3
                bool_3 = validate_split_safe(mesh, ver_valence, heh1, heh3)

            if heh4_idx < 0:
                bool_4 = False
            elif not split_safe_eh(mesh,heh2):
                bool_4 = False

            else:
                heh4 = mesh.halfedge_handle(heh4_idx)
                # 动作4
                bool_4 = validate_split_safe(mesh, ver_valence, heh2, heh4)


            if mesh.is_boundary(act_eh):
                bool_1 = False
                bool_2 = False
                bool_3 = False
                bool_4 = False
                bool_5 = False
                bool_6 = False
            else:
                #动作1
                bool_1 = validate_flip_safe(mesh, ver_valence, heh1, 0)
                #动作2
                bool_2 = validate_flip_safe(mesh, ver_valence, heh1, 1)


                #动作5
                bool_5 = validate_collapse_safe(mesh, ver_valence, heh1)
                if mesh.is_boundary(vh_3) and mesh.is_boundary(vh_1):
                    bool_5=False
                # 动作6
                bool_6 = validate_collapse_safe(mesh, ver_valence, heh2)
                if mesh.is_boundary(vh_4) and mesh.is_boundary(vh_2):
                    bool_6=False

        act_bool_list.append(bool_1)
        act_bool_list.append(bool_2)
        act_bool_list.append(bool_3)
        act_bool_list.append(bool_4)
        act_bool_list.append(bool_5)
        act_bool_list.append(bool_6)

    return act_bool_list


def validate_invalid_topology_operations_1(mesh, select_num):
    act_bool_list = []
    for i in select_num:
        if i == -1:
            bool_1 = False
            bool_2 = False
            bool_3 = False
            bool_4 = False
            bool_5 = False
            bool_6 = False
        else:
            act_eh = mesh.edge_handle(i)
            if mesh.is_boundary(act_eh):
                bool_1 = False
                bool_2 = False
                bool_3 = False
                bool_4 = False
                bool_5 = False
                bool_6 = False
            else:
                heh1 = mesh.halfedge_handle(act_eh,0)
                heh2 = mesh.opposite_halfedge_handle(heh1)
                fh1 = mesh.face_handle(heh1)
                fh2 = mesh.face_handle(heh2)
                vh1 = mesh.to_vertex_handle(heh1)
                vh2 = mesh.from_vertex_handle(heh1)
                if mesh.is_boundary(vh1) or (mesh.valence(vh1)==3):
                    bool_3 = False

                else:
                    bool_3 = True

                if mesh.is_boundary(vh2) or (mesh.valence(vh2)==3):
                    bool_4 = False

                else:
                    bool_4 = True

                bool_1 = True
                bool_2 = True

                if mesh.is_boundary(fh1):
                    bool_5 = False
                else:
                    bool_5 = True
                if mesh.is_boundary(fh2):
                    bool_6 = False
                else:
                    bool_6 = True

        act_bool_list.append(bool_1)
        act_bool_list.append(bool_2)
        act_bool_list.append(bool_3)
        act_bool_list.append(bool_4)
        act_bool_list.append(bool_5)
        act_bool_list.append(bool_6)

    return act_bool_list

def mesh_ver_over_6_(mesh):
    for vh in mesh.vertices():
        if mesh.valence(vh)>6:
            return False

    return True


#验证边分裂是否安全（对于度数为3的不能执行边分裂，不能产生度数大于度数为6的顶点）
def validate_split_safe(mesh, ver_valence, heh1, heh2):
    v1 = mesh.to_vertex_handle(heh1)
    a = ver_valence[v1.idx()]
    v2 = mesh.from_vertex_handle(heh1)
    b = ver_valence[v2.idx()]
    v3 = mesh.from_vertex_handle(heh2)
    c = ver_valence[v3.idx()]
    if a == 3:
        return False
    elif (b+1>6) or (c+1>6):
        return False
    else:
        return True
#验证边翻转是否安全（不能产生度数大于度数为6的顶点）
def validate_flip_safe(mesh, ver_valence, he, a):

    v1 = mesh.from_vertex_handle(he)
    v2 = mesh.to_vertex_handle(he)
    opp = mesh.opposite_halfedge_handle(he)
    he1 = mesh.next_halfedge_handle(he)
    he2 = mesh.next_halfedge_handle(he1)
    v3 = mesh.to_vertex_handle(he1)
    v4 = mesh.to_vertex_handle(he2)
    he3 = mesh.next_halfedge_handle(opp)
    he4 = mesh.next_halfedge_handle(he3)
    v5 = mesh.to_vertex_handle(he3)
    v6 = mesh.to_vertex_handle(he4)


    if a == 0 :
        a = ver_valence[v3.idx()]
        b = ver_valence[v5.idx()]

    else:
        a = ver_valence[v4.idx()]
        b = ver_valence[v6.idx()]
    if (a + 1 > 6) or (b + 1 > 6):
        return False
    else:
        return True




#验证单元塌缩是否安全（不能产生度数大于度数为6的顶点）
def validate_collapse_safe(mesh, ver_valence, heh):
    v1 = mesh.to_vertex_handle(heh)
    heh1 = mesh.next_halfedge_handle(heh)
    heh2 = mesh.next_halfedge_handle(heh1)
    v2 = mesh.to_vertex_handle(heh2)
    a = ver_valence[v1.idx()]
    b = ver_valence[v2.idx()]
    if a+b-2>6:
        return False
    else:
        return True

def samply_data(base_file, mesh, actor, p, done_buffer, not_done_buffer, device, best_score_, b, local_num):

    ver_valence, ver_valence_diff = compute_vertex_valence_diff(mesh)
    #features_vertex, features_edge = find_feature_edges_and_vertices(mesh)
    in_score = sum(ver_valence_diff)
    list1, list2 = build_face_aabb_list(mesh)
    mesh_episode_return = 0
    normalize_cols = list(range(16, 23))
    for i in range(round(p * in_score)):


        features_vertex, features_edge = find_feature_edges_and_vertices(mesh)
        ver_valence, ver_valence_diff = compute_vertex_valence_diff(mesh)
        #获得每条边的特征
        edge_feature_list = get_edge_feature(mesh, ver_valence, ver_valence_diff)
        edge_feature_list = zscore_standardize_feature_list_1(edge_feature_list, normalize_cols)
        #对所有面进行打分
        act_local_eh, fh_score_list = base_fh_score_1(mesh, features_edge, ver_valence_diff)
        #获得候选操作边集合
        select_num, inst_score = get_top_a_values_from_list(fh_score_list, act_local_eh, local_num)
        #构建以边为节点的图结构
        edge_index = build_edge_as_node_edge_index(mesh)
        select_num_tor = torch.tensor(select_num, dtype=torch.long).to(device)
        edge_index_tor = torch.tensor(edge_index, dtype=torch.long).contiguous().to(device)
        edge_attr = torch.tensor(edge_feature_list, dtype=torch.float).to(device)
        valid_actions = validate_invalid_topology_operations(mesh, select_num, ver_valence)

        with torch.no_grad():
            data = Data(edge_attr=edge_attr, edge_index=edge_index_tor, select_num=select_num_tor)
            out_put_1 = actor(data.edge_attr, data.edge_index, data.select_num)

            # 合法动作mask后的logits
            masked_logits = mask_invalid_actions(out_put_1, valid_actions)

            # actor策略对应的合法动作概率
            probs_flat = F.softmax(masked_logits, dim=-1)

            # 转成tensor mask
            valid_mask = torch.tensor(valid_actions, dtype=torch.bool, device=probs_flat.device)
            valid_indices = torch.nonzero(valid_mask, as_tuple=False).view(-1)

            if valid_indices.numel() == 0:
                raise ValueError("当前没有合法动作，无法采样。")

            # 小概率走“合法动作均匀采样”，其余走“策略概率采样”
            eps_uniform = 0.05

            if torch.rand(1, device=probs_flat.device).item() < eps_uniform:
                # 在合法动作索引里平均概率采样
                rand_pos = torch.randint(0, valid_indices.numel(), (1,), device=probs_flat.device)
                action = valid_indices[rand_pos].squeeze(0)
                sample_mode = "uniform"
            else:
                # 按actor输出概率采样
                action_dist = torch.distributions.Categorical(probs=probs_flat)
                action = action_dist.sample()
                sample_mode = "policy"

            action_int = int(action.item())

            # 可选：调试输出
            # print("sample_mode =", sample_mode, "action =", action_int)

            mesh, ill_done = Execute_act_sample(
                mesh, action_int, select_num, features_vertex, list1, list2, i
            )
        # 前处理：检查是否有度数超过2的顶点，如有，进行单元塌缩消除
        mesh = Preprocessing_act(mesh)
        next_ver_valence, next_ver_valence_diff = compute_vertex_valence_diff(mesh)
        next_features_vertex, next_features_edge = find_feature_edges_and_vertices(mesh)
        next_act_local_eh, next_fh_score_list = base_fh_score_1(mesh, next_features_edge, next_ver_valence_diff)
        next_select_num, next_inst_score = get_top_a_values_from_list(next_fh_score_list, next_act_local_eh, local_num)
        next_edge_index = build_edge_as_node_edge_index(mesh)


        next_valid_actions = validate_invalid_topology_operations(mesh, next_select_num, next_ver_valence)
        # 获得每条边的特征
        next_edge_feature_list = get_edge_feature(mesh, next_ver_valence, next_ver_valence_diff)
        next_edge_feature_list = zscore_standardize_feature_list_1(next_edge_feature_list, normalize_cols)
        if sum(next_ver_valence_diff)<best_score_:
            best_score_ = sum(next_ver_valence_diff)
            om.write_mesh(rf'{base_file}\best_score_{b}.obj', mesh)
        if (ill_done == 1) or (not check_mesh_no_degenerate_faces(mesh)):
            reward = -1
        elif next_inst_score == 0:
            reward = 1
        else:
            #reward = (sum(ver_valence_diff) - sum(next_ver_valence_diff))/inst_score
            #reward = sum(ver_valence_diff) - sum(next_ver_valence_diff)
            reward = (sum(ver_valence_diff) - sum(next_ver_valence_diff)) / sum(ver_valence_diff)
        mesh_episode_return += reward
        if (ill_done==1) or (i == round(p * in_score)-1) or (next_inst_score == 0) or (not eh_len_safe(mesh)):
            done =1
            done_buffer.add(edge_index, edge_feature_list, select_num, valid_actions, action_int, reward,
                                next_edge_index, next_edge_feature_list, next_select_num, next_valid_actions, done)
            break
        else:
            done = 0
            not_done_buffer.add(edge_index, edge_feature_list, select_num, valid_actions, action_int, reward, next_edge_index, next_edge_feature_list, next_select_num, next_valid_actions, done)


    return done_buffer, not_done_buffer, best_score_, mesh_episode_return


def samply_data_1(base_file, mesh, actor, p, replay_buffer, device, best_score_, b, local_num):

    ver_valence, ver_valence_diff = compute_vertex_valence_diff(mesh)
    #features_vertex, features_edge = find_feature_edges_and_vertices(mesh)
    in_score = sum(ver_valence_diff)
    list1, list2 = build_face_aabb_list(mesh)
    mesh_episode_return = 0
    normalize_cols = list(range(16, 23))
    for i in range(round(p * in_score)):


        features_vertex, features_edge = find_feature_edges_and_vertices(mesh)
        ver_valence, ver_valence_diff = compute_vertex_valence_diff(mesh)
        #获得每条边的特征
        edge_feature_list = get_edge_feature(mesh, ver_valence, ver_valence_diff)
        edge_feature_list = zscore_standardize_feature_list_1(edge_feature_list, normalize_cols)
        #对所有面进行打分
        act_local_eh, fh_score_list = base_fh_score_1(mesh, features_edge, ver_valence_diff)
        #获得候选操作边集合
        select_num, inst_score = get_top_a_values_from_list(fh_score_list, act_local_eh, local_num)
        #构建以边为节点的图结构
        edge_index = build_edge_as_node_edge_index(mesh)
        select_num_tor = torch.tensor(select_num, dtype=torch.long).to(device)
        edge_index_tor = torch.tensor(edge_index, dtype=torch.long).contiguous().to(device)
        edge_attr = torch.tensor(edge_feature_list, dtype=torch.float).to(device)
        valid_actions = validate_invalid_topology_operations_1(mesh, select_num)

        with torch.no_grad():
            data = Data(edge_attr=edge_attr, edge_index=edge_index_tor, select_num=select_num_tor)
            out_put_1 = actor(data.edge_attr, data.edge_index, data.select_num)

            # 合法动作mask后的logits
            masked_logits = mask_invalid_actions(out_put_1, valid_actions)

            # actor策略对应的合法动作概率
            probs_flat = F.softmax(masked_logits, dim=-1)

            # 转成tensor mask
            valid_mask = torch.tensor(valid_actions, dtype=torch.bool, device=probs_flat.device)
            valid_indices = torch.nonzero(valid_mask, as_tuple=False).view(-1)

            if valid_indices.numel() == 0:
                raise ValueError("当前没有合法动作，无法采样。")

            # 直接按照 actor 输出的概率分布采样
            action_dist = torch.distributions.Categorical(probs=probs_flat)
            action = action_dist.sample()
            sample_mode = "policy"

            action_int = int(action.item())
            max_prob = float(probs_flat.max().item())
            min_prob = float(probs_flat.min().item())
            selected_prob = float(probs_flat[action_int].item())

            print(f"动作概率最大值: {max_prob:.8f}")
            print(f"动作概率最小值: {min_prob:.8f}")
            print(f"选中动作概率: {selected_prob:.8f}")

            # 可选：调试输出
            # print("sample_mode =", sample_mode, "action =", action_int)

            mesh, ill_done = Execute_act_sample(
                mesh, action_int, select_num, features_vertex, list1, list2, i
            )
        # 前处理：检查是否有度数超过2的顶点，如有，进行单元塌缩消除
        mesh = Preprocessing_act(mesh)
        next_ver_valence, next_ver_valence_diff = compute_vertex_valence_diff(mesh)
        next_features_vertex, next_features_edge = find_feature_edges_and_vertices(mesh)
        next_act_local_eh, next_fh_score_list = base_fh_score_1(mesh, next_features_edge, next_ver_valence_diff)
        next_select_num, next_inst_score = get_top_a_values_from_list(next_fh_score_list, next_act_local_eh, local_num)
        next_edge_index = build_edge_as_node_edge_index(mesh)


        next_valid_actions = validate_invalid_topology_operations_1(mesh, next_select_num)
        # 获得每条边的特征
        next_edge_feature_list = get_edge_feature(mesh, next_ver_valence, next_ver_valence_diff)
        next_edge_feature_list = zscore_standardize_feature_list_1(next_edge_feature_list,normalize_cols)
        if sum(next_ver_valence_diff)<best_score_:
            best_score_ = sum(next_ver_valence_diff)
            om.write_mesh(rf'{base_file}\best_score_{b}.obj', mesh)
        if (ill_done == 1) or (not check_mesh_no_degenerate_faces(mesh)) or (not mesh_ver_over_6_(mesh)):
            reward = -1
        elif next_inst_score == 0:
            reward = 1
        else:
            #reward = (sum(ver_valence_diff) - sum(next_ver_valence_diff))/inst_score
            #reward = sum(ver_valence_diff) - sum(next_ver_valence_diff)
            reward = (sum(ver_valence_diff) - sum(next_ver_valence_diff)) / sum(ver_valence_diff)
        mesh_episode_return += reward
        if (ill_done==1) or (i == round(p * in_score)-1) or (next_inst_score == 0) or (not eh_len_safe(mesh)):
            done =1
            replay_buffer.add(edge_index, edge_feature_list, select_num, valid_actions, action_int, reward,
                                next_edge_index, next_edge_feature_list, next_select_num, next_valid_actions, done)
            break
        else:
            done = 0
            replay_buffer.add(edge_index, edge_feature_list, select_num, valid_actions, action_int, reward, next_edge_index, next_edge_feature_list, next_select_num, next_valid_actions, done)


    return replay_buffer, best_score_, mesh_episode_return




def vail_model_data(base_file, best_score_flie, p, actor, device, best_score, loceal_num):
    episode_return = 0
    for vl in range(50):
        obj_file = rf'{base_file}\{vl+1}.obj'
        mesh = om.read_polymesh(obj_file)
        ver_valence, ver_valence_diff = compute_vertex_valence_diff(mesh)
        in_score = sum(ver_valence_diff)
        list1, list2 = build_face_aabb_list(mesh)
        mesh_episode_return = 0
        normalize_cols = list(range(16, 23))
        for i in range(round(p * in_score)):
            features_vertex, features_edge = find_feature_edges_and_vertices(mesh)
            ver_valence, ver_valence_diff = compute_vertex_valence_diff(mesh)
            # 获得每条边的特征
            edge_feature_list = get_edge_feature(mesh, ver_valence, ver_valence_diff)
            edge_feature_list = zscore_standardize_feature_list_1(edge_feature_list,normalize_cols)
            # 对所有面进行打分
            act_local_eh, fh_score_list = base_fh_score_1(mesh, features_edge, ver_valence_diff)
            # 获得候选操作边集合
            select_num, inst_score = get_top_a_values_from_list(fh_score_list, act_local_eh, loceal_num)
            # 构建以边为节点的图结构
            edge_index = build_edge_as_node_edge_index(mesh)
            select_num_tor = torch.tensor(select_num, dtype=torch.long).to(device)
            edge_index_tor = torch.tensor(edge_index, dtype=torch.long).contiguous().to(device)
            edge_attr = torch.tensor(edge_feature_list, dtype=torch.float).to(device)
            valid_actions = validate_invalid_topology_operations(mesh, select_num, ver_valence)

            with torch.no_grad():
                data = Data(edge_attr=edge_attr, edge_index=edge_index_tor, select_num=select_num_tor)
                out_put_1 = actor(data.edge_attr, data.edge_index, data.select_num)

                # 合法动作mask后的logits
                masked_logits = mask_invalid_actions(out_put_1, valid_actions)

                # actor策略对应的合法动作概率
                probs_flat = F.softmax(masked_logits, dim=-1)
                action_int = int(torch.argmax(probs_flat).item())
                mesh, ill_done = Execute_act_sample(
                    mesh, action_int, select_num, features_vertex, list1, list2, i
                )
                # 前处理：检查是否有度数超过2的顶点，如有，进行单元塌缩消除
                mesh = Preprocessing_act(mesh)
                next_ver_valence, next_ver_valence_diff = compute_vertex_valence_diff(mesh)



                if sum(next_ver_valence_diff) < best_score[vl]:
                    best_score[vl] = sum(next_ver_valence_diff)
                    om.write_mesh(rf'{best_score_flie}\best_score_{vl}.obj', mesh)
                if (ill_done == 1) or (not check_mesh_no_degenerate_faces(mesh)):
                    reward = -1
                elif sum(next_ver_valence_diff) == 0:
                    reward = 1
                else:
                    # reward = (sum(ver_valence_diff) - sum(next_ver_valence_diff))/inst_score
                    # reward = sum(ver_valence_diff) - sum(next_ver_valence_diff)
                    reward = (sum(ver_valence_diff) - sum(next_ver_valence_diff)) / sum(ver_valence_diff)
                mesh_episode_return += reward
                if (ill_done == 1) or (i == round(p * in_score) - 1) or (sum(next_ver_valence_diff) == 0) or (
                not eh_len_safe(mesh)):
                    done = 1

                    break
                else:
                    done = 0
        episode_return += mesh_episode_return

    return episode_return, best_score






def vail_model_data_1(base_file, best_score_flie, p, actor, device, best_score, loceal_num):

    mesh = om.read_polymesh(base_file)
    ver_valence, ver_valence_diff = compute_vertex_valence_diff(mesh)
    in_score = sum(ver_valence_diff)
    list1, list2 = build_face_aabb_list(mesh)
    mesh_episode_return = 0
    time_list_1 = []  # 用于存储决策时间数组，单位：ms
    for i in range(round(p * in_score)):

        features_vertex, features_edge = find_feature_edges_and_vertices(mesh)
        ver_valence, ver_valence_diff = compute_vertex_valence_diff(mesh)
        # 获得每条边的特征
        edge_feature_list = get_edge_feature(mesh, ver_valence, ver_valence_diff)
        edge_feature_list = zscore_standardize_feature_list(edge_feature_list)
        # 对所有面进行打分
        act_local_eh, fh_score_list = base_fh_score_1(mesh, features_edge, ver_valence_diff)
        # 获得候选操作边集合
        select_num, inst_score = get_top_a_values_from_list(fh_score_list, act_local_eh, loceal_num)
        # 构建以边为节点的图结构
        edge_index = build_edge_as_node_edge_index(mesh)
        select_num_tor = torch.tensor(select_num, dtype=torch.long).to(device)
        edge_index_tor = torch.tensor(edge_index, dtype=torch.long).contiguous().to(device)
        edge_attr = torch.tensor(edge_feature_list, dtype=torch.float).to(device)
        valid_actions = validate_invalid_topology_operations(mesh, select_num, ver_valence)

        with torch.no_grad():
            data = Data(edge_attr=edge_attr, edge_index=edge_index_tor, select_num=select_num_tor)
            start_time_1 = time.perf_counter()
            out_put_1 = actor(data.edge_attr, data.edge_index, data.select_num)
            end_time_1 = time.perf_counter()
            #计算智能体给出决策耗时
            elapsed_ms_1 = (end_time_1 - start_time_1) * 1000
            time_list_1.append(elapsed_ms_1)
            # 合法动作mask后的logits
            masked_logits = mask_invalid_actions(out_put_1, valid_actions)

            # actor策略对应的合法动作概率
            probs_flat = F.softmax(masked_logits, dim=-1)
            action_dist = torch.distributions.Categorical(probs=probs_flat)
            action_int = action_dist.sample()
            max_prob = float(probs_flat.max().item())
            min_prob = float(probs_flat.min().item())
            selected_prob = float(probs_flat[action_int].item())

            print(f"动作概率最大值: {max_prob:.8f}")
            print(f"动作概率最小值: {min_prob:.8f}")
            print(f"选中动作概率: {selected_prob:.8f}")

            action_int = int(action_int.item())
            mesh, ill_done = Execute_act_sample(
                mesh, action_int, select_num, features_vertex, list1, list2, i
            )
            # 前处理：检查是否有度数超过2的顶点，如有，进行单元塌缩消除
            mesh = Preprocessing_act(mesh)
            next_ver_valence, next_ver_valence_diff = compute_vertex_valence_diff(mesh)
        if sum(next_ver_valence_diff) < best_score:
            best_score = sum(next_ver_valence_diff)
            om.write_mesh(rf'{best_score_flie}\best_score_test.obj', mesh)
        if (ill_done == 1) or (not check_mesh_no_degenerate_faces(mesh)):
            reward = -1
        elif sum(next_ver_valence_diff) == 0:
            reward = 1
        else:
            # reward = (sum(ver_valence_diff) - sum(next_ver_valence_diff))/inst_score
            # reward = sum(ver_valence_diff) - sum(next_ver_valence_diff)
            reward = (sum(ver_valence_diff) - sum(next_ver_valence_diff)) / sum(ver_valence_diff)
        mesh_episode_return += reward
        if (ill_done == 1) or (i == round(p * in_score) - 1) or (sum(next_ver_valence_diff) == 0) or (
                not eh_len_safe(mesh)):
            done = 1

            break
        else:
            done = 0
    juece_avg_time=sum(time_list_1) / len(time_list_1)

    return best_score, mesh_episode_return, juece_avg_time



def create_graph_data(edge_index, edge_feature_list, select_num, device):
    edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).contiguous()
    edge_attr  = torch.tensor(edge_feature_list, dtype=torch.float, device=device)
    select_num = torch.tensor(select_num, dtype=torch.long, device=device)

    data = Data(edge_attr=edge_attr, edge_index=edge_index, select_num=select_num)
    data.num_nodes = edge_attr.size(0)  # 节点数=边节点数（消除num_nodes警告）
    return data


def sample_and_package(done_buffer, not_done_buffer, batch_size, device, done_ratio=0.25):
    # --------- 1. 计算每类采样数 ----------
    n_done = max(1, int(batch_size * done_ratio))
    n_not_done = batch_size - n_done

    # 防止 done_buffer 太小
    n_done = min(n_done, done_buffer.size())
    n_not_done = min(n_not_done, not_done_buffer.size())

    # 如果 not_done 不够，用 done 补
    if n_done + n_not_done < batch_size:
        n_done = min(batch_size - n_not_done, done_buffer.size())

    # --------- 2. 从两个 buffer 采样 ----------
    batch_done = done_buffer.sample(n_done)
    batch_not_done = not_done_buffer.sample(n_not_done)

    # 拼接
    batch = []
    if n_done > 0:
        batch += done_buffer.sample(n_done)
    if n_not_done > 0:
        batch += not_done_buffer.sample(n_not_done)

    # 打乱，避免前面全是 done
    random.shuffle(batch)

    # --------- 3. 解包 ----------
    (b_edge_index, b_edge_feature_list, b_select_num, b_valid_actions,
     b_action, b_reward,
     b_next_edge_index, b_next_edge_feature_list, b_next_select_num, b_next_valid_actions,
     b_done) = zip(*batch)

    # --------- 4. 构造图状态 ----------
    states_list = []
    next_states_list = []
    for i in range(len(batch)):
        states_list.append(
            create_graph_data(b_edge_index[i], b_edge_feature_list[i], b_select_num[i], device)
        )
        next_states_list.append(
            create_graph_data(b_next_edge_index[i], b_next_edge_feature_list[i], b_next_select_num[i], device)
        )

    actions = torch.tensor(b_action, dtype=torch.long, device=device)
    rewards = torch.tensor(b_reward, dtype=torch.float, device=device)
    dones   = torch.tensor(b_done,   dtype=torch.float, device=device)

    return (
        states_list,
        next_states_list,
        actions,
        rewards,
        dones,
        list(b_valid_actions),
        list(b_next_valid_actions)
    )



def sample_and_package_1(replay_buffer, batch_size, device):
    """
    从统一缓冲池中随机采样一个 batch，并打包成训练所需数据

    参数:
        replay_buffer: 统一经验回放池，需支持
                       - size()
                       - sample(n)
        batch_size:    采样批大小
        device:        torch 设备

    返回:
        states_list:         当前状态图对象列表
        next_states_list:    下一状态图对象列表
        actions:             [batch]
        rewards:             [batch]
        dones:               [batch]
        b_valid_actions:     当前状态合法动作列表
        b_next_valid_actions:下一状态合法动作列表
    """
    # --------- 1. 确定实际采样数 ----------
    actual_batch_size = min(batch_size, replay_buffer.size())
    if actual_batch_size <= 0:
        raise ValueError("replay_buffer 为空，无法采样。")

    # --------- 2. 统一缓冲池采样 ----------
    batch = replay_buffer.sample(actual_batch_size)

    # 如有需要可打乱（如果 buffer.sample 本身已随机，可省略）
    random.shuffle(batch)

    # --------- 3. 解包 ----------
    (b_edge_index, b_edge_feature_list, b_select_num, b_valid_actions,
     b_action, b_reward,
     b_next_edge_index, b_next_edge_feature_list, b_next_select_num, b_next_valid_actions,
     b_done) = zip(*batch)

    # --------- 4. 构造图状态 ----------
    states_list = []
    next_states_list = []

    for i in range(len(batch)):
        states_list.append(
            create_graph_data(
                b_edge_index[i],
                b_edge_feature_list[i],
                b_select_num[i],
                device
            )
        )
        next_states_list.append(
            create_graph_data(
                b_next_edge_index[i],
                b_next_edge_feature_list[i],
                b_next_select_num[i],
                device
            )
        )

    # --------- 5. 构造张量 ----------
    actions = torch.tensor(b_action, dtype=torch.long, device=device)
    rewards = torch.tensor(b_reward, dtype=torch.float, device=device)
    dones   = torch.tensor(b_done, dtype=torch.float, device=device)

    return (
        states_list,
        next_states_list,
        actions,
        rewards,
        dones,
        list(b_valid_actions),
        list(b_next_valid_actions)
    )




# 计算目标Q值,直接用策略网络的输出概率进行期望计算
def calc_target(target_critic_1, target_critic_2, log_alpha,
                rewards, next_states_list, dones, gamma,
                next_valid_actions_list, actor, device):

    alpha = log_alpha.exp()
    td_list = []

    with torch.no_grad():
        for i, (s_next, mask_next) in enumerate(zip(next_states_list, next_valid_actions_list)):
            # s_next 已经在 device 上了（create_graph_data 做过）
            # 1) actor 输出 logits（1D: [Ai]，Ai = Ki*6）
            next_logits = actor(s_next.edge_attr, s_next.edge_index, s_next.select_num)  # [Ai]

            # 2) mask 应用于 logits，再 softmax
            masked_logits = mask_invalid_actions(next_logits, mask_next)                # [Ai]
            probs = F.softmax(masked_logits, dim=-1)                                   # [Ai]
            log_probs = torch.log(probs + 1e-12)                                       # [Ai]

            # 3) critic 输出必须和 probs 同长度 [Ai]
            q1 = target_critic_1(s_next.edge_attr, s_next.edge_index, s_next.select_num)  # [Ai] or [Ai,1]
            q2 = target_critic_2(s_next.edge_attr, s_next.edge_index, s_next.select_num)

            # 如果 critic 输出是 [Ai,1]，压成 [Ai]
            if q1.dim() == 2 and q1.size(-1) == 1: q1 = q1.squeeze(-1)
            if q2.dim() == 2 and q2.size(-1) == 1: q2 = q2.squeeze(-1)

            min_q = torch.min(q1, q2)  # [Ai]

            # 4) 离散SAC: V(s') = Σ_a π(a|s')[ minQ(s',a) - α log π(a|s') ]
            v_next = torch.sum(probs * (min_q - alpha * log_probs))  # scalar

            # 5) TD target
            td = rewards[i] + gamma * (1.0 - dones[i]) * v_next
            td_list.append(td)

    td_target = torch.stack(td_list, dim=0).unsqueeze(1)  # [B,1]
    return td_target

#软更新神经网络
def soft_update(net, target_net, tau):
    for param_target, param in zip(target_net.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)


def _to_1d(q: torch.Tensor) -> torch.Tensor:
    """把 critic 输出统一成 1D: [Ai]"""
    if q.dim() == 2 and q.size(-1) == 1:
        q = q.squeeze(-1)  # [Ai,1] -> [Ai]
    if q.dim() != 1:
        raise ValueError(f"critic 输出必须是 [Ai] 或 [Ai,1]，但得到 {tuple(q.shape)}")
    return q

def critic_q_sa_from_list(critic, states_list, actions):
    """
    等价于经典的 critic(states).gather(1, actions)，但适用于 states_list(变长图)
    - critic(s_i) -> q_all_i: [Ai]
    - 取 q_all_i[action_i] -> Q(s_i, a_i)
    返回: [B,1]
    """
    if torch.is_tensor(actions):
        actions_list = actions.view(-1).tolist()  # Tensor[B] -> list[int]
    else:
        actions_list = list(actions)

    q_sa = []
    for s, a in zip(states_list, actions_list):
        q_all = critic(s.edge_attr, s.edge_index, s.select_num)  # [Ai] or [Ai,1]
        q_all = _to_1d(q_all)                                    # [Ai]

        if not (0 <= a < q_all.numel()):
            raise ValueError(f"action={a} 超出该样本动作范围 [0, {q_all.numel()-1}]")

        q_sa.append(q_all[a])  # 标量（可反传）

    return torch.stack(q_sa, dim=0).unsqueeze(1)  # [B,1]



def update_actor_and_alpha(
    clip_norm, actor, critic_1, critic_2,
    actor_optimizer,
    log_alpha, log_alpha_optimizer,
    states_list, valid_actions_list,
    target_entropy,
    device,
    eps: float = 1e-12,
):
    """
    对应经典离散SAC的：
      probs = actor(states)
      entropy = ...
      min_qvalue = ...
      actor_loss = mean(-alpha*entropy - E[minQ])
      alpha_loss = mean((entropy - target_entropy).detach() * alpha)
    但这里 states 是 list，每个样本动作数 Ai 可变，所以逐条算。
    """

    # ------- 1) 更新 actor（α视为常数，detach） -------
    alpha_detached = log_alpha.exp().detach()   # 标量，作为actor更新时的常数

    per_sample_actor_loss = []
    per_sample_entropy = []

    for s, mask in zip(states_list, valid_actions_list):
        # (a) actor logits: [Ai]
        logits = actor(s.edge_attr, s.edge_index, s.select_num)   # [Ai]

        # (b) mask + softmax 得到 probs: [Ai]
        masked_logits = mask_invalid_actions(logits, mask)        # [Ai]
        probs = F.softmax(masked_logits, dim=-1)                  # [Ai]
        probs = torch.clamp(probs, min=eps, max=1.0)              # 防止 log(0)
        log_probs = torch.log(probs)                              # [Ai]

        # (c) entropy: scalar
        entropy = -(probs * log_probs).sum()                      # 标量
        per_sample_entropy.append(entropy)

        # (d) critic 给出 Q(s,a): [Ai]，但我们不希望更新actor时把梯度传进critic => detach
        q1 = _to_1d(critic_1(s.edge_attr, s.edge_index, s.select_num))  # [Ai]
        q2 = _to_1d(critic_2(s.edge_attr, s.edge_index, s.select_num))  # [Ai]
        min_q = torch.min(q1, q2).detach()                              # [Ai] 关键：detach

        # (e) E_{a~pi}[minQ]：scalar
        exp_min_q = (probs * min_q).sum()                         # 标量

        # (f) actor loss（与你案例一致）
        # 你的案例：-alpha*entropy - exp_min_q
        # 这里用同样形式：
        actor_loss_i = -alpha_detached * entropy - exp_min_q       # 标量
        per_sample_actor_loss.append(actor_loss_i)

    actor_loss = torch.stack(per_sample_actor_loss).mean()         # 标量
    mean_entropy = torch.stack(per_sample_entropy).mean().detach() # 仅用于日志显示

    actor_optimizer.zero_grad()
    actor_loss.backward()
    total_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), clip_norm)
    print("actor grad norm before clip =", float(total_norm))
    actor_optimizer.step()

    # ------- 2) 更新 alpha（只更新 log_alpha，不更新 actor） -------
    # 经典做法：alpha_loss = mean( alpha * (entropy - target_entropy).detach() )
    # 这里 entropy 是逐样本标量，所以 stack 再算
    entropies = torch.stack(per_sample_entropy)                    # [B]
    target_entropy_t = torch.tensor(target_entropy, device=device, dtype=entropies.dtype)

    alpha = log_alpha.exp()                             # 不detach，需要对log_alpha求梯度
    alpha_loss = (alpha * (entropies - target_entropy_t).detach()).mean()

    log_alpha_optimizer.zero_grad()
    alpha_loss.backward()
    log_alpha_optimizer.step()

    return actor_loss.item(), alpha_loss.item(), mean_entropy.item()


def append_integer_to_file(file_path, number):
    """
    将整数写入指定txt文件末尾，另起一行。

    :param file_path: str，目标文件的路径
    :param number: int，要写入的整数
    """
    try:
        with open(file_path, 'a') as file:
            file.write(f"\n{number}")
    except Exception as e:
        print(f"写入文件时出错: {e}")


def pd_norm(normal_1, normal_2):
    """
    计算两个法向量的平均法向，处理退化情况。
    normal_1, normal_2: np.ndarray, shape=(3,)
    """
    ZERO_VEC = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    # normal_1 退化或 normal_2 非退化
    if np.array_equal(normal_1, ZERO_VEC) or not np.array_equal(normal_2, ZERO_VEC):
        aver_norm = normal_2
    # normal_2 退化或 normal_1 非退化
    elif np.array_equal(normal_2, ZERO_VEC) or not np.array_equal(normal_1, ZERO_VEC):
        aver_norm = normal_1
    # 两个法向量都非退化
    elif not np.array_equal(normal_1, ZERO_VEC) or not np.array_equal(normal_2, ZERO_VEC):
        print('操作前状态出现退化单元！')
        aver_norm = average_normals(normal_1, normal_2)
    else:
        aver_norm = average_normals(normal_1, normal_2)

    return aver_norm