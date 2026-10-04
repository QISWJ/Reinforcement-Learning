import openmesh as om
import heapq
from typing import Set
import math
from typing import List
import numpy as np
from GCN import *
from torch_geometric.data import Data, Batch
import pickle
import os
import random
import time
def get_two_ring_neighbors_exclude_features(mesh, vertex_handle, features_vertex) :
    """
    获取指定顶点的 2 环邻域所有顶点索引，并去除在 features_vertex 中的顶点。

    参数:
    - mesh: OpenMesh 网格对象
    - vertex_handle: 顶点句柄（指定的起始顶点）
    - features_vertex: 需要排除的顶点索引列表

    返回:
    - list: 处理后的 2 环邻域顶点索引列表（去除 features_vertex 中的顶点）
    """
    # 使用集合去除重复顶点
    two_ring_neighbors = set()

    # 获取顶点的邻居（1环）
    first_ring_neighbors = list(mesh.vv(vertex_handle))

    # 1环顶点加入2环邻域集合
    for vh in first_ring_neighbors:
        two_ring_neighbors.add(vh.idx())

        # 获取邻居的邻居（2环）
        second_ring_neighbors = list(mesh.vv(vh))
        for second_vh in second_ring_neighbors:
            two_ring_neighbors.add(second_vh.idx())

    # 将 features_vertex 中的元素从 two_ring_neighbors 中去除
    two_ring_neighbors = two_ring_neighbors - set(features_vertex)

    # 返回 2环邻域顶点的索引列表
    return list(two_ring_neighbors)

def get_multi_vertices_two_ring_neighbors_exclude_features(mesh, vertex_indices, features_vertex):
    """
    输入多个顶点索引，获取这些顶点的非 features_vertex 二环邻域所有顶点，并合并去重返回。

    参数:
    - mesh: OpenMesh 网格对象
    - vertex_indices: list，存储多个顶点索引
    - features_vertex: list，需排除的特征顶点索引

    返回:
    - list: 所有输入顶点对应的二环邻域顶点索引（去重后）
    """
    all_neighbors = set()

    for vid in vertex_indices:
        # 跳过非法索引
        if vid < 0 or vid >= mesh.n_vertices():
            continue

        vh = mesh.vertex_handle(vid)

        # 调用你原来的函数
        two_ring_neighbors = get_two_ring_neighbors_exclude_features(
            mesh, vh, features_vertex
        )

        # 合并到总集合中
        all_neighbors.update(two_ring_neighbors)

    return list(all_neighbors)


def get_two_ring_neighbors(mesh: om.PolyMesh, vertex_handle: om.VertexHandle) -> list:
    """
    获取指定顶点的 2 环邻域所有顶点索引，并返回索引列表。

    参数:
    - mesh: OpenMesh 网格对象
    - vertex_handle: 顶点句柄（指定的起始顶点）

    返回:
    - list: 2环邻域内所有顶点的索引列表
    """
    # 使用集合去除重复顶点
    two_ring_neighbors = set()

    # 获取顶点的邻居（1环）
    first_ring_neighbors = list(mesh.vv(vertex_handle))

    # 1环顶点加入2环邻域集合
    for vh in first_ring_neighbors:
        two_ring_neighbors.add(vh.idx())

        # 获取邻居的邻居（2环）
        second_ring_neighbors = list(mesh.vv(vh))
        for second_vh in second_ring_neighbors:
            two_ring_neighbors.add(second_vh.idx())

    # 返回 2环邻域顶点的索引列表
    return list(two_ring_neighbors)


#填充每个顶点的度数、不规则性
def compute_vertex_valence_diff(mesh):
    """
    计算每个顶点的度数，并计算度数的变化

    :param mesh: OpenMesh 网格对象
    :param ver_valence: 用于存储每个顶点度数的空列表
    :param ver_valence_diff: 用于存储度数变化的空列表
    """
    # 清空输入数组，确保它们是空的
    ver_valence = []
    ver_valence_diff = []

    # 遍历网格中的每个顶点
    for i in range(mesh.n_vertices()):
        vh = mesh.vertex_handle(i)
        # 获取当前顶点的度数（即该顶点相邻的边的数量）
        degree = mesh.valence(vh)
        ver_valence.append(degree)
        l = ideal_degree_quad(mesh, vh)
        diff = abs(l - degree)
        ver_valence_diff.append(diff)


    return ver_valence, ver_valence_diff







#获取特征边与特征顶点
def find_feature_edges_and_vertices(mesh):
    features_vertex = []
    features_edge = []

    unique_vertices = set()  # 用于避免重复存储顶点
    unique_edges = set()  # 用于避免重复存储边

    for eh in mesh.edges():
        angle = compute_face_normal_angle_in_degrees(mesh, eh)

        # 检查二面角是否满足条件
        if angle > 15.0:
            edge_idx = eh.idx()
            # 如果边是特征边，将索引加入特征边集合
            if edge_idx not in unique_edges:
                features_edge.append(edge_idx)
                unique_edges.add(edge_idx)

            # 将特征边的两个顶点加入特征点集合
            heh = mesh.halfedge_handle(eh, 0)
            vertex1 = mesh.from_vertex_handle(heh).idx()
            vertex2 = mesh.to_vertex_handle(heh).idx()

            if vertex1 not in unique_vertices:
                features_vertex.append(vertex1)
                unique_vertices.add(vertex1)

            if vertex2 not in unique_vertices:
                features_vertex.append(vertex2)
                unique_vertices.add(vertex2)

    return features_vertex, features_edge



def is_strict_interior_face(mesh, fh):
    """
    输入：OpenMesh 的 mesh（全四边形网格），以及一个面句柄 fh
    规则：
      1) 如果该面触碰边界（面上存在边界半边/边界边） -> False
      2) 否则检查该面“一环所有顶点”（面顶点 + 这些顶点的相邻顶点）：
         只要有任意顶点是边界顶点 -> False
      3) 都不是 -> True
    """
    # ---------- (1) 该面是否触碰边界：面上一旦出现 boundary halfedge 就认为是“边界面” ----------
    # mesh.fh(fh) : face-halfedge iterator
    for heh in mesh.fh(fh):
        if mesh.is_boundary(heh):   # 该半边在边界上
            return False

    # ---------- (2) 收集该面的一环顶点：面顶点 + 面顶点的相邻顶点 ----------
    one_ring_verts: Set[int] = set()

    # 面本身的顶点
    face_verts = []
    for vh in mesh.fv(fh):          # face-vertex iterator
        face_verts.append(vh)
        one_ring_verts.add(vh.idx())

    # 面顶点的相邻顶点（1-ring）
    for vh in face_verts:
        # mesh.vv(vh) : vertex-vertex iterator
        for vnb in mesh.vv(vh):
            one_ring_verts.add(vnb.idx())

    # ---------- (3) 一环顶点中有任意边界顶点就返回 False ----------
    for vid in one_ring_verts:
        vh = mesh.vertex_handle(vid)
        if mesh.is_boundary(vh):
            return False

    return True


#计算基于面的局部区域的得分
def fh_2_ring_score(mesh,fh, ver_valence_diff):
    ring0 = []
    ids = set()
    for vh in mesh.fv(fh):  # face-vertex iterator
        ring0.append(vh)
        ids.add(vh.idx())

    # ---------- 1-ring：ring0 的邻接顶点 ----------
    ring1_ids: Set[int] = set()
    for vh in ring0:
        for nb in mesh.vv(vh):  # vertex-vertex iterator
            nid = nb.idx()
            if nid not in ids:
                ring1_ids.add(nid)
            ids.add(nid)

    # ---------- 2-ring：ring1 的邻接顶点 ----------
    for nid in ring1_ids:
        vh1 = mesh.vertex_handle(nid)
        for nb2 in mesh.vv(vh1):
            ids.add(nb2.idx())

    score = 0
    for vhi in ids:
        score_vh = ver_valence_diff[vhi]

        score += score_vh

    return score



def get_face_two_ring_vertex_score_sum(mesh, fh, ver_valence_diff):
    """
    计算面 fh 的二环邻域内所有顶点得分之和。

    二环定义（按面邻接）：
        - 第0环：面 fh 自身
        - 第1环：与 fh 共边的相邻面
        - 第2环：与第1环面共边的相邻面

    参数
    ----
    mesh : openmesh.TriMesh 或 openmesh.PolyMesh
        OpenMesh 构建的网格对象
    fh : openmesh.FaceHandle
        面句柄
    ver_valence_diff : list[float] 或 list[int]
        顶点得分数组，其中 ver_valence_diff[i] 表示顶点索引 i 的得分

    返回
    ----
    float
        面 fh 的二环邻域内所有顶点得分之和
    """
    if not fh.is_valid():
        raise ValueError("输入的面句柄 fh 无效。")

    visited_face_ids = set()
    visited_face_handles = []

    frontier = [fh]
    visited_face_ids.add(fh.idx())
    visited_face_handles.append(fh)

    # 扩展到二环
    for _ in range(2):
        next_frontier = []

        for cur_fh in frontier:
            for heh in mesh.fh(cur_fh):
                # 不要直接用 opposite_face_handle(heh)
                opp_heh = mesh.opposite_halfedge_handle(heh)
                nfh = mesh.face_handle(opp_heh)

                # 边界 halfedge 的对侧 face 无效
                if not nfh.is_valid():
                    continue

                nid = nfh.idx()
                if nid not in visited_face_ids:
                    visited_face_ids.add(nid)
                    visited_face_handles.append(nfh)
                    next_frontier.append(nfh)

        frontier = next_frontier
        if not frontier:
            break

    # 收集二环邻域内所有顶点索引
    vertex_ids = set()
    for cur_fh in visited_face_handles:
        for vh in mesh.fv(cur_fh):
            vid = vh.idx()
            vertex_ids.add(vid)

    # 计算得分和
    score_sum = 0.0
    for vid in vertex_ids:
        if vid < 0 or vid >= len(ver_valence_diff):
            raise IndexError(
                f"顶点索引 {vid} 超出 ver_valence_diff 长度范围 [0, {len(ver_valence_diff)-1}]"
            )
        score_sum += ver_valence_diff[vid]

    return score_sum


#计算边分数，由周围6个顶点abs(理想度数-实际度数），但是对于边的两个对边顶点的分数计算；理想度数-实际度数（因为度数越小，才趋于该顶点分裂)，该函数只适用与面的一环域不存在边界顶点的的边集合
def eh_local_score(mesh, list_, ver_valence_diff):
    score = 0
    for i in list_:
        eh = mesh.edge_handle(i)
        heh1 = mesh.halfedge_handle(eh,0)
        heh2 = mesh.next_halfedge_handle(heh1)
        heh3 = mesh.next_halfedge_handle(heh2)
        heh4 = mesh.next_halfedge_handle(heh3)
        heh5 = mesh.opposite_halfedge_handle(heh1)
        heh6 = mesh.next_halfedge_handle(heh5)
        heh7 = mesh.next_halfedge_handle(heh6)
        heh8 = mesh.next_halfedge_handle(heh7)

        v1 = mesh.to_vertex_handle(heh1)
        v2 = mesh.to_vertex_handle(heh2)
        v3 = mesh.to_vertex_handle(heh3)
        v4 = mesh.to_vertex_handle(heh4)
        v5 = mesh.to_vertex_handle(heh6)
        v6 = mesh.to_vertex_handle(heh7)

        if mesh.valence(v1)==3:
            a_7 = -1
        else:
            v7_idx = halfedge_vh_score_(mesh, heh1, bool_=True)
            v7 = mesh.vertex_handle(v7_idx)
            a_7 = (ideal_degree_quad(mesh,v7)) - (mesh.valence(v7))
        if mesh.valence(v4)==3:
            a_8 = -1
        else:
            v8_idx = halfedge_vh_score_(mesh, heh5, bool_=True)
            v8 = mesh.vertex_handle(v8_idx)
            a_8 = (ideal_degree_quad(mesh,v8)) - (mesh.valence(v8))

        s = ver_valence_diff[v1.idx()] + ver_valence_diff[v2.idx()] + ver_valence_diff[v3.idx()] + ver_valence_diff[v4.idx()] + ver_valence_diff[v5.idx()] + ver_valence_diff[v6.idx()] + a_7 + a_8
        score += s

    return score




def count_in_other(list1, list2):
    s2 = set(list2)             # O(1) 平均查找
    cnt = 0
    for x in list1:
        if x in s2:
            cnt += 1
    return cnt



def is_boundary_face(mesh, fh):
    """
    判断一个面 fh 是否为边界面：
    只要该面至少有一条边界 halfedge，就返回 True，否则返回 False。

    参数
    ----
    mesh : openmesh.TriMesh 或 openmesh.PolyMesh
        OpenMesh 构建的网格对象
    fh : openmesh.FaceHandle
        面句柄

    返回
    ----
    bool
        True  -> 该面至少有一个边界 halfedge
        False -> 该面所有 halfedge 都不是边界
    """
    for heh in mesh.fh(fh):   # 遍历该面的所有 halfedge handle
        if mesh.is_boundary(heh):
            return True
    return False





#计算所有非边界单元的二环局部区域得分
def base_fh_score(mesh, features_edge, ver_valence_diff):

    act_local_eh = []
    fh_score_list = []
    for fh in mesh.faces():
        if is_strict_interior_face(mesh, fh):
            heh1 = mesh.halfedge_handle(fh)
            heh2 = mesh.next_halfedge_handle(heh1)
            heh3 = mesh.next_halfedge_handle(heh2)
            heh4 = mesh.next_halfedge_handle(heh3)

            vh1 = mesh.from_vertex_handle(heh1)
            vh2 = mesh.from_vertex_handle(heh2)
            vh3 = mesh.from_vertex_handle(heh3)
            vh4 = mesh.from_vertex_handle(heh4)

            eh1 = mesh.edge_handle(heh1)
            eh2 = mesh.edge_handle(heh2)
            eh3 = mesh.edge_handle(heh3)
            eh4 = mesh.edge_handle(heh4)

            heh5 = mesh.opposite_halfedge_handle(heh1)
            heh6 = mesh.opposite_halfedge_handle(heh2)
            heh7 = mesh.opposite_halfedge_handle(heh3)
            heh8 = mesh.opposite_halfedge_handle(heh4)

            heh9 = mesh.next_halfedge_handle(heh5)
            heh10 = mesh.next_halfedge_handle(heh9)
            heh11 = mesh.next_halfedge_handle(heh10)

            heh12 = mesh.next_halfedge_handle(heh6)
            heh13 = mesh.next_halfedge_handle(heh12)
            heh14 = mesh.next_halfedge_handle(heh13)

            heh15 = mesh.next_halfedge_handle(heh7)
            heh16 = mesh.next_halfedge_handle(heh15)
            heh17 = mesh.next_halfedge_handle(heh16)

            heh18 = mesh.next_halfedge_handle(heh8)
            heh19 = mesh.next_halfedge_handle(heh18)
            heh20 = mesh.next_halfedge_handle(heh19)

            eh5 = mesh.edge_handle(heh9)
            eh6 = mesh.edge_handle(heh10)
            eh7 = mesh.edge_handle(heh11)
            eh8 = mesh.edge_handle(heh12)
            eh9 = mesh.edge_handle(heh13)
            eh10 = mesh.edge_handle(heh14)
            eh11 = mesh.edge_handle(heh15)
            eh12 = mesh.edge_handle(heh16)
            eh13 = mesh.edge_handle(heh17)
            eh14 = mesh.edge_handle(heh18)
            eh15 = mesh.edge_handle(heh19)
            eh16 = mesh.edge_handle(heh20)


            hl = [eh1.idx(), eh2.idx(), eh3.idx(), eh4.idx(), eh5.idx(), eh6.idx(), eh7.idx(), eh8.idx(),
                                 eh9.idx(), eh10.idx(), eh11.idx(), eh12.idx(), eh13.idx(), eh14.idx(), eh15.idx(), eh16.idx()]

            cout = count_in_other(hl, features_edge)

            if cout<1:

                act_local_eh.append([eh1.idx(), eh2.idx(), eh3.idx(), eh4.idx(), eh5.idx(), eh6.idx(), eh7.idx(), eh8.idx(),
                                     eh9.idx(), eh10.idx(), eh11.idx(), eh12.idx(), eh13.idx(), eh14.idx(), eh15.idx(), eh16.idx()])

                fh_score = eh_local_score(mesh, hl, ver_valence_diff)
                fh_score_list.append(fh_score)

        else:
            continue

    return act_local_eh, fh_score_list

def heh_eh_idx(mesh,heh):
    heh1 = mesh.opposite_halfedge_handle(heh)
    if mesh.is_boundary(heh1):
        #边界半边直接返回3个虚拟半边
        return -1,-1,-1
    else:
        heh2 = mesh.next_halfedge_handle(heh1)
        heh3 = mesh.next_halfedge_handle(heh2)
        heh4 = mesh.next_halfedge_handle(heh3)

        eh2 = mesh.edge_handle(heh2)
        eh3 = mesh.edge_handle(heh3)
        eh4 = mesh.edge_handle(heh4)

        return eh2.idx(), eh3.idx(), eh4.idx()


#计算所有非边界单元的二环局部区域得分
def base_fh_score_1(mesh, features_edge, ver_valence_diff):

    act_local_eh = []
    fh_score_list = []
    for fh in mesh.faces():
        heh1 = mesh.halfedge_handle(fh)
        heh2 = mesh.next_halfedge_handle(heh1)
        heh3 = mesh.next_halfedge_handle(heh2)
        heh4 = mesh.next_halfedge_handle(heh3)

        eh1 = mesh.edge_handle(heh1)
        eh2 = mesh.edge_handle(heh2)
        eh3 = mesh.edge_handle(heh3)
        eh4 = mesh.edge_handle(heh4)
        eh5_idx, eh6_idx, eh7_idx = heh_eh_idx(mesh,heh1)
        eh8_idx, eh9_idx, eh10_idx = heh_eh_idx(mesh, heh2)
        eh11_idx, eh12_idx, eh13_idx = heh_eh_idx(mesh, heh3)
        eh14_idx, eh15_idx, eh16_idx = heh_eh_idx(mesh, heh4)

        hl = [eh1.idx(), eh2.idx(), eh3.idx(), eh4.idx(), eh5_idx, eh6_idx, eh7_idx, eh8_idx,
                  eh9_idx, eh10_idx, eh11_idx, eh12_idx, eh13_idx, eh14_idx, eh15_idx, eh16_idx]
        cout = count_in_other(hl, features_edge)


        fh_score = get_face_two_ring_vertex_score_sum(mesh, fh, ver_valence_diff)
        fh_score_list.append(fh_score)
        act_local_eh.append(hl)

    return act_local_eh, fh_score_list



def get_top_a_values_from_list(list1, list2, a):
    """
    输入：
    - list1: 一维数组，元素用于排序，获取最大值的前a个索引。
    - list2: 二维数组，索引对应 list1 中的元素索引。
    - a: 整数，指定要返回的前 a 个最大元素。

    输出：
    - result: 一维数组，包含 list2 中对应前 a 个最大值的元素（已扁平化为一维）。
    - selected_sum: 被选中的 a 个 list1 元素之和
    """
    # 找到 list1 中最大值的前 a 个索引
    top_a_indices = heapq.nlargest(a, range(len(list1)), key=lambda i: list1[i])

    # 取出对应的 list1 值
    selected_values = [list1[i] for i in top_a_indices]

    # 求和（就是你要的 episode_return 风格的量）
    selected_sum = sum(selected_values)

    # 根据这些索引，从 list2 中提取对应的值并将它们扁平化为一维
    result = [item for i in top_a_indices for item in list2[i]]

    return result, selected_sum



#计算指定顶点的理想度数
def _round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def _angle_deg(u, v) -> float:
    """
    返回两个向量夹角（单位：度），范围 [0, 180]
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return 0.0

    c = float(np.dot(u, v) / (nu * nv))
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


def _face_corner_angle_deg(mesh: om.PolyMesh, fh: om.FaceHandle, vh: om.VertexHandle) -> float:
    """
    计算顶点 vh 在面 fh 内部对应的角度（单位：度）
    """
    vids = [fv.idx() for fv in mesh.fv(fh)]
    n = len(vids)
    if n < 3:
        return 0.0

    try:
        k = vids.index(vh.idx())
    except ValueError:
        return 0.0

    prev_vh = mesh.vertex_handle(vids[(k - 1) % n])
    next_vh = mesh.vertex_handle(vids[(k + 1) % n])

    p0 = np.asarray(mesh.point(vh), dtype=float)
    p_prev = np.asarray(mesh.point(prev_vh), dtype=float)
    p_next = np.asarray(mesh.point(next_vh), dtype=float)

    u = p_prev - p0
    v = p_next - p0

    return _angle_deg(u, v)


def ideal_degree_quad(mesh: om.PolyMesh, vh: om.VertexHandle) -> int:
    """
    全四边形网格顶点理想度数：
      - 内部顶点：d* = 360 / 90 = 4
      - 边界顶点：d* = max(round(theta / 90) + 1, 2)

    其中 theta 不再通过两条边界边直接求夹角，
    而是通过“该边界顶点所有相邻面的角度和”来计算，
    这样可以正确处理凹角（theta > 180°）的情况。
    """
    alpha = 90.0

    # 内部点：直接 4
    if not mesh.is_boundary(vh):
        return 4

    # 边界点：theta = 相邻面的角度和
    theta = 0.0
    for fh in mesh.vf(vh):
        theta += _face_corner_angle_deg(mesh, fh, vh)

    # 异常兜底
    if theta <= 1e-12:
        theta = 180.0

    # 理论上边界点的离散内角应在 (0, 360] 内
    theta = max(0.0, min(theta, 360.0))

    d_star = max(_round_half_up(theta / alpha) + 1, 2)
    return int(d_star)

#找到指定半边做边分裂最优另一个半边所指顶点的索引(
def halfedge_vh_score_(mesh, heh, bool_):
    vh = mesh.to_vertex_handle(heh)
    degree = mesh.valence(vh)
    if degree == 3 or (mesh.is_boundary(vh) and degree == 2):
        if mesh.is_boundary(vh):
            p = -2
            heh_idx = -2
        else:
            p = -1
            heh_idx = -1
    elif degree == 4:
        heh1 = mesh.next_halfedge_handle(heh)
        heh2 = mesh.opposite_halfedge_handle(heh1)

        heh3 = mesh.next_halfedge_handle(heh2)
        vh = mesh.to_vertex_handle(heh3)
        heh4 = mesh.opposite_halfedge_handle(heh3)
        p = vh.idx()
        heh_idx = heh4.idx()
    elif degree == 5:
        heh1 = mesh.next_halfedge_handle(heh)
        heh2 = mesh.opposite_halfedge_handle(heh1)

        heh3 = mesh.next_halfedge_handle(heh2)
        p_1 = mesh.valence(mesh.to_vertex_handle(heh3))

        heh4 = mesh.opposite_halfedge_handle(heh3)
        heh5 = mesh.next_halfedge_handle(heh4)
        p_2 = mesh.valence(mesh.to_vertex_handle(heh5))
        heh6 = mesh.opposite_halfedge_handle(heh5)

        if p_1 > p_2:
            p = mesh.to_vertex_handle(heh5).idx()
            heh_idx = heh6.idx()
        else:
            p = mesh.to_vertex_handle(heh3).idx()
            heh_idx = heh4.idx()

    elif degree == 6:
        heh1 = mesh.next_halfedge_handle(heh)
        heh2 = mesh.opposite_halfedge_handle(heh1)
        heh3 = mesh.next_halfedge_handle(heh2)
        heh4 = mesh.opposite_halfedge_handle(heh3)
        heh5 = mesh.next_halfedge_handle(heh4)
        p = mesh.to_vertex_handle(heh5).idx()
        heh6 = mesh.opposite_halfedge_handle(heh5)
        heh_idx = heh6.idx()


    elif degree==7:
        heh1 = mesh.next_halfedge_handle(heh)
        heh2 = mesh.opposite_halfedge_handle(heh1)
        heh3 = mesh.next_halfedge_handle(heh2)
        heh4 = mesh.opposite_halfedge_handle(heh3)
        heh5 = mesh.next_halfedge_handle(heh4)
        heh6 = mesh.opposite_halfedge_handle(heh5)
        p_1 = mesh.valence(mesh.from_vertex_handle(heh6))
        heh7 = mesh.next_halfedge_handle(heh6)
        heh8 = mesh.opposite_halfedge_handle(heh7)
        p_2 = mesh.valence(mesh.from_vertex_handle(heh8))
        if p_1 > p_2:
            p = mesh.from_vertex_handle(heh8).idx()
            heh_idx = heh8.idx()
        else:
            p = mesh.from_vertex_handle(heh6).idx()
            heh_idx = heh6.idx()


    else:
        #print('网格顶点没有清理干净！存在度数大于6的顶点')
        p= -1
        heh_idx = -1

    if bool_:

        return p
    else:
        return heh_idx


def calculate_edge_length(mesh, eh):
    """
    计算指定边句柄 eh 的长度。

    参数:
    - mesh: OpenMesh 的 PolyMesh 对象。
    - eh: 边句柄（EdgeHandle），表示要计算的边。

    返回:
    - 边的长度。
    """
    # 获取边的两个半边句柄
    heh1 = mesh.halfedge_handle(eh, 0)
    heh2 = mesh.halfedge_handle(eh, 1)

    # 获取半边对应的顶点句柄
    v1 = mesh.to_vertex_handle(heh1)
    v2 = mesh.to_vertex_handle(heh2)

    # 获取顶点的坐标
    p1 = mesh.point(v1)  # 获取顶点v1的坐标 (x1, y1, z1)
    p2 = mesh.point(v2)  # 获取顶点v2的坐标 (x2, y2, z2)

    # 计算两点之间的欧几里得距离（边的长度）
    length = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2)

    return length


#计算边长比
def Calculate_side_length_ratio(mesh, eh):
    l = calculate_edge_length(mesh, eh)
    if not mesh.is_boundary(eh):
        heh1 = mesh.halfedge_handle(eh, 0)
        heh2 = mesh.opposite_halfedge_handle(heh1)
        heh3 = mesh.next_halfedge_handle(heh1)
        heh4 = mesh.next_halfedge_handle(heh3)
        heh5 = mesh.next_halfedge_handle(heh4)
        heh6 = mesh.next_halfedge_handle(heh2)
        heh7 = mesh.next_halfedge_handle(heh6)
        heh8 = mesh.next_halfedge_handle(heh7)

        eh1 = mesh.edge_handle(heh3)
        eh2 = mesh.edge_handle(heh5)
        eh3 = mesh.edge_handle(heh8)
        eh4 = mesh.edge_handle(heh6)
        eh5 = mesh.edge_handle(heh4)
        eh6 = mesh.edge_handle(heh7)


        l1 = calculate_edge_length(mesh, eh1)
        l2 = calculate_edge_length(mesh, eh2)
        l3 = calculate_edge_length(mesh, eh3)
        l4 = calculate_edge_length(mesh, eh4)
        l5 = calculate_edge_length(mesh, eh5)
        l6 = calculate_edge_length(mesh, eh6)
        if l==0:
            a_1=0
            a_2=0
            a_3=0
            a_4=0
        else:

            a_1 = (l1+l2)/(2*l)
            a_2 = (l3 + l4) / (2 * l)
            a_3 = l5/l
            a_4 = l6 / l

    else:
        heh = mesh.halfedge_handle(eh, 0)
        heh_ = mesh.opposite_halfedge_handle(heh)
        if not mesh.is_boundary(heh):
            heh1 = heh
        else:
            heh1 = heh_
        heh3 = mesh.next_halfedge_handle(heh1)
        heh4 = mesh.next_halfedge_handle(heh3)
        heh5 = mesh.next_halfedge_handle(heh4)

        eh1 = mesh.edge_handle(heh3)
        eh5 = mesh.edge_handle(heh4)
        eh2 = mesh.edge_handle(heh5)

        l1 = calculate_edge_length(mesh, eh1)
        l5 = calculate_edge_length(mesh, eh5)
        l2 = calculate_edge_length(mesh, eh2)
        if l==0:
            a_1=0
            a_2=0
            a_3=0
            a_4=0
        else:

            if heh1 == heh:
                a_1 = (l1 + l2) / (2 * l)
                a_2 = 0
                a_3 = l5 / l
                a_4 = 0

            else:
                a_1 = 0
                a_2 = (l1 + l2) / (2 * l)
                a_3 = 0
                a_4 = l5 / l




    return a_1, a_2, a_3, a_4



#计算输出边的平均二面角
def compute_face_normal_angle_in_degrees(mesh, eh):
    if mesh.is_boundary(eh):
        angle = 90
    else:
        heh1 = mesh.halfedge_handle(eh, 0)
        heh2 = mesh.halfedge_handle(eh, 1)

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

        a_1 = calculate_dihedral_angle(mesh, v_2, v_1, v_4, v_6)
        a_2 = calculate_dihedral_angle(mesh, v_3, v_1, v_4, v_5)
        angle = (a_1 + a_2) / 2

    return angle


def calculate_dihedral_angle(mesh, vh1, vh2, vh3, vh4):
    """
    二面角：
    - 三角形1 顶点顺序: (vh3, vh2, vh1)
    - 三角形2 顶点顺序: (vh4, vh2, vh3)

    返回：夹角（度），范围 [0, 180]
    """
    # 顶点坐标
    p1 = mesh.point(vh1)
    p2 = mesh.point(vh2)
    p3 = mesh.point(vh3)
    p4 = mesh.point(vh4)

    p1 = np.array([p1[0], p1[1], p1[2]], dtype=float)
    p2 = np.array([p2[0], p2[1], p2[2]], dtype=float)
    p3 = np.array([p3[0], p3[1], p3[2]], dtype=float)
    p4 = np.array([p4[0], p4[1], p4[2]], dtype=float)

    # ---- 三角形1: (vh3, vh2, vh1) ----
    # n1 = (p2 - p3) x (p1 - p3)
    v1 = p2 - p3
    v2 = p1 - p3
    n1 = np.cross(v1, v2)

    # ---- 三角形2: (vh4, vh2, vh3) ----
    # n2 = (p2 - p4) x (p3 - p4)
    v3 = p2 - p4
    v4 = p3 - p4
    n2 = np.cross(v3, v4)

    # 法向长度
    norm_n1 = np.linalg.norm(n1)
    norm_n2 = np.linalg.norm(n2)
    if norm_n1 < 1e-15 or norm_n2 < 1e-15:
        # 退化三角形，无法稳定算角度
        return 0.0

    # 夹角
    cos_theta = np.dot(n1, n2) / (norm_n1 * norm_n2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    return float(np.degrees(angle_rad))


def calculate_angle_between_lines(mesh, vh1, vh2, vh3):
    """
    计算由 vh1, vh2, vh3 构成的两条线段 vh1-vh2 和 vh2-vh3 之间的夹角。

    参数:
    - mesh: OpenMesh 对象
    - vh1, vh2, vh3: 顶点句柄

    返回:
    - 夹角（单位：度）
    """
    # 获取顶点的坐标
    p1 = mesh.point(vh1)
    p2 = mesh.point(vh2)
    p3 = mesh.point(vh3)

    # 计算向量 v1 = p2 - p1 和 v2 = p3 - p2
    v1 = np.array([p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]])

    # 计算点积
    dot_product = np.dot(v1, v2)

    # 计算向量的模（长度）
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    # 计算夹角的余弦值，避免除以0
    cos_theta = dot_product / (norm_v1 * norm_v2)

    # 数值稳定性：确保 cos_theta 在 [-1, 1] 范围内
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # 计算夹角（弧度）
    angle_rad = np.arccos(cos_theta)

    # 转换为度
    angle_deg = np.degrees(angle_rad)

    return angle_deg

def compute_face_normal_newell(mesh: om.PolyMesh, fh: om.FaceHandle) -> np.ndarray:
    pts = [np.array(mesh.point(vh), dtype=float) for vh in mesh.fv(fh)]
    n = len(pts)
    if n < 3:
        return np.array([0.0, 0.0, 0.0], dtype=float)

    nx, ny, nz = 0.0, 0.0, 0.0
    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i + 1) % n]
        nx += (p0[1] - p1[1]) * (p0[2] + p1[2])
        ny += (p0[2] - p1[2]) * (p0[0] + p1[0])
        nz += (p0[0] - p1[0]) * (p0[1] + p1[1])

    normal = np.array([nx, ny, nz], dtype=float)
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0], dtype=float)

    return normal / norm


def compute_face_vertex_inner_angle(mesh: om.PolyMesh,
                                    fh: om.FaceHandle,
                                    vh: om.VertexHandle) -> float:
    """
    返回顶点 vh 在面 fh 上的真实内角（单位：度）
    支持凹四边形，返回范围 [0, 360)
    """
    face_vhs = [fv for fv in mesh.fv(fh)]
    n = len(face_vhs)
    if n < 3:
        raise ValueError("该面顶点数小于3，无法计算内角。")

    idx = -1
    for i, fv in enumerate(face_vhs):
        if fv.idx() == vh.idx():
            idx = i
            break
    if idx == -1:
        raise ValueError("输入顶点不在该面上。")

    vh_prev = face_vhs[(idx - 1) % n]
    vh_next = face_vhs[(idx + 1) % n]

    p_prev = np.array(mesh.point(vh_prev), dtype=float)
    p_curr = np.array(mesh.point(vh), dtype=float)
    p_next = np.array(mesh.point(vh_next), dtype=float)

    e_in = p_curr - p_prev
    e_out = p_next - p_curr

    norm_in = np.linalg.norm(e_in)
    norm_out = np.linalg.norm(e_out)
    if norm_in < 1e-12 or norm_out < 1e-12:
        return 0.0

    e_in /= norm_in
    e_out /= norm_out

    normal = compute_face_normal_newell(mesh, fh)
    if np.linalg.norm(normal) < 1e-12:
        return 0.0

    cross_io = np.cross(e_in, e_out)
    turn = math.atan2(np.dot(cross_io, normal), np.dot(e_in, e_out))

    angle = math.pi - turn
    if angle < 0.0:
        angle += 2.0 * math.pi
    elif angle >= 2.0 * math.pi:
        angle -= 2.0 * math.pi

    return math.degrees(angle)





def laplacian_smooth_selected_vertices(
    mesh: om.PolyMesh,
    vertex_indices: List[int],
    iterations: int = 10,
    lam: float = 1.0
) -> om.PolyMesh:
    """
    对给定顶点索引集合做拉普拉斯平滑（只移动这些点）。

    参数:
      mesh: openmesh.PolyMesh
      vertex_indices: 一维 list[int]，顶点索引（vh.idx()）
      iterations: 迭代次数
      lam: 平滑步长/强度，0~1（建议 0.1~1.0）

    返回:
      平滑后的 mesh（原地修改后返回 mesh 本身）
    """
    if iterations <= 0 or lam <= 0.0:
        return mesh

    # 去重，避免重复计算
    idx_set = set(int(i) for i in vertex_indices)

    for _ in range(iterations):
        new_pos = {}  # {vidx: np.array([x,y,z])}

        for vidx in idx_set:
            vh = om.VertexHandle(vidx)
            if not vh.is_valid():
                continue

            p = np.asarray(mesh.point(vh), dtype=np.float64)  # (3,)

            # 邻居顶点
            neigh_pts = []
            for vhn in mesh.vv(vh):  # vhn 是 VertexHandle
                neigh_pts.append(np.asarray(mesh.point(vhn), dtype=np.float64))

            if len(neigh_pts) == 0:
                continue

            avg = np.mean(np.stack(neigh_pts, axis=0), axis=0)  # (3,)
            p_new = (1.0 - lam) * p + lam * avg
            new_pos[vidx] = p_new

        # 统一写回
        for vidx, p_new in new_pos.items():
            vh = om.VertexHandle(vidx)
            mesh.set_point(vh, p_new)  # p_new 是长度3的数组即可

    return mesh


#边翻转函数，a=0与a=1分别是不同方向的翻转
def flip_quad_halfedge(mesh, he, a):


    fh0 = mesh.face_handle(he)
    opp = mesh.opposite_halfedge_handle(he)
    fh1 = mesh.face_handle(opp)

    # 依 he 提取 1..6
    v1 = mesh.from_vertex_handle(he)                      # 1
    v2 = mesh.to_vertex_handle(he)                        # 2
    he_b = mesh.next_halfedge_handle(he)
    v3 = mesh.to_vertex_handle(he_b)                      # 3
    he_c = mesh.next_halfedge_handle(he_b)
    v4 = mesh.to_vertex_handle(he_c)                      # 4

    he_e = mesh.next_halfedge_handle(opp)
    v5 = mesh.to_vertex_handle(he_e)                      # 5
    he_f = mesh.next_halfedge_handle(he_e)
    v6 = mesh.to_vertex_handle(he_f)                      # 6


    # 删除旧两面（旧对角 (1,2) 后续 GC 清理）
    mesh.delete_face(fh0, False)
    mesh.delete_face(fh1, False)

    # 尝试添加面的辅助：失败则尝试反向顺序以适配局部定向
    def add_face_try(verts):
        fh = mesh.add_face(verts)
        return fh if fh.is_valid() else mesh.add_face(list(reversed(verts)))

    if a == 0:

        mesh.add_face([v5, v6, v2, v3])
        mesh.add_face([v5, v3, v4, v1])

    else:
        mesh.add_face([v5, v6, v4, v1])
        mesh.add_face([v4, v6, v2, v3])

    mesh.garbage_collection()


    return mesh



#四边形分裂，根据输入的两个半边
def _ring_incoming_halfedges_ccw(mesh: om.PolyMesh, v_center: om.VertexHandle):
    """
    返回围绕顶点 v_center 的、按 CCW 顺序的“指向 v_center 的半边”（vih）列表。
    """
    return [heh for heh in mesh.vih(v_center)]


def _index_in_ring(ring, target_heh: om.HalfedgeHandle) -> int:
    for i, heh in enumerate(ring):
        if heh.idx() == target_heh.idx():
            return i
    return -1


def _walk_arc_from_a_to_b(mesh: om.PolyMesh, heh_a: om.HalfedgeHandle, heh_b: om.HalfedgeHandle):
    """
    基于 vih 环来取弧段：从 heh_a 沿 CCW 顺序走到 heh_b（含 a，不含 b）。
    要求 heh_a 与 heh_b 的 to-vertex 相同。
    若传入的是从 v2 发出的半边（voh），本函数会自动取其 opposite 并继续。
    """
    v2_a = mesh.to_vertex_handle(heh_a)
    v2_b = mesh.to_vertex_handle(heh_b)
    if v2_a.idx() != v2_b.idx():
        raise ValueError("heh_a 与 heh_b 必须具有相同的 to-vertex（同一顶点）。")
    v2 = v2_a

    ring = _ring_incoming_halfedges_ccw(mesh, v2)
    if len(ring) == 0:
        raise RuntimeError("在该顶点没有 incoming 半边；检查网格或句柄类型（需 PolyMesh 的 vih）。")

    ia = _index_in_ring(ring, heh_a)
    ib = _index_in_ring(ring, heh_b)

    # 从 ia 沿 CCW 前进到 ib（含 a，不含 b）
    arc = []
    n = len(ring)
    i = ia
    guard = 0
    while i != ib:
        arc.append(ring[i])
        i = (i + 1) % n
        guard += 1
        if guard > n + 5:
            raise RuntimeError("vih 环推进异常（可能非流形/重复句柄），请检查网格。")
    return arc


def _faces_incident_on_arc(mesh, arc):
    """
    给定围绕某顶点的一条半边弧，收集与这些半边“同向侧”的相邻面（去重）。
    """
    faces = []
    seen = set()
    for heh in arc:
        fh = mesh.face_handle(heh)
        if fh.is_valid() and fh.idx() not in seen:
            faces.append(fh)
            seen.add(fh.idx())
    return faces


def _replace_vertex_in_face_loop(mesh, fh, old_vh, new_vh):
    """
    读取面 fh 的顶点环，把 old_vh 替换为 new_vh：
    实现为：删除旧面 -> 按相同顺序重建新面（仅顶点处替换）。
    返回新面句柄。
    """
    vloop = [vh for vh in mesh.fv(fh)]

    replaced = False
    for i, vh in enumerate(vloop):
        if vh.idx() == old_vh.idx():
            vloop[i] = new_vh
            replaced = True
    if not replaced:
        # 不应该发生（除非Topo不一致）
        return fh

    mesh.delete_face(fh, False)  # 延迟GC
    new_fh = mesh.add_face(vloop)
    if not new_fh.is_valid():
        raise RuntimeError("重建面失败；可能是顶点顺序/拓扑不一致。")
    return new_fh


def _centroid_of_vertices(mesh, vhs):
    if not vhs:
        return np.zeros(3, dtype=float)
    pts = np.array([mesh.point(vh) for vh in vhs], dtype=float)
    return np.mean(pts, axis=0)




def quad_vertex_split_by_two_halfedges(mesh, heh_a, heh_b, smoothing_alpha: float = 0.5):
    """
    在同一 to-vertex 处，由半边 heh_a 与 heh_b 将顶点 v2 分裂为 v4、v5，
    并在两侧替换；可选地在 (v1, v4, v3, v5) 之间添加一个新四边形。

    参数
    ----
    mesh : om.PolyMesh
    heh_a, heh_b : om.HalfedgeHandle
        要求 mesh.to_vertex_handle(heh_a) == mesh.to_vertex_handle(heh_b)
        且网格为封闭四边形网格（无边界/非流形）
    add_quad : bool
        是否添加新四边形面 (v1, v4, v3, v5)
    smoothing_alpha : float in [0,1]
        新点坐标 = (1 - alpha) * v2 + alpha * 侧邻居质心

    返回
    ----
    mesh : om.PolyMesh
        原位修改后的同一网格对象
    """

    v2 = mesh.to_vertex_handle(heh_a)
    if not v2.is_valid():
        raise ValueError("heh_a 的 to-vertex 非法。")
    if mesh.to_vertex_handle(heh_b).idx() != v2.idx():
        raise ValueError("heh_a 与 heh_b 的 to-vertex 必须相同（同一顶点）。")


    v1 = mesh.from_vertex_handle(heh_a)  # heh_a 的起点
    v3 = mesh.from_vertex_handle(heh_b)  # heh_b 的起点
    p2 = np.array(mesh.point(v2), dtype=float)

    # 1) 用 vih 环切分两侧弧段
    arc_a2b = _walk_arc_from_a_to_b(mesh, heh_a, heh_b)
    arc_b2a = _walk_arc_from_a_to_b(mesh, heh_b, heh_a)

    faces_side_A = _faces_incident_on_arc(mesh, arc_a2b)
    faces_side_B = _faces_incident_on_arc(mesh, arc_b2a)

    # 2) 新增两个顶点 v4、v5（初值放在 p2），稍后设置平滑位置
    v4 = mesh.add_vertex(p2)
    v5 = mesh.add_vertex(p2)

    # 3) 计算两侧邻居（不包含 v2），用于平滑
    def gather_side_neighbors(faces):
        nbrs = set()
        for fh in faces:
            for vh in mesh.fv(fh):
                if vh.idx() != v2.idx():
                    nbrs.add(vh.idx())
        return [om.VertexHandle(i) for i in nbrs]

    nbr_A = gather_side_neighbors(faces_side_A)
    nbr_B = gather_side_neighbors(faces_side_B)

    # 侧质心（若无邻居则退化为 p2）
    p4_c = _centroid_of_vertices(mesh, nbr_A) if len(nbr_A) > 0 else p2
    p5_c = _centroid_of_vertices(mesh, nbr_B) if len(nbr_B) > 0 else p2

    alpha = float(np.clip(smoothing_alpha, 0.0, 1.0))
    p4 = (1.0 - alpha) * p2 + alpha * p4_c
    p5 = (1.0 - alpha) * p2 + alpha * p5_c
    mesh.set_point(v4, p4)
    mesh.set_point(v5, p5)

    # 4) 在两侧所有相关面中，把 v2 替换为 v4 / v5（延迟GC）
    for fh in faces_side_A:
        if fh.is_valid():
            _replace_vertex_in_face_loop(mesh, fh, v2, v4)
    for fh in faces_side_B:
        if fh.is_valid():
            _replace_vertex_in_face_loop(mesh, fh, v2, v5)

    # 5) 删除旧顶点 v2（延迟GC）
    mesh.delete_vertex(v2, False)

    # 6) 新增四边形面，连接 [v1, v4, v3, v5]

    fh_new = mesh.add_face([v1, v4, v3, v5])
    if not fh_new.is_valid():
        # 若失败，尝试反序（有些数据要求一致的环方向）
        fh_new = mesh.add_face([v1, v5, v3, v4])
        if not fh_new.is_valid():
            raise RuntimeError("新增四边形失败；请检查局部顶点顺序与法向一致性。")

    # 7) 垃圾回收，修复句柄
    mesh.garbage_collection()

    return mesh

#根据输入半边执行最优分裂动作
def split_edge_base_heh(mesh, heh):
    heh1_idx = halfedge_vh_score_(mesh, heh, bool_=False)
    heh1 = mesh.halfedge_handle(heh1_idx)
    #heh2 = mesh.opposite_halfedge_handle(heh1)
    mesh = quad_vertex_split_by_two_halfedges(mesh, heh, heh1)
    return mesh


def _is_valid(h):
    return hasattr(h, "idx") and h.idx() != -1
def _add_face_oriented(mesh, vlist):
    fh = mesh.add_face(vlist)
    if _is_valid(fh):
        return fh
    fh = mesh.add_face(list(reversed(vlist)))
    if _is_valid(fh):
        return fh
    raise RuntimeError("add_face 在两种方向下均失败：可能产生非流形或退化多边形。")

def _dedup_loop(vlist):
    # 去除连续重复与首尾重复，防止退化
    out = []
    for i, vh in enumerate(vlist):
        if i == 0 or vh.idx() != vlist[i-1].idx():
            out.append(vh)
    if len(out) >= 2 and out[0].idx() == out[-1].idx():
        out.pop()
    return out


def quad_collapse_diagonal(mesh, heh_a, features_vertex):
    if not _is_valid(heh_a):
        raise ValueError("半边 a 句柄无效。")
    fh = mesh.face_handle(heh_a)
    if not _is_valid(fh):
        raise ValueError("半边 a 不属于有效的面（应为内部四边形）。")

    a = heh_a
    b = mesh.next_halfedge_handle(a)
    c = mesh.next_halfedge_handle(b)
    d = mesh.next_halfedge_handle(c)
    if mesh.next_halfedge_handle(d).idx() != a.idx():
        raise RuntimeError("半边环未闭合：a->b->c->d->a 不成立。")

    v1 = mesh.from_vertex_handle(a)
    v2 = mesh.to_vertex_handle(a)
    v3 = mesh.to_vertex_handle(b)
    v4 = mesh.to_vertex_handle(c)
    p2 = mesh.point(v2)
    p4 = mesh.point(v4)

    ring = list(mesh.fv(fh))
    if len(ring) != 4 or len({vh.idx() for vh in ring}) != 4:
        raise ValueError("目标面不是简单四边形（顶点数≠4或存在重复顶点）。")

    p5 = (mesh.point(v2) + mesh.point(v4)) * 0.5

    new_mesh = om.PolyMesh()
    vmap = {}
    for vh in mesh.vertices():
        if vh.idx() in (v2.idx(), v4.idx()):
            continue
        vmap[vh.idx()] = new_mesh.add_vertex(mesh.point(vh))

    v5_new = new_mesh.add_vertex(p5)
    vmap[v2.idx()] = v5_new
    vmap[v4.idx()] = v5_new

    for f in mesh.faces():
        if f.idx() == fh.idx():
            continue
        vs_old = list(mesh.fv(f))
        vs_old = _dedup_loop(vs_old)
        vs_new = [vmap[vh.idx()] for vh in vs_old]
        vs_new = _dedup_loop(vs_new)
        if len(vs_new) < 3:
            raise RuntimeError("塌缩 2/4→5 后出现退化面（顶点数 < 3）。")
        _add_face_oriented(new_mesh, vs_new)

    ring5 = list(new_mesh.vv(v5_new))
    if len(ring5) >= 2:
        acc = new_mesh.point(v5_new) * 0.0
        for nb in ring5:
            acc = acc + new_mesh.point(nb)

        if mesh.is_boundary(v4):
            new_mesh.set_point(v5_new, p4)
        elif mesh.is_boundary(v2):
            new_mesh.set_point(v5_new, p2)
        elif (v2.idx() in features_vertex) and (v4.idx() not in features_vertex):
            new_mesh.set_point(v5_new, p4)
        elif (v4.idx() in features_vertex) and (v2.idx() not in features_vertex):
            new_mesh.set_point(v5_new, p2)
        else:
            new_mesh.set_point(v5_new, acc / float(len(ring5)))

    return new_mesh, v5_new

def copy_polymesh(input_mesh):
    """
    正确复制 OpenMesh PolyMesh 对象（适配 pyopenmesh 官方 API）

    参数:
        input_mesh: om.PolyMesh - 待复制的网格

    返回:
        om.PolyMesh - 复制后的新网格
    """
    # 1. 提取顶点坐标 + 顶点句柄索引映射（用 idx() 转整数作为键）
    points = []
    vh_idx_map = {}  # 键：vertex handle 的 idx（整数），值：新网格的顶点索引
    new_vert_idx = 0
    for vh in input_mesh.vertices():
        if vh.is_valid():  # 句柄自身的 is_valid() 方法
            vh_int = vh.idx()
            vh_idx_map[vh_int] = new_vert_idx
            points.append(input_mesh.point(vh))
            new_vert_idx += 1
    points = np.array(points, dtype=np.float64)  # 形状：(n_verts, 3)

    # 2. 提取面的顶点索引（二维数组，每行对应一个四边形面）
    face_vertex_indices = []
    for fh in input_mesh.faces():
        if fh.is_valid():  # 检查面句柄有效性
            face_verts_int = []
            for vh in input_mesh.fv(fh):
                if vh.is_valid():
                    face_verts_int.append(vh_idx_map[vh.idx()])
            # 仅保留四边形面（全四边形网格约束）
            if len(face_verts_int) == 4:
                face_vertex_indices.append(face_verts_int)

    # 转换为二维 numpy 数组（shape=(n_faces, 4)，符合 pyopenmesh 要求）
    if face_vertex_indices:
        face_vertex_indices = np.array(face_vertex_indices, dtype=np.int32)
    else:
        face_vertex_indices = np.array([], dtype=np.int32).reshape(0, 4)  # 空网格兼容

    # 3. 构建新网格
    new_mesh = om.PolyMesh(points, face_vertex_indices)
    return new_mesh


def eliminate_non_boundary_valence2_vertices(input_mesh):
    """
    消除全四边形网格中所有非边界的度数为2的顶点，保持网格为纯四边形

    参数:
        input_mesh: openmesh.PolyMesh 对象（全四边形网格）

    返回:
        om.PolyMesh: 处理后的网格，内部无度数2顶点
    """
    # 1. 正确复制网格，避免修改原对象
    mesh = copy_polymesh(input_mesh)

    # 循环处理：直到无目标顶点（删除一个可能产生新的2度顶点）
    while True:
        # 2. 收集所有需要处理的顶点（非边界 + 度数=2）
        target_vertices = []
        for vh in mesh.vertices():
            if vh.is_valid() and not mesh.is_boundary(vh) and mesh.valence(vh) == 2:
                target_vertices.append(vh)

        # 如果没有度数为 2 的非边界顶点，退出循环
        if not target_vertices:
            break

        # 3. 逐个处理2度内部顶点
        vh = target_vertices[0]  # 只处理第一个找到的度数为2的内部顶点
        if not vh.is_valid():  # 检查顶点句柄有效性
            continue

        # 获取相邻顶点（确保只有2个有效顶点）
        adj_vertices = []
        for v in mesh.vv(vh):
            if v.is_valid():
                adj_vertices.append(v)
                if len(adj_vertices) > 2:
                    break  # 超过2个直接跳过
        if len(adj_vertices) != 2:
            continue
        v1, v2 = adj_vertices[0], adj_vertices[1]

        # 获取关联的两个四边形面（确保只有2个有效面）
        adj_faces = []
        for f in mesh.vf(vh):
            if f.is_valid():
                adj_faces.append(f)
                if len(adj_faces) > 2:
                    break  # 超过2个直接跳过
        if len(adj_faces) != 2:
            continue
        f1, f2 = adj_faces[0], adj_faces[1]

        # 提取两个面的顶点，剔除当前2度顶点
        f1_verts = [v for v in mesh.fv(f1) if v.is_valid() and v != vh]
        f2_verts = [v for v in mesh.fv(f2) if v.is_valid() and v != vh]

        # ---------------- 核心修复：VertexHandle 不可哈希的去重逻辑 ----------------
        # 步骤1：将 VertexHandle 转换为整数索引（可哈希）
        f1_verts_idx = [v.idx() for v in f1_verts]
        f2_verts_idx = [v.idx() for v in f2_verts]
        # 步骤2：对整数索引去重
        unique_verts_idx = list(set(f1_verts_idx + f2_verts_idx))
        # 步骤3：根据整数索引找回对应的 VertexHandle，并校验有效性
        new_face_verts = []
        for idx in unique_verts_idx:
            vert_handle = mesh.vertex_handle(idx)  # 通过索引获取句柄
            if vert_handle.is_valid():
                new_face_verts.append(vert_handle)
        # -----------------------------------------------------------------------------

        # 确保新面是4个顶点（纯四边形）
        if len(new_face_verts) != 4:
            continue

        # ---------------- 拓扑操作：删除旧元素 ----------------
        # 先删除两个旧面（避免删除顶点时拓扑冲突）
        if f1.is_valid():
            mesh.delete_face(f1)
        if f2.is_valid():
            mesh.delete_face(f2)

        # 安全获取并删除旧边（v-v1 和 v-v2）
        def get_edge_between(mesh_obj, vh_a, vh_b):
            """安全获取两个顶点之间的边句柄"""
            if not (vh_a.is_valid() and vh_b.is_valid()):
                return None
            for he in mesh_obj.voh(vh_a):
                if he.is_valid() and mesh_obj.to_vertex_handle(he) == vh_b:
                    return mesh_obj.edge_handle(he)
            return None

        e1 = get_edge_between(mesh, vh, v1)
        e2 = get_edge_between(mesh, vh, v2)

        if e1 and e1.is_valid():
            mesh.delete_edge(e1)
        if e2 and e2.is_valid():
            mesh.delete_edge(e2)

        # 删除2度顶点
        if vh.is_valid():
            mesh.delete_vertex(vh)

        # 清理拓扑垃圾（OpenMesh 必需步骤）
        mesh.garbage_collection()

        # ---------------- 拓扑操作：添加新面 ----------------
        # 顶点顺时针排序（保证面的环序正确，避免法线反向）
        def sort_verts_clockwise(verts):
            coords = np.array([mesh.point(v) for v in verts])
            centroid = np.mean(coords, axis=0)
            angles = np.arctan2(coords[:, 1] - centroid[1], coords[:, 0] - centroid[0])
            sorted_idx = np.argsort(angles)
            return [verts[i] for i in sorted_idx]

        sorted_verts = sort_verts_clockwise(new_face_verts)
        # 添加新的四边形面
        mesh.add_face(sorted_verts)

    # 最终清理垃圾，确保网格状态正常
    mesh.garbage_collection()
    return mesh

def Preprocessing_act(mesh):

    while True:
        if  not is_degree_2_ver_(mesh):
            break

        for i in range(mesh.n_vertices()):
            vh  = mesh.vertex_handle(i)
            if vh.is_valid() and not mesh.is_boundary(vh) and mesh.valence(vh) == 2:
                for heh in mesh.vih(vh):
                    heh_ = mesh.prev_halfedge_handle(heh)
                    vh_ = mesh.from_vertex_handle(heh_)
                    ffeatures_vertex = [vh_.idx()]
                    mesh, vh5 = quad_collapse_diagonal(mesh, heh, ffeatures_vertex)
                    break
                break



    #mesh = eliminate_non_boundary_valence2_vertices(mesh)

    return mesh

def is_degree_2_ver_(mesh):
    for i in range(mesh.n_vertices()):
        vh = mesh.vertex_handle(i)
        degree = mesh.valence(vh)
        if degree == 2 and (not mesh.is_boundary(vh)):
            return True
    return False

def Temporary_features_vertex(mesh, heh):
    heh1 = mesh.next_halfedge_handle(heh)
    vh1 = mesh.to_vertex_handle(heh1)
    return [vh1.idx()]


def process_degree_2_(mesh, vh):
    for eh in mesh.ve(vh):
        heh = mesh.halfedge_handle(eh, 0)
        tem_features_vertex = Temporary_features_vertex(mesh, heh)
        mesh, v5 = quad_collapse_diagonal(mesh, heh, tem_features_vertex)
        break

    return mesh



def triangle_unit_normal(mesh: om.PolyMesh,
                         vh1: om.VertexHandle,
                         vh2: om.VertexHandle,
                         vh3: om.VertexHandle,
                         eps: float = 1e-12) -> np.ndarray:
    """
    计算由 (vh1, vh2, vh3) 构成的三角形的单位法向量（按右手定则）。
    法向方向由顶点顺序决定：n = (p2 - p1) × (p3 - p1)。

    返回:
      np.ndarray shape (3,) 的单位法向量。
    异常:
      若三点共线/退化（法向长度过小），抛 ValueError。
    """
    p1 = np.asarray(mesh.point(vh1), dtype=np.float64)
    p2 = np.asarray(mesh.point(vh2), dtype=np.float64)
    p3 = np.asarray(mesh.point(vh3), dtype=np.float64)

    v1 = p2 - p1
    v2 = p3 - p1
    n = np.cross(v1, v2)

    norm = np.linalg.norm(n)
    if norm < eps:
        #raise ValueError("三角形退化（共线或面积接近0），无法计算单位法向量。")
        return np.array([0,0,0])

    return n / norm


#计算两个向量组之间的最大夹角
def compute_max_angle(face_normals, act_face_normals):
    """
    计算两个法向量数组之间的最大夹角。

    :param face_normals: 原始法向量列表，每个法向量为 numpy 数组（3D 向量）
    :param act_face_normals: 另一个法向量列表，格式与 face_normals 相同
    :return: 返回最大夹角（单位：度）
    """
    # 检查输入数组大小是否一致
    if len(face_normals) != len(act_face_normals):
        raise ValueError("Input arrays must have the same size")

    max_angle = 0.0  # 最大夹角，单位为度

    # 遍历每一对法向量，计算夹角
    for normal1, normal2 in zip(face_normals, act_face_normals):
        # 计算两个法向量的点积
        dot_product = np.dot(normal1, normal2)

        # 限制点积的范围在 [-1, 1] 之间，以避免浮点误差
        dot_product = np.clip(dot_product, -1.0, 1.0)

        # 计算夹角的余弦值并转换为弧度
        angle_rad = np.arccos(dot_product)

        # 转换为度
        angle_deg = np.degrees(angle_rad)

        # 更新最大夹角
        max_angle = max(max_angle, angle_deg)

    return max_angle


#将多维list数组中数值为a的替换为b
def replace_in_nested_list(x, a, b):
    if isinstance(x, list):
        return [replace_in_nested_list(v, a, b) for v in x]
    return b if x == a else x

def contains_zero_row(list_of_arrays):
    target_array = np.array([0, 0, 0])
    return any(np.array_equal(row, target_array) for row in list_of_arrays)


def eh_len_safe(mesh):
    for eh in mesh.edges():
        lenth = calculate_edge_length(mesh, eh)
        if lenth==0:
            return False
    return True


def _compute_face_normal_newell_raw(mesh: om.PolyMesh, fh: om.FaceHandle) -> np.ndarray:
    """
    用 Newell 法计算面的未归一化法向量。
    若面退化，该向量模长会接近 0。
    """
    pts = [np.array(mesh.point(vh), dtype=float) for vh in mesh.fv(fh)]
    n = len(pts)

    if n < 3:
        return np.array([0.0, 0.0, 0.0], dtype=float)

    nx, ny, nz = 0.0, 0.0, 0.0
    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i + 1) % n]

        nx += (p0[1] - p1[1]) * (p0[2] + p1[2])
        ny += (p0[2] - p1[2]) * (p0[0] + p1[0])
        nz += (p0[0] - p1[0]) * (p0[1] + p1[1])

    return np.array([nx, ny, nz], dtype=float)


def check_mesh_no_degenerate_faces(mesh: om.PolyMesh, eps: float = 1e-12) -> bool:
    """
    检查 mesh 中是否存在退化面。

    参数
    ----
    mesh : om.PolyMesh
        OpenMesh 网格对象
    eps : float
        判定退化的容差

    返回
    ----
    bool
        若 mesh 中不存在退化面，返回 True；
        若存在任意一个退化面，返回 False。
    """
    for fh in mesh.faces():
        face_vhs = [vh for vh in mesh.fv(fh)]
        face_vids = [vh.idx() for vh in face_vhs]

        # 1. 顶点数不足
        if len(face_vids) < 3:
            return False

        # 2. 存在重复顶点
        if len(set(face_vids)) < len(face_vids):
            return False

        # 3. 存在零长度边（相邻点重合或极近）
        pts = [np.array(mesh.point(vh), dtype=float) for vh in face_vhs]
        n = len(pts)
        for i in range(n):
            edge_vec = pts[(i + 1) % n] - pts[i]
            if np.linalg.norm(edge_vec) < eps:
                return False

        # 4. 面面积接近 0（Newell 法向量模长接近 0）
        normal = _compute_face_normal_newell_raw(mesh, fh)
        if np.linalg.norm(normal) < eps:
            return False

    return True



def random_select_integers(a, b):
    """
    从 1 到 b（包含 b）中随机选择 a 个不重复的整数，并返回 list

    参数:
        a (int): 需要选择的整数个数
        b (int): 取值范围上限（范围为 1~b）

    返回:
        list: 长度为 a 的随机整数列表
    """
    if a < 0 or b < 1:
        raise ValueError("a 必须 >= 0，b 必须 >= 1")
    if a > b:
        raise ValueError("当不允许重复选择时，a 不能大于 b")

    return random.sample(range(1, b + 1), a)
