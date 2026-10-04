from main import *
from ph_pmian import *
from pjie import *

import os
import csv
import json
import time
import random
import shutil
import traceback
import gc

import numpy as np
import torch


# ============================================================
# 1. 用户配置区：一般只需要修改这里
# ============================================================

# 待验证例子。若只跑一个，例如只跑 2.obj，可改为 [2]


# 原始待优化 obj 所在目录
INPUT_DIR = r""

# 结果总目录。

OUTPUT_ROOT = r""

# 随机种子
SEEDS = []

# 两种环域
RING_SIZES = [4, 5]

# 两种模板。这个数同时传给原 vail_model_data_1(..., loceal_num)
TEMPLATE_NUMS = [2]

# 两套网络参数：与操作手册的 local_1 / local_2 对应
CHECKPOINTS = {
    2: r"checkpoint.pth",

}

# 每个局部区域的验证回合数
NUM_EPISODES = 100

# 原程序参数
P = 0.2
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
ALPHA_LR = 1e-4

# 顶点邻域优先级
VERTEX_RANKS = (1, 2, 3)

# 完整网格不规则性必须严格下降才接受。
# 当前指标通常是整数，保留极小容差是为了兼容未来浮点指标。
IMPROVEMENT_EPS = 1e-12

# True：若组合中途异常退出，重启脚本后从最后一个“已接受步骤”继续。
# False：每次都删除旧组合结果并从原始 obj 重新开始。
RESUME = True
CASE_IDS = []
# 从指定组合开始批处理。元组顺序：(case_id, seed, template_num, ring_size)
# 当前要求：从 G:\\swj\\case3\\seg\\1\\seed_1\\r_5 开始。
# 如果以后希望从第一个组合开始，改成 None。
START_FROM = (None)

# 每个组合结束/异常后主动释放 Python 与 CUDA 缓存，降低长时间批处理的显存累积风险。
CLEAN_GPU_BETWEEN_RUNS = True

# 每隔多少个 episode 刷新一次 progress.json；1 表示每个 episode 都记录。
PROGRESS_EVERY_EPISODE = 1


# ============================================================
# 2. 基础工具函数
# ============================================================

def set_random_seed(seed):
    """与原 main2_1.py 一致，并尽量保证 CUDA 可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _fix_optimizer_state_device(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                if k == "step":
                    # Adam 默认 capturable=False 时，step 留在 CPU
                    state[k] = v.cpu()
                else:
                    state[k] = v.to(device)


def load_checkpoint(path, device,
                    actor, critic_1, critic_2, target_critic_1, target_critic_2,
                    actor_optimizer, critic_1_optimizer, critic_2_optimizer,
                    log_alpha, log_alpha_optimizer):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 checkpoint: {path}")

    ckpt = torch.load(path, map_location="cpu")

    actor.load_state_dict(ckpt["actor"])
    critic_1.load_state_dict(ckpt["critic_1"])
    critic_2.load_state_dict(ckpt["critic_2"])
    target_critic_1.load_state_dict(ckpt["target_critic_1"])
    target_critic_2.load_state_dict(ckpt["target_critic_2"])

    log_alpha.data.copy_(ckpt["log_alpha"].to(device))

    actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
    critic_1_optimizer.load_state_dict(ckpt["critic_1_optimizer"])
    critic_2_optimizer.load_state_dict(ckpt["critic_2_optimizer"])
    log_alpha_optimizer.load_state_dict(ckpt["log_alpha_optimizer"])

    _fix_optimizer_state_device(actor_optimizer, device)
    _fix_optimizer_state_device(critic_1_optimizer, device)
    _fix_optimizer_state_device(critic_2_optimizer, device)
    _fix_optimizer_state_device(log_alpha_optimizer, device)

    return ckpt


def win_long_path(path):
    """Windows 长路径兼容。"""
    path = os.path.abspath(path)

    if os.name != "nt":
        return path
    if path.startswith("\\\\?\\"):
        return path
    if path.startswith("\\\\"):
        return "\\\\?\\UNC\\" + path[2:]
    return "\\\\?\\" + path


def ensure_dir(path):
    os.makedirs(win_long_path(path), exist_ok=True)


def ensure_parent_dir(file_path):
    parent = os.path.dirname(file_path)
    if parent:
        ensure_dir(parent)


def safe_rmtree(path):
    if os.path.isdir(win_long_path(path)):
        shutil.rmtree(win_long_path(path), ignore_errors=True)


def write_json(path, data):
    """原子写 JSON：避免程序/系统中断时留下半个 state.json。"""
    ensure_parent_dir(path)
    tmp_path = path + ".tmp"
    with open(win_long_path(tmp_path), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(win_long_path(tmp_path), win_long_path(path))


def read_json(path, default=None):
    if not os.path.exists(win_long_path(path)):
        return default
    with open(win_long_path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def append_csv(path, fieldnames, row):
    ensure_parent_dir(path)
    exists = os.path.exists(win_long_path(path))
    with open(win_long_path(path), "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def mesh_irregularity(mesh_or_path):
    """
    使用现有代码中的不规则性定义：sum(ver_valence_diff)。
    与 main2_1 中 score_ / next_score_ 的来源保持一致，
    但这里用于“完整网格”验收。
    """
    if isinstance(mesh_or_path, (str, os.PathLike)):
        mesh = om.read_polymesh(win_long_path(str(mesh_or_path)))
    else:
        mesh = mesh_or_path

    _, ver_valence_diff = compute_vertex_valence_diff(mesh)
    return float(sum(ver_valence_diff))


def is_improved(before, after):
    return after < before - IMPROVEMENT_EPS


def cleanup_runtime():
    """组合之间主动清理，减少长时间连续验证时的内存/显存累积。"""
    gc.collect()
    if CLEAN_GPU_BETWEEN_RUNS and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception as exc:
            print(f"[警告] CUDA 清理失败，但继续批处理: {exc}")


def preflight_check():
    """在创建任何新实验目录前，一次性检查输入和 checkpoint，避免跑到一半才发现路径错误。"""
    errors = []

    for case_id in CASE_IDS:
        p = os.path.join(INPUT_DIR, f"{case_id}.obj")
        if not os.path.exists(win_long_path(p)):
            errors.append(f"case 输入不存在: {p}")

    for template_num in TEMPLATE_NUMS:
        p = CHECKPOINTS.get(template_num)
        if not p:
            errors.append(f"template={template_num} 未配置 checkpoint")
        elif not os.path.exists(win_long_path(p)):
            errors.append(f"checkpoint 不存在: {p}")

    if START_FROM is not None:
        c, s, t, r = START_FROM
        if c not in CASE_IDS:
            errors.append(f"START_FROM 的 case_id={c} 不在 CASE_IDS={CASE_IDS}")
        if s not in SEEDS:
            errors.append(f"START_FROM 的 seed={s} 不在 SEEDS={SEEDS}")
        if t not in TEMPLATE_NUMS:
            errors.append(f"START_FROM 的 template={t} 不在 TEMPLATE_NUMS={TEMPLATE_NUMS}")
        if r not in RING_SIZES:
            errors.append(f"START_FROM 的 ring={r} 不在 RING_SIZES={RING_SIZES}")

    if errors:
        raise RuntimeError("启动前检查失败：\n- " + "\n- ".join(errors))

    print("[启动检查通过] case 输入、checkpoint 与 START_FROM 配置均有效。")


# ============================================================
# 3. 建立网络：每个 seed/template/ring 组合建立一次
# ============================================================

def build_agent(ckpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    actor = ActorGCN(in_dim=23, action_dim=6).to(device)
    critic_1 = CriticGCN(in_dim=23, action_dim=6).to(device)
    critic_2 = CriticGCN(in_dim=23, action_dim=6).to(device)
    target_critic_1 = CriticGCN(in_dim=23, action_dim=6).to(device)
    target_critic_2 = CriticGCN(in_dim=23, action_dim=6).to(device)

    target_critic_1.load_state_dict(critic_1.state_dict())
    target_critic_2.load_state_dict(critic_2.state_dict())

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=ACTOR_LR)
    critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=CRITIC_LR)
    critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=CRITIC_LR)

    log_alpha = torch.tensor(
        np.log(0.01), dtype=torch.float, device=device, requires_grad=True
    )
    log_alpha_optimizer = torch.optim.Adam([log_alpha], lr=ALPHA_LR)

    load_checkpoint(
        ckpt_path, device,
        actor, critic_1, critic_2, target_critic_1, target_critic_2,
        actor_optimizer, critic_1_optimizer, critic_2_optimizer,
        log_alpha, log_alpha_optimizer,
    )

    # 纯验证模式更明确；不会改变你 vail_model_data_1 内部已有逻辑
    actor.eval()
    return actor, device


# ============================================================
# 4. 单轮尝试：等价于 main2_1/main1 中的一次局部优化
# ============================================================

def run_one_attempt(
    current_obj,
    run_dir,
    next_step_index,
    vertex_rank,
    ring_size,
    template_num,
    actor,
    device,
    global_before,
):
    """
    从 current_obj 出发，仅尝试一个指定 vertex_rank 的局部邻域。

    成功：
        - 完整 stitched.obj 不规则性严格下降；
        - trial 文件夹转为 steps/step_xxxx_rank_y；
        - 返回新的 stitched.obj。

    失败：
        - 自动删除 trial；
        - current_obj 不变；
        - 上层自动换 rank=2/3。
    """

    attempt_started = time.time()
    progress_file = os.path.join(run_dir, "progress.json")
    write_json(progress_file, {
        "status": "running_attempt",
        "step_candidate": next_step_index,
        "vertex_rank": vertex_rank,
        "ring_size": ring_size,
        "template_num": template_num,
        "current_episode": 0,
        "total_episodes": NUM_EPISODES,
        "global_before": global_before,
        "current_obj": current_obj,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    trial_name = f"_trial_step_{next_step_index:04d}_rank_{vertex_rank}"
    trial_dir = os.path.join(run_dir, trial_name)
    safe_rmtree(trial_dir)
    ensure_dir(trial_dir)

    attempt_log = {
        "step_candidate": next_step_index,
        "vertex_rank": vertex_rank,
        "ring_size": ring_size,
        "template_num": template_num,
        "global_before": global_before,
        "global_after": None,
        "local_before": None,
        "local_after": None,
        "accepted": False,
        "reason": "",
        "elapsed_seconds": None,
    }

    try:
        print("\n" + "=" * 90)
        print(
            f"[尝试] step={next_step_index}, rank={vertex_rank}, "
            f"ring={ring_size}, template={template_num}, global={global_before}"
        )
        print(f"[输入] {current_obj}")

        # ---------- A. 读取当前完整网格并选择第 rank 差顶点 ----------
        mesh = om.read_polymesh(win_long_path(current_obj))
        _, ver_valence_diff = compute_vertex_valence_diff(mesh)

        scores, target_vid, sorted_list = get_a_ring_vertex_scores(
            mesh, ring_size, ver_valence_diff, vertex_rank
        )

        print(f"[顶点] 评分第 {vertex_rank} 大的顶点索引: {target_vid}")
        print(f"[顶点] 该顶点评分: {scores[target_vid]}")
        print(f"[顶点] 前10名: {sorted_list[:10]}")

        center_vh = mesh.vertex_handle(target_vid)
        patch_face_ids, patch_vertex_ids, boundary_edge_ids = get_quad_a_ring_patch(
            mesh, center_vh, ring_size
        )

        final_boundary_edge_ids, final_patch_face_ids, final_patch_vertex_ids = repair_patch_concavity(
            mesh,
            patch_face_ids=patch_face_ids,
            boundary_edge_ids=boundary_edge_ids,
            min_shared_edges=2,
            require_consecutive=True,
            verbose=True,
        )

        # ---------- B. 切出局部区域 ----------
        selected_path, remaining_path = cut_region_by_openmesh_edge_loop(
            input_obj_path=win_long_path(current_obj),
            loop_edge_indices=final_boundary_edge_ids,
            output_dir_path=win_long_path(trial_dir),
            selected_output_filename="selected.obj",
            remaining_output_filename="remaining.obj",
            pick_region="smaller",
            target_face_idx=None,
            save_debug_json=True,
            verbose=True,
        )

        # 为避免下游函数返回路径格式差异，这里使用确定的输出名
        selected_obj = os.path.join(trial_dir, "selected.obj")
        remaining_obj = os.path.join(trial_dir, "remaining.obj")

        if not os.path.exists(win_long_path(selected_obj)):
            raise FileNotFoundError(f"局部区域文件未生成: {selected_obj}")
        if not os.path.exists(win_long_path(remaining_obj)):
            raise FileNotFoundError(f"剩余区域文件未生成: {remaining_obj}")

        local_before = mesh_irregularity(selected_obj)
        attempt_log["local_before"] = local_before

        # ---------- C. 神经网络验证 ----------
        network_dir = os.path.join(trial_dir, "network")
        ensure_dir(network_dir)

        episode_return_file = os.path.join(network_dir, "return.txt")
        with open(win_long_path(episode_return_file), "w", encoding="utf-8") as f:
            f.write("")

        best_score = 1000
        decision_time_sum = 0.0

        for episode_idx in range(NUM_EPISODES):
            best_score, mesh_episode_return, juece_avg_time = vail_model_data_1(
                selected_obj,
                network_dir,
                P,
                actor,
                device,
                best_score,
                template_num,
            )
            decision_time_sum += juece_avg_time
            append_integer_to_file(episode_return_file, mesh_episode_return)

            if (episode_idx + 1) % PROGRESS_EVERY_EPISODE == 0 or episode_idx + 1 == NUM_EPISODES:
                write_json(progress_file, {
                    "status": "running_attempt",
                    "step_candidate": next_step_index,
                    "vertex_rank": vertex_rank,
                    "ring_size": ring_size,
                    "template_num": template_num,
                    "current_episode": episode_idx + 1,
                    "total_episodes": NUM_EPISODES,
                    "global_before": global_before,
                    "current_obj": current_obj,
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                })

        avg_decision_time = decision_time_sum / NUM_EPISODES
        append_integer_to_file(
            os.path.join(run_dir, "juece_time.txt"), avg_decision_time
        )
        attempt_log["avg_decision_time"] = avg_decision_time

        best_obj = os.path.join(network_dir, "best_score_test.obj")
        if not os.path.exists(win_long_path(best_obj)):
            raise FileNotFoundError(
                "100 回合后未找到 best_score_test.obj，自动判为本次尝试失败"
            )

        # ---------- D. 与原代码相同的平滑处理 ----------
        patch_mesh = om.read_polymesh(win_long_path(best_obj))
        ph_ver = ver_no_bou(patch_mesh)
        list1, list2 = build_face_aabb_list(mesh)

        smoothed_mesh = laplacian_smooth_internal_vertices(
            patch_mesh,
            iterations=10,
            lam=0.3,
            inplace=False,
        )

        # 保留原 main1/main2_1 的调用，以免改变你现有算法行为。
        # 原代码将返回值赋给 patch_mesh，但最终写出的是 smoothed_mesh。
        patch_mesh = project_vertex_aabb(mesh, ph_ver, list1, list2)

        smoothed_obj = os.path.join(network_dir, "smoothed.obj")
        om.write_mesh(win_long_path(smoothed_obj), smoothed_mesh)

        local_after = mesh_irregularity(smoothed_obj)
        attempt_log["local_after"] = local_after

        # ---------- E. 拼接回完整网格 ----------
        stitch_input_dir = os.path.join(trial_dir, "stitch_input")
        ensure_dir(stitch_input_dir)
        shutil.copy2(win_long_path(remaining_obj), win_long_path(stitch_input_dir))
        shutil.copy2(win_long_path(smoothed_obj), win_long_path(stitch_input_dir))

        stitched_obj = os.path.join(trial_dir, "stitched.obj")
        stitch_obj_meshes_from_folder(
            input_folder=win_long_path(stitch_input_dir),
            output_obj_path=win_long_path(stitched_obj),
            tol=1e-8,
            verbose=True,
        )

        if not os.path.exists(win_long_path(stitched_obj)):
            raise FileNotFoundError(f"拼接结果未生成: {stitched_obj}")

        # ---------- F. 用完整网格不规则性做最终验收 ----------
        global_after = mesh_irregularity(stitched_obj)
        attempt_log["global_after"] = global_after

        print(
            f"[验收] 局部: {local_before} -> {local_after}; "
            f"完整网格: {global_before} -> {global_after}"
        )

        if not is_improved(global_before, global_after):
            attempt_log["reason"] = "完整网格不规则性未下降，自动回滚"
            print(
                f"[回滚] rank={vertex_rank} 未改善: "
                f"{global_before} -> {global_after}"
            )
            return None, attempt_log

        # ---------- G. 成功：trial 转成正式 step ----------
        steps_dir = os.path.join(run_dir, "steps")
        ensure_dir(steps_dir)
        final_step_dir = os.path.join(
            steps_dir, f"step_{next_step_index:04d}_rank_{vertex_rank}"
        )
        safe_rmtree(final_step_dir)
        shutil.move(win_long_path(trial_dir), win_long_path(final_step_dir))

        accepted_stitched_obj = os.path.join(final_step_dir, "stitched.obj")
        attempt_log["accepted"] = True
        attempt_log["reason"] = "完整网格不规则性下降，接受本轮"
        attempt_log["accepted_step_dir"] = final_step_dir

        print(
            f"[接受] step={next_step_index}, rank={vertex_rank}, "
            f"global: {global_before} -> {global_after}"
        )
        return accepted_stitched_obj, attempt_log

    except Exception as exc:
        attempt_log["reason"] = f"异常失败并回滚: {type(exc).__name__}: {exc}"
        attempt_log["traceback"] = traceback.format_exc()
        print(f"[异常回滚] rank={vertex_rank}: {exc}")
        print(attempt_log["traceback"])
        return None, attempt_log

    finally:
        attempt_log["elapsed_seconds"] = time.time() - attempt_started
        write_json(progress_file, {
            "status": "attempt_finished" if attempt_log["accepted"] else "attempt_failed_or_rolled_back",
            "step_candidate": next_step_index,
            "vertex_rank": vertex_rank,
            "ring_size": ring_size,
            "template_num": template_num,
            "accepted": attempt_log["accepted"],
            "reason": attempt_log.get("reason", ""),
            "global_before": attempt_log.get("global_before"),
            "global_after": attempt_log.get("global_after"),
            "elapsed_seconds": attempt_log["elapsed_seconds"],
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        # 只有成功时 trial 已被 move；失败/异常则自动删除。
        if not attempt_log["accepted"]:
            safe_rmtree(trial_dir)


# ============================================================
# 5. 单个 case/template/ring/seed 的完整自动优化
# ============================================================

ATTEMPT_FIELDS = [
    "case_id", "seed", "template_num", "ring_size",
    "step_candidate", "vertex_rank",
    "global_before", "global_after", "local_before", "local_after",
    "accepted", "reason", "avg_decision_time", "elapsed_seconds",
]


def run_single_validation(case_id, input_obj, template_num, ring_size, seed):
    set_random_seed(seed)

    run_dir = os.path.join(
        OUTPUT_ROOT,
        f"case{case_id}",
        "seg",
        str(template_num),
        f"seed_{seed}",
        f"r_{ring_size}",
    )

    state_file = os.path.join(run_dir, "state.json")
    result_file = os.path.join(run_dir, "result.json")
    attempts_csv = os.path.join(run_dir, "attempts.csv")

    # 已完成组合直接跳过，便于批量任务断点续跑
    old_result = read_json(result_file)
    if RESUME and old_result and old_result.get("status") == "finished":
        print(f"\n[跳过已完成] case={case_id}, template={template_num}, ring={ring_size}, seed={seed}")
        return old_result

    if not RESUME and os.path.isdir(win_long_path(run_dir)):
        safe_rmtree(run_dir)

    ensure_dir(run_dir)
    ensure_dir(os.path.join(run_dir, "steps"))

    if not os.path.exists(win_long_path(input_obj)):
        raise FileNotFoundError(f"找不到 case 输入网格: {input_obj}")

    # 清理上次崩溃遗留的 trial，正式 accepted step 不动
    for name in os.listdir(win_long_path(run_dir)):
        if name.startswith("_trial_"):
            safe_rmtree(os.path.join(run_dir, name))

    # ---------- 恢复状态 / 从头开始 ----------
    state = read_json(state_file) if RESUME else None

    if state and state.get("current_obj") and os.path.exists(win_long_path(state["current_obj"])):
        current_obj = state["current_obj"]
        current_score = mesh_irregularity(current_obj)  # 重算，避免 state 数据过期
        accepted_steps = int(state.get("accepted_steps", 0))
        initial_score = float(state.get("initial_score", current_score))
        print(
            f"\n[断点恢复] case={case_id}, template={template_num}, ring={ring_size}, seed={seed}, "
            f"accepted_steps={accepted_steps}, score={current_score}"
        )
    else:
        initial_copy = os.path.join(run_dir, "initial.obj")
        shutil.copy2(win_long_path(input_obj), win_long_path(initial_copy))
        current_obj = initial_copy
        initial_score = mesh_irregularity(current_obj)
        current_score = initial_score
        accepted_steps = 0

        write_json(state_file, {
            "case_id": case_id,
            "seed": seed,
            "template_num": template_num,
            "ring_size": ring_size,
            "input_obj": input_obj,
            "initial_score": initial_score,
            "current_obj": current_obj,
            "current_score": current_score,
            "accepted_steps": accepted_steps,
            "status": "running",
        })

    ckpt_path = CHECKPOINTS[template_num]
    actor, device = build_agent(ckpt_path)

    validation_started = time.time()

    print("\n" + "#" * 100)
    print(
        f"开始验证: case={case_id}, seed={seed}, template={template_num}, "
        f"ring={ring_size}, initial_score={initial_score}"
    )
    print("#" * 100)

    # 核心状态机：1失败 -> 2；2失败 -> 3；任意成功 -> 回 1；1/2/3全失败 -> 结束
    while True:
        next_step = accepted_steps + 1
        accepted_in_this_cycle = False

        for vertex_rank in VERTEX_RANKS:
            new_obj, attempt = run_one_attempt(
                current_obj=current_obj,
                run_dir=run_dir,
                next_step_index=next_step,
                vertex_rank=vertex_rank,
                ring_size=ring_size,
                template_num=template_num,
                actor=actor,
                device=device,
                global_before=current_score,
            )

            attempt_row = {
                "case_id": case_id,
                "seed": seed,
                "template_num": template_num,
                "ring_size": ring_size,
                "step_candidate": attempt.get("step_candidate"),
                "vertex_rank": attempt.get("vertex_rank"),
                "global_before": attempt.get("global_before"),
                "global_after": attempt.get("global_after"),
                "local_before": attempt.get("local_before"),
                "local_after": attempt.get("local_after"),
                "accepted": attempt.get("accepted"),
                "reason": attempt.get("reason"),
                "avg_decision_time": attempt.get("avg_decision_time"),
                "elapsed_seconds": attempt.get("elapsed_seconds"),
            }
            append_csv(attempts_csv, ATTEMPT_FIELDS, attempt_row)

            if new_obj is not None:
                # 成功：正式接受这一轮，然后立即重新从 rank=1 开始下一轮
                current_obj = new_obj
                current_score = mesh_irregularity(current_obj)
                accepted_steps += 1
                accepted_in_this_cycle = True

                write_json(state_file, {
                    "case_id": case_id,
                    "seed": seed,
                    "template_num": template_num,
                    "ring_size": ring_size,
                    "input_obj": input_obj,
                    "initial_score": initial_score,
                    "current_obj": current_obj,
                    "current_score": current_score,
                    "accepted_steps": accepted_steps,
                    "last_success_vertex_rank": vertex_rank,
                    "status": "running",
                })
                write_json(os.path.join(run_dir, "progress.json"), {
                    "status": "accepted_step_saved",
                    "accepted_steps": accepted_steps,
                    "last_success_vertex_rank": vertex_rank,
                    "current_score": current_score,
                    "current_obj": current_obj,
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                })
                break

            # 当前 rank 失败，for 循环自动进入 rank=2 或 rank=3

        if accepted_in_this_cycle:
            continue

        # rank 1、2、3 全部失败：达到当前网络/环域/种子的终止条件
        break

    elapsed = time.time() - validation_started

    final_obj = os.path.join(run_dir, "final.obj")
    shutil.copy2(win_long_path(current_obj), win_long_path(final_obj))

    result = {
        "case_id": case_id,
        "seed": seed,
        "template_num": template_num,
        "ring_size": ring_size,
        "checkpoint": ckpt_path,
        "input_obj": input_obj,
        "initial_score": initial_score,
        "final_score": current_score,
        "improvement": initial_score - current_score,
        "accepted_steps": accepted_steps,
        "final_obj": final_obj,
        "elapsed_seconds": elapsed,
        "status": "finished",
        "stop_reason": "rank 1、2、3 均无法继续降低完整网格不规则性",
    }

    write_json(result_file, result)
    write_json(state_file, {
        **result,
        "current_obj": current_obj,
        "current_score": current_score,
    })
    write_json(os.path.join(run_dir, "progress.json"), {
        "status": "finished",
        "accepted_steps": accepted_steps,
        "initial_score": initial_score,
        "final_score": current_score,
        "final_obj": final_obj,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    print("\n" + "-" * 100)
    print(
        f"[当前组合完成] case={case_id}, seed={seed}, template={template_num}, ring={ring_size}\n"
        f"初始不规则性: {initial_score}\n"
        f"最终不规则性: {current_score}\n"
        f"下降量: {initial_score - current_score}\n"
        f"成功优化步数: {accepted_steps}\n"
        f"最终网格: {final_obj}"
    )
    print("-" * 100)

    return result


# ============================================================
# 6. 每个 case 自动跑 12 组；多个 case 自动批处理
# ============================================================

SUMMARY_FIELDS = [
    "case_id", "seed", "template_num", "ring_size",
    "initial_score", "final_score", "improvement",
    "accepted_steps", "elapsed_seconds", "status", "final_obj",
]


def combo_run_dir(case_id, seed, template_num, ring_size):
    return os.path.join(
        OUTPUT_ROOT, f"case{case_id}", "seg", str(template_num), f"seed_{seed}", f"r_{ring_size}"
    )


def save_combo_error(case_id, seed, template_num, ring_size, exc):
    """组合级错误也写 result.json，避免只留下空 steps 文件夹。"""
    run_dir = combo_run_dir(case_id, seed, template_num, ring_size)
    ensure_dir(run_dir)
    state = read_json(os.path.join(run_dir, "state.json"), default={}) or {}
    result = {
        "case_id": case_id,
        "seed": seed,
        "template_num": template_num,
        "ring_size": ring_size,
        "initial_score": state.get("initial_score"),
        "final_score": state.get("current_score"),
        "improvement": None,
        "accepted_steps": state.get("accepted_steps", 0),
        "elapsed_seconds": None,
        "status": "error",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
        "current_obj": state.get("current_obj"),
        "final_obj": "",
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(os.path.join(run_dir, "result.json"), result)
    write_json(os.path.join(run_dir, "progress.json"), {
        "status": "error",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    return result


def main():
    batch_started = time.time()
    batch_summary = os.path.join(OUTPUT_ROOT, "validation_summary.csv")

    preflight_check()
    start_reached = START_FROM is None

    print("=" * 100)
    print("端到端自动验证启动")
    print(f"CASE_IDS       = {CASE_IDS}")
    print(f"SEEDS          = {SEEDS}")
    print(f"RING_SIZES     = {RING_SIZES}")
    print(f"TEMPLATE_NUMS  = {TEMPLATE_NUMS}")
    print(f"START_FROM     = {START_FROM}")
    print(f"每个 case 验证数 = {len(SEEDS) * len(RING_SIZES) * len(TEMPLATE_NUMS)}")
    print(f"全部计划验证数 = {len(CASE_IDS) * len(SEEDS) * len(RING_SIZES) * len(TEMPLATE_NUMS)}")
    print("=" * 100)

    for case_id in CASE_IDS:
        input_obj = os.path.join(INPUT_DIR, f"{case_id}.obj")

        for seed in SEEDS:
            for template_num in TEMPLATE_NUMS:
                for ring_size in RING_SIZES:
                    combo = (case_id, seed, template_num, ring_size)
                    if not start_reached:
                        if combo == START_FROM:
                            start_reached = True
                            print(f"\n[到达指定起点] {combo}，从这里开始运行。")
                        else:
                            print(f"[起点之前跳过] {combo}")
                            continue

                    try:
                        result = run_single_validation(
                            case_id=case_id,
                            input_obj=input_obj,
                            template_num=template_num,
                            ring_size=ring_size,
                            seed=seed,
                        )

                        append_csv(
                            batch_summary,
                            SUMMARY_FIELDS,
                            {k: result.get(k) for k in SUMMARY_FIELDS},
                        )

                    except Exception as exc:
                        # 某一个组合异常不会阻断后续其它组合；同时把错误落盘到该组合 result.json。
                        print("\n" + "!" * 100)
                        print(
                            f"[组合异常] case={case_id}, seed={seed}, "
                            f"template={template_num}, ring={ring_size}: {exc}"
                        )
                        traceback.print_exc()
                        print("!" * 100)

                        error_result = save_combo_error(case_id, seed, template_num, ring_size, exc)
                        append_csv(
                            batch_summary,
                            SUMMARY_FIELDS,
                            {
                                "case_id": case_id,
                                "seed": seed,
                                "template_num": template_num,
                                "ring_size": ring_size,
                                "initial_score": error_result.get("initial_score", ""),
                                "final_score": error_result.get("final_score", ""),
                                "improvement": "",
                                "accepted_steps": error_result.get("accepted_steps", 0),
                                "elapsed_seconds": "",
                                "status": f"error: {type(exc).__name__}: {exc}",
                                "final_obj": "",
                            },
                        )

                    finally:
                        cleanup_runtime()

    total_elapsed = time.time() - batch_started
    print("\n" + "=" * 100)
    print(f"全部批处理结束，总耗时: {total_elapsed:.2f} s")
    print(f"汇总结果: {batch_summary}")
    print("=" * 100)


if __name__ == "__main__":
    main()
