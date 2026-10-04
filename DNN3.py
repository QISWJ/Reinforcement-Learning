import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv


# =========================
# Utils
# =========================
def _as_long_tensor(x, device):
    if torch.is_tensor(x):
        return x.to(device=device, dtype=torch.long)
    return torch.tensor(x, device=device, dtype=torch.long)


def _check_index_array(index_array: torch.Tensor, num_nodes: int, name: str = "index_array"):
    """
    允许 index_array 中出现 -1：
      -1 : 虚拟边 / 占位符
      >=0: 合法真实索引
    """
    if index_array.numel() == 0:
        raise ValueError(f"{name} 为空：候选集合 K=0，会导致后续维度异常。")

    bad_mask = (index_array < -1) | (index_array >= num_nodes)
    if bad_mask.any():
        bad_vals = index_array[bad_mask].detach().cpu().tolist()
        raise IndexError(
            f"{name} 越界：合法范围为 -1 或 [0, {num_nodes - 1}]，"
            f"但检测到非法值 {bad_vals}。"
        )


def _gather_candidates_with_virtual_zero(
    x: torch.Tensor,          # [N, C]
    index_array: torch.Tensor,
    name: str = "index_array",
):
    """
    根据 index_array 提取候选特征：
      - index >= 0: 提取真实特征
      - index == -1: 该候选位置填充为 0 特征（虚拟边）

    返回：
      x_sel [K, C]
    """
    if index_array.numel() == 0:
        raise ValueError(f"{name} 为空：候选集合 K=0，会导致后续维度异常。")

    bad_mask = (index_array < -1) | (index_array >= x.size(0))
    if bad_mask.any():
        bad_vals = index_array[bad_mask].detach().cpu().tolist()
        raise IndexError(
            f"{name} 存在非法索引，只允许 -1 或 [0, {x.size(0) - 1}]，"
            f"但检测到: {bad_vals}"
        )

    K = index_array.size(0)
    C = x.size(1)

    # 默认全部初始化为 0，虚拟边天然就是 0 特征
    x_sel = torch.zeros(K, C, device=x.device, dtype=x.dtype)

    # 只对真实边进行 gather
    valid_mask = index_array >= 0
    if valid_mask.any():
        valid_index = index_array[valid_mask]
        x_sel[valid_mask] = x.index_select(0, valid_index)

    return x_sel


def make_mlp(
    in_dim: int,
    hidden_dims: list,
    out_dim: int,
    dropout: float = 0.1,
    act_layer=nn.GELU,
    *,
    activate_out: bool = False,
    dropout_out: bool = False,
) -> nn.Sequential:
    """
    更规范的 MLP：
    - 中间层：Linear -> Act -> Dropout
    - 输出层：Linear（默认不接激活/Dropout，适合 logits / Q head）
    """
    layers = []
    last = in_dim
    for hd in hidden_dims:
        layers += [nn.Linear(last, hd), act_layer(), nn.Dropout(dropout)]
        last = hd

    layers.append(nn.Linear(last, out_dim))
    if activate_out:
        layers.append(act_layer())
    if dropout_out:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


# =========================
# Actor (Discrete SAC logits)
# =========================
class ActorGIN(nn.Module):
    """
    Edge-based Actor（边作为“节点”）：
      x_edge [E, in_dim]
        -> encoder [E, hidden]
        -> 3-layer GIN (+res + per-layer LN) [E, hidden]
        -> concat[x_init, x_enc, x_gin] [E, in_dim + 2*hidden]
        -> fuse [E, hidden]
        -> select K candidates
        -> head -> logits [K, action_dim]
        -> flatten [K*action_dim]
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        gin_layers: int = 3,
        dropout: float = 0.1,
        action_dim: int = 6,
        train_eps: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.gin_layers = gin_layers
        self.dropout = dropout
        self.action_dim = action_dim

        # 1) encoder: in_dim -> hidden_dim
        self.encoder = make_mlp(
            in_dim=in_dim,
            hidden_dims=[hidden_dim // 2],
            out_dim=hidden_dim,
            dropout=dropout,
            activate_out=True,
            dropout_out=True,
        )

        # 2) GIN stack
        # - 每层一个独立 LayerNorm
        # - GIN 内部 MLP 不再使用 dropout，避免重复正则
        self.gin_convs = nn.ModuleList()
        self.lns = nn.ModuleList()
        for _ in range(gin_layers):
            gin_mlp = make_mlp(
                in_dim=hidden_dim,
                hidden_dims=[hidden_dim],
                out_dim=hidden_dim,
                dropout=0.0,
                activate_out=False,
                dropout_out=False,
            )
            self.gin_convs.append(GINConv(nn=gin_mlp, train_eps=train_eps))
            self.lns.append(nn.LayerNorm(hidden_dim))

        # 3) fuse: (in_dim + 2*hidden_dim) -> hidden_dim
        self.fuse = make_mlp(
            in_dim=in_dim + 2 * hidden_dim,
            hidden_dims=[hidden_dim],
            out_dim=hidden_dim,
            dropout=dropout,
            activate_out=True,
            dropout_out=True,
        )

        # 4) head: hidden_dim -> action_dim
        # 输出 logits，末层不激活/不dropout
        self.head = make_mlp(
            in_dim=hidden_dim,
            hidden_dims=[hidden_dim // 2],
            out_dim=action_dim,
            dropout=dropout,
            activate_out=False,
            dropout_out=False,
        )

    def forward(
        self,
        x_edge: torch.Tensor,          # [E, in_dim]
        edge_index: torch.Tensor,      # [2, M]
        index_array,                   # [K]，允许 -1
        *,
        return_matrix: bool = False,   # True: return [K, action_dim]
        stabilize: bool = True,        # True: 行内减 max 做数值稳定
    ):
        device = x_edge.device
        edge_index = edge_index.to(device=device, dtype=torch.long)
        index_array = _as_long_tensor(index_array, device=device)
        _check_index_array(index_array, num_nodes=x_edge.size(0), name="index_array(actor)")

        # --- x_init ---
        x_init = x_edge  # [E, in_dim]

        # --- encoder ---
        x_enc = self.encoder(x_init)  # [E, hidden]

        # --- GIN stack (+res + per-layer LN) ---
        x_gin = x_enc
        for i, conv in enumerate(self.gin_convs):
            res = x_gin
            x_gin = conv(x_gin, edge_index)           # [E, hidden]
            x_gin = F.gelu(x_gin)
            x_gin = F.dropout(x_gin, p=self.dropout, training=self.training)
            x_gin = self.lns[i](x_gin + res)

        # --- fuse ---
        x_cat = torch.cat([x_init, x_enc, x_gin], dim=-1)  # [E, in_dim + 2*hidden]
        x_new = self.fuse(x_cat)                           # [E, hidden]

        # --- select candidates ---
        # 若 index_array 中有 -1，则对应候选边特征置为 0
        x_sel = _gather_candidates_with_virtual_zero(
            x_new, index_array, name="index_array(actor)"
        )  # [K, hidden]

        # --- logits ---
        logits = self.head(x_sel)  # [K, action_dim]

        # 行内数值稳定，不额外做 mask，由外部统一动作掩码处理
        if stabilize:
            logits = logits - logits.max(dim=-1, keepdim=True).values

        if return_matrix:
            return logits  # [K, action_dim]
        return logits.reshape(-1)  # [K * action_dim]


# =========================
# Critic (Discrete SAC Q-values)
# =========================
class CriticGIN(nn.Module):
    """
    Discrete SAC Critic：对候选节点输出每个动作的 Q 值
      x [N, in_dim]
        -> encoder [N, hidden]
        -> 3-layer GIN (+res + per-layer LN) [N, hidden]
        -> concat[x_init, x_enc, x_gin] [N, in_dim + 2*hidden]
        -> fuse [N, fuse_hidden]
        -> select K candidates
        -> q_head -> Q [K, action_dim]
        -> flatten [K*action_dim]
    """
    def __init__(
        self,
        in_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        gin_layers: int = 3,
        dropout: float = 0.1,
        train_eps: bool = True,
        fuse_hidden: int = 256,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gin_layers = gin_layers
        self.dropout = dropout
        self.fuse_hidden = fuse_hidden

        # 1) encoder: in_dim -> hidden_dim
        self.encoder = make_mlp(
            in_dim=in_dim,
            hidden_dims=[hidden_dim // 2],
            out_dim=hidden_dim,
            dropout=dropout,
            activate_out=True,
            dropout_out=True,
        )

        # 2) GIN stack
        # - 每层一个独立 LayerNorm
        # - GIN 内部 MLP 不再使用 dropout
        self.gin_convs = nn.ModuleList()
        self.lns = nn.ModuleList()
        for _ in range(gin_layers):
            gin_mlp = make_mlp(
                in_dim=hidden_dim,
                hidden_dims=[hidden_dim],
                out_dim=hidden_dim,
                dropout=0.0,
                activate_out=False,
                dropout_out=False,
            )
            self.gin_convs.append(GINConv(nn=gin_mlp, train_eps=train_eps))
            self.lns.append(nn.LayerNorm(hidden_dim))

        # 3) fuse: (in_dim + 2*hidden_dim) -> fuse_hidden
        self.fuse = make_mlp(
            in_dim=in_dim + 2 * hidden_dim,
            hidden_dims=[hidden_dim],
            out_dim=fuse_hidden,
            dropout=dropout,
            activate_out=True,
            dropout_out=True,
        )

        # 4) q head: fuse_hidden -> action_dim
        # 输出 Q 值，末层不激活/不dropout
        self.q_head = make_mlp(
            in_dim=fuse_hidden,
            hidden_dims=[fuse_hidden],
            out_dim=action_dim,
            dropout=dropout,
            activate_out=False,
            dropout_out=False,
        )

    def forward(
        self,
        x: torch.Tensor,              # [N, in_dim]
        edge_index: torch.Tensor,     # [2, M]
        index_array,                  # [K]，允许 -1
        *,
        return_matrix: bool = False,  # True: return [K, action_dim]
    ):
        device = x.device
        edge_index = edge_index.to(device=device, dtype=torch.long)
        index_array = _as_long_tensor(index_array, device=device)
        _check_index_array(index_array, num_nodes=x.size(0), name="index_array(critic)")

        # --- x_init / encoder ---
        x_init = x
        x_enc = self.encoder(x_init)

        # --- GIN stack (+res + per-layer LN) ---
        x_gin = x_enc
        for i, conv in enumerate(self.gin_convs):
            res = x_gin
            x_gin = conv(x_gin, edge_index)
            x_gin = F.gelu(x_gin)
            x_gin = F.dropout(x_gin, p=self.dropout, training=self.training)
            x_gin = self.lns[i](x_gin + res)

        # --- fuse ---
        x_cat = torch.cat([x_init, x_enc, x_gin], dim=-1)
        x_new = self.fuse(x_cat)

        # --- select candidates ---
        # 若 index_array 中有 -1，则对应候选边特征置为 0
        x_sel = _gather_candidates_with_virtual_zero(
            x_new, index_array, name="index_array(critic)"
        )

        # --- q values ---
        q = self.q_head(x_sel)  # [K, action_dim]

        if return_matrix:
            return q
        return q.reshape(-1)