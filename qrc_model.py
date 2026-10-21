#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QRC 模型架构（独立版，v3）

二维玻色-哈伯德量子储备池 + 频率分带多池 + 四类相干/关联读出 + 线性分类读出。

本文件只包含**模型本身**：没有任何实验流程、数据加载、日志或绘图代码。
数值行为与定稿实验（储备池计算2D频分digits.py）逐位一致。

用法
----
    import numpy as np
    from qrc_model import QRCModel

    X = np.random.rand(100, 64)          # 8×8 图像展平，取值 [0,1]
    y = np.random.randint(0, 10, 100)

    model = QRCModel()                   # 3 池 / 2 玻色子 / T=10 / 16 快照
    F = model.transform(X)               # (100, 39936) 量子特征
    model.fit(X, y)                      # StandardScaler + Ridge
    acc = model.score(X, y)

结构
----
    输入层    8×8 → FFT 三频带 → 每带归一化
    池        64 格点、2 玻色子、Fock 2080 维、幺正演化 T=10、16 快照
    读出      密度 1024 / 二阶关联 4096 / |G| 4096 / ReG 4096（每池）
    分类      StandardScaler → RidgeClassifier
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

__all__ = [
    'build_fock_basis',
    'BoseHubbardReservoir',
    'build_coherence_ops',
    'phase_readout',
    'freq_bands',
    'normalize_band',
    'QRCModel',
]

# ============================================================
# 1. Fock 基
# ============================================================

def build_fock_basis(L: int, N: int) -> list[tuple]:
    """所有满足 Σ nᵢ = N 的占据数配置；维度 C(L+N-1, N)。"""
    basis: list[tuple] = []

    def rec(remaining: int, current: list) -> None:
        if len(current) == L:
            if remaining == 0:
                basis.append(tuple(current))
            return
        for n in range(remaining + 1):
            rec(remaining - n, current + [n])

    rec(N, [])
    return basis


# ============================================================
# 2. 池：二维玻色-哈伯德
# ============================================================

class BoseHubbardReservoir:
    """单个储备池。

    H = -t Σ⟨ij⟩ (b̂ᵢ†b̂ⱼ + h.c.) + Σᵢ εᵢ n̂ᵢ + (U/2) Σᵢ n̂ᵢ(n̂ᵢ-1)

    - 初态：中心格点双占据
    - 输入：ε = ε_base + input_strength·(2x-1)
    - 演化：总时长 t_evolve，分 n_snapshots 个快照、每快照 n_substeps 个子步
    """

    def __init__(self, L_x: int = 8, L_y: int = 8, n_particles: int = 2,
                 t: float = 1.0, U: float = 0.25, input_strength: float = 1.0,
                 t_evolve: float = 10.0, n_snapshots: int = 16, n_substeps: int = 1,
                 epsilon_scale: float = 0.1, seed: int = 42):
        for name, val in (('t_evolve', t_evolve), ('n_snapshots', n_snapshots),
                          ('n_substeps', n_substeps), ('L_x', L_x), ('L_y', L_y)):
            if val <= 0:
                raise ValueError(f'{name} 必须为正，收到 {val!r}')
        if n_particles < 0:
            raise ValueError(f'n_particles 不能为负，收到 {n_particles!r}')

        self.L_x, self.L_y = L_x, L_y
        self.L = L_x * L_y
        self.N = n_particles
        self.t = t
        self.U = U
        self.input_strength = input_strength
        self.t_evolve = t_evolve
        self.n_snapshots = n_snapshots
        self.n_substeps = int(n_substeps)

        self.basis = build_fock_basis(self.L, self.N)
        self.dim = len(self.basis)
        self.idx = {b: i for i, b in enumerate(self.basis)}

        self.epsilon_base = epsilon_scale * np.random.default_rng(seed).standard_normal(self.L)
        self.psi0 = self._initial_state()
        self.pairs = self._build_pairs()
        self._precompute()

    # ---------- 构造成分 ----------

    def _build_pairs(self) -> list[tuple]:
        """2D 最近邻对（开放边界）。"""
        pairs = []
        for x in range(self.L_x):
            for y in range(self.L_y):
                k = x * self.L_y + y
                if x + 1 < self.L_x:
                    pairs.append((k, (x + 1) * self.L_y + y))
                if y + 1 < self.L_y:
                    pairs.append((k, x * self.L_y + (y + 1)))
        return pairs

    def _initial_state(self) -> np.ndarray:
        cx, cy = self.L_x // 2, self.L_y // 2
        k = cx * self.L_y + cy
        config = [0] * self.L
        config[k] = self.N
        psi = np.zeros(self.dim, dtype=complex)
        psi[self.idx[tuple(config)]] = 1.0
        return psi

    def _precompute(self) -> None:
        """与输入无关的部分一次性算好：跳跃矩阵、U 对角、占据数矩阵、关联系数矩阵。"""
        dim = self.dim
        rows, cols, vals = [], [], []
        self._diag_U = np.zeros(dim)
        self._N_mat = np.array(self.basis, dtype=float)          # (dim, L)

        # 二阶关联：⟨nᵢnⱼ⟩ = probs @ NN_mat，NN_mat[config, i*L+j] = nᵢ·nⱼ
        nn_rows, nn_cols, nn_vals = [], [], []
        for i, config in enumerate(self.basis):
            occ = [(k, n) for k, n in enumerate(config) if n > 0]
            for k1, n1 in occ:
                for k2, n2 in occ:
                    nn_rows.append(i)
                    nn_cols.append(k1 * self.L + k2)
                    nn_vals.append(float(n1 * n2))
        self._NN_mat = sp.csr_matrix((nn_vals, (nn_rows, nn_cols)),
                                     shape=(dim, self.L * self.L))

        # 跳跃 + U 对角
        for i, config in enumerate(self.basis):
            for k in range(self.L):
                self._diag_U[i] += 0.5 * self.U * config[k] * (config[k] - 1)
            for k, k2 in self.pairs:
                if config[k2] > 0:
                    new = list(config)
                    new[k] += 1
                    new[k2] -= 1
                    rows.append(self.idx[tuple(new)])
                    cols.append(i)
                    vals.append(-self.t * np.sqrt(config[k] + 1) * np.sqrt(config[k2]))
                if config[k] > 0:
                    new = list(config)
                    new[k] -= 1
                    new[k2] += 1
                    rows.append(self.idx[tuple(new)])
                    cols.append(i)
                    vals.append(-self.t * np.sqrt(config[k]) * np.sqrt(config[k2] + 1))
        self._H_hop = sp.csr_matrix((vals, (rows, cols)), shape=(dim, dim))

    def _build_H(self, epsilon: np.ndarray) -> sp.csr_matrix:
        """H = 预计算跳跃 + 对角(ε·n + U 部分)。"""
        diag = self._N_mat @ epsilon + self._diag_U
        H = self._H_hop.tolil(copy=True)
        H.setdiag(diag)
        return H.tocsr()

    # ---------- 输入 / 演化 / 读出 ----------

    def encode_input(self, x) -> np.ndarray:
        """输入向量 → 格点在位势扰动（形状 (L,)），取值 [-input_strength, +input_strength]。"""
        x = np.asarray(x, dtype=float).flatten()
        n = len(x)
        if n == self.L:
            V = x.copy()
        elif n > self.L:
            V = np.array([np.mean(c) for c in np.array_split(x, self.L)])
        else:
            V = np.array([x[i % n] for i in range(self.L)])
        return self.input_strength * (2.0 * V - 1.0)

    def evolve(self, V_input: np.ndarray):
        """返回 (snapshots (S, dim), 末态 ψ)。"""
        H = self._build_H(self.epsilon_base + V_input)
        dt = self.t_evolve / (self.n_snapshots * self.n_substeps)
        psi = self.psi0.copy()
        snapshots = np.zeros((self.n_snapshots, self.dim), dtype=complex)
        for k in range(self.n_snapshots):
            for _ in range(self.n_substeps):
                psi = expm_multiply(-1j * H * dt, psi)
            snapshots[k] = psi
        return snapshots, psi

    def readout(self, snapshots: np.ndarray) -> np.ndarray:
        """密度 + 二阶关联（16×16 通道），形状 (S·(L + C²),)。"""
        probs = np.abs(snapshots) ** 2
        n_i_all = probs @ self._N_mat                            # (S, L)
        C = 16
        idx = np.linspace(0, self.L - 1, C, dtype=int)
        corr_all = (probs @ self._NN_mat).reshape(-1, self.L, self.L)
        corr_sub = corr_all[:, idx][:, :, idx]                   # (S, C, C)
        return np.concatenate([n_i_all.flatten(), corr_sub.flatten()])


# ============================================================
# 3. 相干读出
# ============================================================

def build_coherence_ops(basis, idx, L, pairs_ij):
    """预计算 b̂ᵢ†b̂ⱼ 的稀疏矩阵。

    i≠j：⟨n+eᵢ−eⱼ|b̂ᵢ†b̂ⱼ|n⟩ = √(nᵢ+1)·√(nⱼ)
    i=j：b̂ᵢ†b̂ᵢ = n̂ᵢ，对角元为 nᵢ（旧实现误用 √(nᵢ(nᵢ+1))，v3 已修）
    """
    ops = {}
    for (i, j) in pairs_ij:
        rows, cols, vals = [], [], []
        for c_i, config in enumerate(basis):
            nj = config[j]
            if nj == 0:
                continue
            if i == j:
                rows.append(c_i)
                cols.append(c_i)
                vals.append(float(config[i]))
                continue
            new = list(config)
            new[j] -= 1
            new[i] += 1
            rows.append(idx[tuple(new)])
            cols.append(c_i)
            vals.append(np.sqrt(config[i] + 1) * np.sqrt(nj))
        ops[(i, j)] = sp.csr_matrix((vals, (rows, cols)), shape=(len(basis), len(basis)))
    return ops


def phase_readout(snapshots: np.ndarray, ops: dict, pairs_ij: list):
    """返回 (|G_ij| (S, |pairs|), Re G_ij)。G = ⟨ψ|b̂ᵢ†b̂ⱼ|ψ⟩。"""
    S = snapshots.shape[0]
    G = np.empty((S, len(pairs_ij)), dtype=complex)
    for e, key in enumerate(pairs_ij):
        M = ops[key]
        G[:, e] = np.einsum('sd,ds->s', snapshots.conj(), (M @ snapshots.T))
    return np.abs(G), G.real


# ============================================================
# 4. 频率分带
# ============================================================

def freq_bands(img8x8: np.ndarray, edges=(1.2, 2.4)) -> list[np.ndarray]:
    """8×8 图 → (低, 中, 高) 三个频带的空域分量（实部）。"""
    F = np.fft.fftshift(np.fft.fft2(img8x8))
    u = np.fft.fftshift(np.fft.fftfreq(8) * 8)
    U, V = np.meshgrid(u, u, indexing='ij')
    r = np.sqrt(U ** 2 + V ** 2)
    lo, hi = edges
    masks = (r < lo, (r >= lo) & (r < hi), r >= hi)
    return [np.real(np.fft.ifft2(np.fft.ifftshift(F * m))) for m in masks]


def normalize_band(b: np.ndarray) -> np.ndarray:
    lo, hi = b.min(), b.max()
    if hi - lo < 1e-9:
        return np.full_like(b, 0.5)
    return (b - lo) / (hi - lo)


# ============================================================
# 5. 完整模型
# ============================================================

class QRCModel:
    """频率分带多池 QRC。

    transform(X) 返回拼接特征，列顺序 = [各池 密度+关联] + [各池 |G|] + [各池 ReG]，
    与定稿实验的拼接顺序一致（同一池内：16 快照 × 通道）。
    """

    def __init__(self, n_pools: int = 3, n_particles: int = 2,
                 t: float = 1.0, U: float = 0.25, t_evolve: float = 10.0,
                 n_snapshots: int = 16, n_substeps: int = 1,
                 n_channels: int = 16, band_edges=(1.2, 2.4),
                 input_strength: float = 1.0, epsilon_scale: float = 0.1,
                 grid: tuple = (8, 8), seed: int = 42, alpha: float = 300.0):
        if not 1 <= n_pools <= 3:
            raise ValueError('n_pools 只能是 1/2/3（对应低/中/高频带）')
        self.n_pools = n_pools
        self.n_channels = n_channels
        self.band_edges = band_edges
        self.alpha = alpha
        self.seed = seed

        self.pools = [
            BoseHubbardReservoir(
                L_x=grid[0], L_y=grid[1], n_particles=n_particles, t=t, U=U,
                input_strength=input_strength, t_evolve=t_evolve,
                n_snapshots=n_snapshots, n_substeps=n_substeps,
                epsilon_scale=epsilon_scale, seed=seed + 100 + k)
            for k in range(n_pools)
        ]
        L = self.pools[0].L
        idx = np.linspace(0, L - 1, n_channels, dtype=int)
        self.pairs = [(a, b) for a in idx for b in idx]
        self.ops = [build_coherence_ops(p.basis, p.idx, p.L, self.pairs) for p in self.pools]

        self.scaler = None
        self.clf = None

    # ---------- 特征 ----------

    def _band_matrix(self, X: np.ndarray) -> list[np.ndarray]:
        """把每张图拆成三带并归一化，返回 [band0(N,64), band1, band2]。"""
        out = [[], [], []]
        for x in np.asarray(X, dtype=float):
            for i, b in enumerate(freq_bands(x.reshape(8, 8), self.band_edges)):
                out[i].append(normalize_band(b).flatten())
        return [np.array(o) for o in out]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """X: (N, 64) → 特征 (N, n_pools × 13312)。"""
        X = np.asarray(X, dtype=float)
        bands = self._band_matrix(X)
        blocks = [[], [], []]      # [密度+关联, |G|, ReG]
        for k, pool in enumerate(self.pools):
            orig, absg, reg = [], [], []
            for x in bands[k]:
                snaps, _ = pool.evolve(pool.encode_input(x))
                orig.append(pool.readout(snaps))
                a, r = phase_readout(snaps, self.ops[k], self.pairs)
                absg.append(a.flatten())
                reg.append(r.flatten())
            blocks[0].append(np.array(orig))
            blocks[1].append(np.array(absg))
            blocks[2].append(np.array(reg))
        return np.concatenate([
            np.concatenate(blocks[0], axis=1),
            np.concatenate(blocks[1], axis=1),
            np.concatenate(blocks[2], axis=1),
        ], axis=1)

    @property
    def n_features(self) -> int:
        S, L, C = self.pools[0].n_snapshots, self.pools[0].L, self.n_channels
        return self.n_pools * S * (L + 3 * C * C)

    # ---------- 分类读出 ----------

    def fit(self, X: np.ndarray, y: np.ndarray, alpha: float | None = None):
        """StandardScaler + RidgeClassifier；alpha 缺省用构造时的值（应事先在验证集选定）。"""
        from sklearn.linear_model import RidgeClassifier
        from sklearn.preprocessing import StandardScaler

        F = self.transform(X)
        self.scaler = StandardScaler().fit(F)
        self.clf = RidgeClassifier(alpha=self.alpha if alpha is None else alpha).fit(
            self.scaler.transform(F), np.asarray(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(self.scaler.transform(self.transform(X)))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))
