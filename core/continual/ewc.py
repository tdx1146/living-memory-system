"""
活体记忆系统 - 持续学习（continual learning）：EWC 弹性权重巩固
=============================================================

ABC 操作规划 §S4「C 条件触发（EWC 保护）」的实现落点（默认关，env 门控）。
Kirkpatrick et al. 2017 标准做法的对角近似，两个组件：

  1. FisherAccumulator —— 对角 Fisher 信息近似：健康窗口上梯度平方的
     EMA 累积（F = E[g²]）。**关键设计约束**：Fisher 计算必须在健康窗口上
     进行，绝不在崩塌/过渡态计算（否则保护的是坏工作点本身）。
  2. EwcPenalty —— 二次罚项 λ·ΣFᵢ(θ̂ᵢ − θ̂*ᵢ)²，定义在**缩放后的方向内容**
     上（scale-invariant），豁免 A 重锚的整体缩放方向（§S4 红线：C 不能
     抵抗 A 重锚）。

------------------------------------------------------------------------
关键设计约束：C 豁免 A 的重锚缩放方向（scale-invariant）
------------------------------------------------------------------------
A 重锚 = 标量乘 J（整体缩放：只改 ‖J‖_F，不改方向）。若罚项定义在原始
参数上（λ·ΣF(θ−θ*)²），A 重锚会把 ‖θ−θ*‖ 整体放大 → 罚项增长 → C 抵抗
A 的重锚（保护变成阻碍）。因此罚项定义在「缩放后的方向分量」上：

      θ̂   = θ / ‖θ‖_F          （当前 J 的方向）
      θ̂*  = θ* / ‖θ*‖_F        （保护锚点 J* 的方向）
      P(θ) = λ · Σᵢ Fᵢ · (θ̂ᵢ − θ̂*ᵢ)²

对任意标量 c > 0：θ̂(cθ) = θ̂(θ) → **P(cθ) = P(θ)**（scale-invariant）。
A 重锚只改 ‖θ‖ 不改方向 → 罚项不增长 → C 不抵抗 A。

罚项梯度同样缩放豁免（径向分量恒为 0，与 A 的径向重锚正交）：

      ∂P/∂θ = (2λ/‖θ‖) · [ F⊙(θ̂−θ̂*) − ⟨F⊙(θ̂−θ̂*), θ̂⟩·θ̂ ]

（⟨·,·⟩ 为逐元素点积；该项把 F⊙(θ̂−θ̂*) 投影到单位球面切空间，移除径向
分量。验证：⟨∂P/∂θ, θ̂⟩ = 0 —— EWC 力永不沿「整体缩放」方向施加。）

------------------------------------------------------------------------
健康窗口约束（Fisher 绝不在坏工作点计算）
------------------------------------------------------------------------
update(weights, grads, healthy)：healthy=False（崩塌/过渡态）→ 跳过本轮，
Fisher 不更新。吸引子集成点（AttractorNetwork.learn）用 σ 统计判定健康：
valid 且非崩塌（act05 > col_act）且非饱和（frac_gt0_9 < sat_frac）。

------------------------------------------------------------------------
治理开关（全部 env 化；fail-open：任何异常/非法输入 → 中性零参与）
------------------------------------------------------------------------
  LMS_EWC_ENABLE        0/1     主开关（默认 0=关 → 零参与：penalty 返回
                                0、不更新 Fisher、snapshot enabled=False）
  LMS_EWC_LAMBDA        0.1     罚项强度 λ（初值；过大→学习僵化、过小→无效）
  LMS_EWC_FISHER_WINDOW 500     健康窗口长度（前 window 个健康样本运行平均，
                                之后转 EMA 追踪）
  LMS_EWC_FISHER_EMA    0.02    EMA 系数（窗口满后旧样本的指数衰减率）

状态为纯进程内存（同 allostatic/precision_adapt 先例：重启即失、快照不
落盘、回滚干净）。本模块只依赖标准库与 torch，不 import 本仓库运行时。
"""

import os
import logging
from typing import Optional, Sequence, Union

import torch

logger = logging.getLogger(__name__)

# 方向归一化的数值下限：‖θ‖ 低于此值 → 方向未定义 → fail-open 中性（不施力）
_EPS = 1e-8


def ewc_enabled(explicit: Optional[bool] = None) -> bool:
    """治理开关解析：显式参数 > 环境变量 LMS_EWC_ENABLE（默认 0=关）。

    布尔接受（不区分大小写）：1/true/yes/on 视为开，其余为关。
    """
    if explicit is not None:
        return bool(explicit)
    raw = os.environ.get("LMS_EWC_ENABLE", "0")
    return raw.strip().lower() in ("1", "true", "yes", "on")


class FisherAccumulator:
    """对角 Fisher 信息近似（Kirkpatrick 2017 标准做法）：健康窗口上的 EMA 累积。

    F = E[g²]（g = 学习目标对参数 θ 的梯度；本系统取 learn() 的 FEP 更新
    方向 ΔJ = −∂F/∂J，平方后与 ∂F/∂J 同），以「健康窗口」为界：

      - healthy=False（崩塌/过渡态）→ 跳过本轮。**关键约束**：Fisher 计算
        必须在健康窗口上进行，绝不在崩塌/过渡态计算（否则保护的是坏工作点
        本身，保护失效且可能放大坏点）；
      - healthy=True → 累积到 F：
          * 前 window 个健康样本：运行平均 F ← ((n−1)·F + g²)/n（无偏估计，
            有效窗口 = 已积累的健康样本数 n）；
          * 窗口满后（n ≥ window）：EMA，F ← (1−ema)·F + ema·g²（旧样本
            指数衰减，记忆时标 ≈ 1/ema 轮）。

    首个健康更新同时定格「保护锚点」θ* = weights（防崩塌子空间的方向内容
    在保护起点冻结；EwcPenalty 以此为中心施加二次罚项）。

    参数:
        shape: 参数张量形状（如 J 的 [num_nodes, num_nodes]）。
        device: 内部状态张量的设备。
        ema: EMA 系数（LMS_EWC_FISHER_EMA，默认 0.02）。
        window: 健康窗口长度（LMS_EWC_FISHER_WINDOW，默认 500）。
        enabled: 开关（None → 环境变量 LMS_EWC_ENABLE；关 → 全 no-op）。
    """

    def __init__(self, shape: Sequence[int], device: Union[str, torch.device],
                 ema: float = 0.02, window: int = 500,
                 enabled: Optional[bool] = None) -> None:
        self.enabled: bool = ewc_enabled(enabled)
        self.shape: tuple = tuple(shape)
        self.device: torch.device = (
            device if isinstance(device, torch.device) else torch.device(device))
        self.ema: float = float(ema)
        self.window: int = int(window)
        self._fisher: Optional[torch.Tensor] = None     # 对角 Fisher（同 shape）
        self._theta_star: Optional[torch.Tensor] = None  # 保护锚点 θ*
        self.count: int = 0      # 已累积的健康更新次数
        self.skipped: int = 0    # 被跳过（健康门控 / 非法输入）的次数

    # ------------------------------------------------------------------ #
    #  主入口：健康窗口累积
    # ------------------------------------------------------------------ #

    def update(self, weights: torch.Tensor, grads: torch.Tensor,
               healthy: bool) -> None:
        """健康窗口上的累积；healthy=False → 跳过（崩塌/过渡态保护）。

        内部全部 detach，不建计算图；非法输入（NaN/Inf）→ fail-open 跳过
        本轮（不污染已累积的 Fisher）。weights/grads 形状须与构造时 shape
        一致。首个健康更新定格保护锚点 θ* = weights。
        """
        if not self.enabled:
            return
        if not healthy:
            self.skipped += 1
            return
        assert tuple(weights.shape) == self.shape, (
            f"EWC Fisher: weights shape {tuple(weights.shape)} "
            f"!= {self.shape}")
        assert tuple(grads.shape) == self.shape, (
            f"EWC Fisher: grads shape {tuple(grads.shape)} != {self.shape}")
        w = weights.detach().to(self.device)
        g2 = grads.detach().to(self.device).pow(2)
        if not torch.isfinite(w).all() or not torch.isfinite(g2).all():
            logger.warning("EWC Fisher: 输入含 NaN/Inf，跳过本轮累积（fail-open）")
            self.skipped += 1
            return
        if self._theta_star is None:
            self._theta_star = w.clone()
        if self._fisher is None:
            self._fisher = g2.clone()
        elif self.count < self.window:
            # 健康窗口内：运行平均（无偏估计，有效窗口 = 已积累样本数）
            n = self.count + 1
            self._fisher = self._fisher * ((n - 1) / n) + g2 / n
        else:
            # 窗口满后：EMA 追踪（旧样本指数衰减）
            self._fisher = (1.0 - self.ema) * self._fisher + self.ema * g2
        self.count += 1

    # ------------------------------------------------------------------ #
    #  观测
    # ------------------------------------------------------------------ #

    def fisher(self) -> Optional[torch.Tensor]:
        """对角 Fisher（与 weights 同形状）；未积累 → None。返回克隆（只读）。"""
        return None if self._fisher is None else self._fisher.clone()

    def reference(self) -> Optional[torch.Tensor]:
        """保护锚点 θ*（首个健康更新时的 weights 克隆）；未定格 → None。"""
        return None if self._theta_star is None else self._theta_star.clone()

    def set_reference(self, theta: torch.Tensor) -> None:
        """显式设置保护锚点 θ*（测试/预置场景；覆盖首个健康更新的自动定格）。"""
        if self.enabled:
            self._theta_star = theta.detach().to(self.device).clone()


class EwcPenalty:
    """EWC 二次罚项（缩放不变）：P = λ·ΣFᵢ(θ̂ᵢ − θ̂*ᵢ)²。

    以「方向分量」θ̂ = θ/‖θ‖_F、θ̂* = θ*/‖θ*‖_F 定义（scale-invariant）：
    A 重锚 = 标量乘 J → 只改 ‖θ‖ 不改 θ̂ → **P(cθ) = P(θ)**，罚项不增长
    （C 豁免 A 重锚缩放方向，红线见模块 docstring）。

    罚项梯度为球面切空间投影（径向分量恒为 0）：

        ∂P/∂θ = (2λ/‖θ‖)·[F⊙(θ̂−θ̂*) − ⟨F⊙(θ̂−θ̂*), θ̂⟩·θ̂]

    —— EWC 力永不沿「整体缩放」方向施加，与 A 的径向重锚正交。

    penalty(theta) -> float      罚项值（未启用/未积累/方向未定义/异常 → 0.0）
    gradient(theta) -> Tensor    ∂P/∂θ（同上 fail-open → 与 theta 同形状全 0）
    update(weights, grads, healthy)  委托 FisherAccumulator（健康窗口 EMA）

    参数:
        shape: 参数张量形状（如 J 的 [num_nodes, num_nodes]）。
        device: 内部状态张量设备（计算随输入 theta 的设备自适应）。
        lam: λ（LMS_EWC_LAMBDA，默认 0.1）。
        window / ema: 透传给 FisherAccumulator。
        enabled: 开关（None → 环境变量 LMS_EWC_ENABLE；关 → 零参与）。
        reference: 可选预置 θ*（None → 首个健康更新自动定格）。

    状态为纯进程内存（不落盘，同 allostatic 先例）。
    """

    def __init__(self, shape: Sequence[int],
                 device: Union[str, torch.device],
                 lam: Optional[float] = None,
                 window: Optional[int] = None,
                 ema: Optional[float] = None,
                 enabled: Optional[bool] = None,
                 reference: Optional[torch.Tensor] = None) -> None:
        self.enabled: bool = ewc_enabled(enabled)
        self.lam: float = (
            float(os.environ.get("LMS_EWC_LAMBDA", "0.1"))
            if lam is None else float(lam))
        self.fisher_window: int = (
            int(os.environ.get("LMS_EWC_FISHER_WINDOW", "500"))
            if window is None else int(window))
        self.ema: float = (
            float(os.environ.get("LMS_EWC_FISHER_EMA", "0.02"))
            if ema is None else float(ema))
        self.shape: tuple = tuple(shape)
        self.device: torch.device = (
            device if isinstance(device, torch.device) else torch.device(device))
        self._acc = FisherAccumulator(
            self.shape, self.device, ema=self.ema, window=self.fisher_window,
            enabled=self.enabled)
        if reference is not None and self.enabled:
            self._acc.set_reference(reference)
        self.penalty_last: float = 0.0   # 最近一次罚项值（快照用）
        self.failopen_count: int = 0     # fail-open 触发次数（可观测）

    # ------------------------------------------------------------------ #
    #  主入口
    # ------------------------------------------------------------------ #

    def update(self, weights: torch.Tensor, grads: torch.Tensor,
               healthy: bool) -> None:
        """委托 FisherAccumulator（健康窗口 EMA 累积；healthy=False 跳过）。"""
        if not self.enabled:
            return
        self._acc.update(weights, grads, healthy)

    def penalty(self, theta: torch.Tensor) -> float:
        """罚项值 P = λ·ΣFᵢ(θ̂ᵢ−θ̂*ᵢ)²（scale-invariant）。

        未启用 / Fisher 未积累 / θ* 未定格 / 方向未定义（‖θ‖≈0）/
        NaN-Inf / 异常 → 0.0（fail-open，中性零参与）。
        """
        if not self.enabled:
            return 0.0
        try:
            prep = self._prep(theta)
            if prep is None:
                return 0.0
            Ff, th_hat, th_star_hat, _n_t = prep
            dev = th_hat - th_star_hat
            value = float((self.lam * torch.sum(Ff * dev * dev)).item())
            self.penalty_last = value
            return value
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("EWC 罚项计算失败（fail-open 返回 0）: %s", e)
            self.failopen_count += 1
            return 0.0

    def gradient(self, theta: torch.Tensor) -> torch.Tensor:
        """罚项梯度 ∂P/∂θ（球面切空间投影，径向分量为 0）。

        未启用 / Fisher 未积累 / θ* 未定格 / 方向未定义 / NaN-Inf / 异常
        → 与 theta 同形状全 0（fail-open，中性零参与）。
        """
        zeros = torch.zeros_like(theta)
        if not self.enabled:
            return zeros
        try:
            prep = self._prep(theta)
            if prep is None:
                return zeros
            Ff, th_hat, th_star_hat, n_t = prep
            dev = th_hat - th_star_hat
            Fd = Ff * dev
            radial = torch.dot(Fd, th_hat)   # ⟨F⊙(θ̂−θ̂*), θ̂⟩ 标量
            g = (2.0 * self.lam / float(n_t)) * (Fd - radial * th_hat)
            # 顺带记录罚项值（与 penalty() 同源，供 snapshot.penalty_last）
            self.penalty_last = float(
                (self.lam * torch.sum(Ff * dev * dev)).item())
            return g.reshape(theta.shape)
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("EWC 梯度计算失败（fail-open 返回 0）: %s", e)
            self.failopen_count += 1
            return zeros

    # ------------------------------------------------------------------ #
    #  内部：方向归一化前件（scale-invariant 的核心）
    # ------------------------------------------------------------------ #

    def _prep(self, theta: torch.Tensor):
        """罚项/梯度的公共前件。

        返回 (Ff, θ̂, θ̂*, ‖θ‖_F)（Ff、θ̂、θ̂* 为展平向量，设备随 theta）；
        任一前提缺失 / 方向未定义 / 非有限 → None（fail-open 中性）。
        """
        F = self._acc.fisher()
        if F is None:
            return None
        s = self._acc.reference()
        if s is None:
            return None
        t = theta.detach().reshape(-1)
        s = s.to(t.device).reshape(-1)
        n_t = torch.linalg.vector_norm(t)
        n_s = torch.linalg.vector_norm(s)
        if n_t < _EPS or n_s < _EPS:
            return None
        if not (torch.isfinite(t).all() and torch.isfinite(s).all()):
            return None
        Ff = F.to(t.device).reshape(-1)
        if not torch.isfinite(Ff).all():
            return None
        th_hat = t / n_t
        th_star_hat = s / n_s
        return Ff, th_hat, th_star_hat, n_t

    # ------------------------------------------------------------------ #
    #  观测
    # ------------------------------------------------------------------ #

    def snapshot(self) -> dict:
        """观测块（/status ewc；灵魂指标：保护在动）：固定键集合。

        {enabled, lambda, fisher_window, fisher_ready, penalty_last}
        """
        return {
            "enabled": bool(self.enabled),
            "lambda": self.lam,
            "fisher_window": self.fisher_window,
            "fisher_ready": self._acc.fisher() is not None,
            "penalty_last": self.penalty_last,
        }
