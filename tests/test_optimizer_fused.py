"""PB03 · fused Adam 配置与数值一致性测试。

- optimizer_fused=True 在 CPU 上明确报错（不静默回退）
- CUDA 上：固定同一梯度的 FP32 单步，fused 与默认实现参数/state 在 atol=1e-6/rtol=1e-5 内一致
"""

import pytest
import torch
import torch.nn as nn

from training.trainer import build_optimizer


def _cfg(fused: bool, **overrides) -> dict:
    cfg = {
        "learning_rate": 1e-3, "weight_decay": 1e-4,
        "optimizer": "adam", "optimizer_fused": fused,
    }
    cfg.update(overrides)
    return cfg


def test_fused_rejected_on_cpu():
    model = nn.Linear(4, 2)
    with pytest.raises(ValueError, match="CUDA"):
        build_optimizer(model, _cfg(True))


def test_default_off_cpu_ok():
    opt = build_optimizer(nn.Linear(4, 2), _cfg(False))
    assert isinstance(opt, torch.optim.Adam)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
def test_fused_single_step_matches_default():
    """固定同一梯度的 FP32 单步：参数与优化器 state 在容差内一致。"""
    torch.manual_seed(0)
    base = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)).cuda()
    m1 = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)).cuda()
    m2 = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)).cuda()
    m1.load_state_dict(base.state_dict())
    m2.load_state_dict(base.state_dict())

    opt1 = build_optimizer(m1, _cfg(False))
    opt2 = build_optimizer(m2, _cfg(True))

    torch.manual_seed(1)
    x = torch.randn(16, 8, device="cuda")
    target = torch.randn(16, 4, device="cuda")

    for m, opt in ((m1, opt1), (m2, opt2)):
        opt.zero_grad()
        loss = ((m(x) - target) ** 2).mean()
        loss.backward()
        opt.step()

    for p1, p2 in zip(m1.parameters(), m2.parameters(), strict=True):
        assert torch.allclose(p1, p2, atol=1e-6, rtol=1e-5), "单步参数超出容差"

    s1 = opt1.state_dict()["state"]
    s2 = opt2.state_dict()["state"]
    assert s1.keys() == s2.keys()
    for k in s1:
        for name in ("exp_avg", "exp_avg_sq"):
            a, b = s1[k][name], s2[k][name]
            assert torch.allclose(a, b, atol=1e-6, rtol=1e-5), f"state[{k}][{name}] 超出容差"
        assert int(s1[k]["step"]) == int(s2[k]["step"]) == 1
