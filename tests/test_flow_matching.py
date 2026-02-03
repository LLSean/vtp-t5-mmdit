from __future__ import annotations

import torch

from mmdit.diffusion.flow_matching import FlowMatchingConfig, make_xt_and_vt, sample_t, time_embedding_value


def test_flow_matching_linear_target_velocity() -> None:
    cfg = FlowMatchingConfig(path="linear", t_min=0.0, t_max=1.0, time_scale=999.0)
    b = 2
    x0 = torch.randn(b, 1, 4, 4, 8)
    x1 = torch.randn_like(x0)
    t = torch.tensor([0.25, 0.75], dtype=torch.float32)
    x_t, v_t = make_xt_and_vt(x0, x1, t, cfg=cfg)
    assert x_t.shape == x0.shape
    assert v_t.shape == x0.shape
    # * Linear path velocity is constant: v = x1 - x0
    assert torch.allclose(v_t, x1 - x0)


def test_flow_matching_time_embedding_value_shape() -> None:
    cfg = FlowMatchingConfig(path="linear", t_min=0.0, t_max=1.0, time_scale=999.0)
    t = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)
    te = time_embedding_value(t, cfg=cfg)
    assert te.shape == (3,)


def test_flow_matching_sample_t_range() -> None:
    cfg = FlowMatchingConfig(path="linear", t_min=0.2, t_max=0.8, time_scale=999.0)
    t = sample_t(100, cfg=cfg, device=torch.device("cpu"), dtype=torch.float32)
    assert float(t.min()) >= 0.2
    assert float(t.max()) <= 0.8


