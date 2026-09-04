"""Numerical correctness of amd_tuned_torch.fused_ce_ops.fused_linear_cross_entropy
-- loss AND gradients (input/weight/bias) -- against plain, unfused
F.cross_entropy + autograd, on real hardware. This is the most complex
kernel in this package (a real backward pass, label smoothing, softcap,
class weights, ignore_index all interacting), so this file is more
thorough than the other tests_hardware/ files: every optional feature is
exercised both alone and in combination with the others.

Run with:

    pytest tests_hardware/test_fused_ce_ops.py
"""
import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA/ROCm device available"
)
pytest.importorskip(
    "amd_tuned_torch",
    reason="amd_tuned_torch native extension not built -- run "
           "`pip install -e . --no-build-isolation` first",
)
from amd_tuned_torch import fused_ce_ops  # noqa: E402

if not fused_ce_ops.available():
    pytest.skip("triton is not importable in this environment", allow_module_level=True)


def _reference(input, weight, target, bias=None, ce_weight=None, ignore_index=-100,
               label_smoothing=0.0, reduction="mean", softcap=None):
    logits = input.float() @ weight.float().t()
    if bias is not None:
        logits = logits + bias.float()
    if softcap is not None:
        logits = softcap * torch.tanh(logits / softcap)
    return F.cross_entropy(
        logits, target, weight=ce_weight.float() if ce_weight is not None else None,
        ignore_index=ignore_index, label_smoothing=label_smoothing, reduction=reduction,
    )


def _run_both(input, weight, target, **kwargs):
    input_ref = input.clone().requires_grad_()
    weight_ref = weight.clone().requires_grad_()
    bias = kwargs.pop("bias", None)
    bias_ref = bias.clone().requires_grad_() if bias is not None else None

    loss_ref = _reference(input_ref, weight_ref, target, bias=bias_ref, **kwargs)
    loss_ref.backward()

    input_fused = input.clone().requires_grad_()
    weight_fused = weight.clone().requires_grad_()
    bias_fused = bias.clone().requires_grad_() if bias is not None else None
    loss_fused = fused_ce_ops.fused_linear_cross_entropy(
        input_fused, weight_fused, target, bias=bias_fused, **kwargs
    )
    loss_fused.backward()

    return (loss_ref, input_ref.grad, weight_ref.grad, bias_ref.grad if bias is not None else None), \
           (loss_fused, input_fused.grad, weight_fused.grad, bias_fused.grad if bias is not None else None)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("BT, H, V", [(8, 16, 32), (37, 64, 257)])
def test_basic_matches_reference(dtype, BT, H, V):
    torch.manual_seed(0)
    input = torch.randn(BT, H, dtype=dtype, device="cuda") * 0.1
    weight = torch.randn(V, H, dtype=dtype, device="cuda") * 0.1
    target = torch.randint(0, V, (BT,), device="cuda")

    (loss_ref, gi_ref, gw_ref, _), (loss_fused, gi_fused, gw_fused, _) = _run_both(input, weight, target)

    torch.testing.assert_close(loss_fused.float(), loss_ref.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(gi_fused.float(), gi_ref.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(gw_fused.float(), gw_ref.float(), atol=2e-2, rtol=2e-2)


def test_with_bias():
    torch.manual_seed(1)
    BT, H, V = 16, 32, 100
    input = torch.randn(BT, H, device="cuda") * 0.1
    weight = torch.randn(V, H, device="cuda") * 0.1
    bias = torch.randn(V, device="cuda") * 0.1
    target = torch.randint(0, V, (BT,), device="cuda")

    (loss_ref, gi_ref, gw_ref, gb_ref), (loss_fused, gi_fused, gw_fused, gb_fused) = _run_both(
        input, weight, target, bias=bias
    )
    torch.testing.assert_close(loss_fused, loss_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(gi_fused, gi_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(gw_fused, gw_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(gb_fused, gb_ref, atol=2e-2, rtol=2e-2)


def test_ignore_index():
    torch.manual_seed(2)
    BT, H, V = 20, 32, 64
    input = torch.randn(BT, H, device="cuda") * 0.1
    weight = torch.randn(V, H, device="cuda") * 0.1
    target = torch.randint(0, V, (BT,), device="cuda")
    target[::4] = -100  # ignore every 4th token

    (loss_ref, gi_ref, gw_ref, _), (loss_fused, gi_fused, gw_fused, _) = _run_both(
        input, weight, target, ignore_index=-100
    )
    torch.testing.assert_close(loss_fused, loss_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(gi_fused, gi_ref, atol=2e-2, rtol=2e-2)
    # Ignored rows must get exactly zero gradient, not just "close to stock".
    assert torch.all(gi_fused[::4] == 0)
    torch.testing.assert_close(gw_fused, gw_ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("label_smoothing", [0.1, 0.3])
def test_label_smoothing(label_smoothing):
    torch.manual_seed(3)
    BT, H, V = 16, 32, 64
    input = torch.randn(BT, H, device="cuda") * 0.1
    weight = torch.randn(V, H, device="cuda") * 0.1
    target = torch.randint(0, V, (BT,), device="cuda")

    (loss_ref, gi_ref, gw_ref, _), (loss_fused, gi_fused, gw_fused, _) = _run_both(
        input, weight, target, label_smoothing=label_smoothing
    )
    torch.testing.assert_close(loss_fused, loss_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(gi_fused, gi_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(gw_fused, gw_ref, atol=2e-2, rtol=2e-2)


def test_ce_weight():
    torch.manual_seed(4)
    BT, H, V = 16, 32, 64
    input = torch.randn(BT, H, device="cuda") * 0.1
    weight = torch.randn(V, H, device="cuda") * 0.1
    target = torch.randint(0, V, (BT,), device="cuda")
    ce_weight = torch.rand(V, device="cuda") + 0.5

    (loss_ref, gi_ref, gw_ref, _), (loss_fused, gi_fused, gw_fused, _) = _run_both(
        input, weight, target, ce_weight=ce_weight
    )
    torch.testing.assert_close(loss_fused, loss_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(gi_fused, gi_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(gw_fused, gw_ref, atol=3e-2, rtol=3e-2)


def test_softcap():
    torch.manual_seed(5)
    BT, H, V = 16, 32, 64
    input = torch.randn(BT, H, device="cuda") * 0.1
    weight = torch.randn(V, H, device="cuda") * 0.1
    target = torch.randint(0, V, (BT,), device="cuda")

    (loss_ref, gi_ref, gw_ref, _), (loss_fused, gi_fused, gw_fused, _) = _run_both(
        input, weight, target, softcap=20.0
    )
    torch.testing.assert_close(loss_fused, loss_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(gi_fused, gi_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(gw_fused, gw_ref, atol=2e-2, rtol=2e-2)


def test_reduction_sum():
    torch.manual_seed(6)
    BT, H, V = 16, 32, 64
    input = torch.randn(BT, H, device="cuda") * 0.1
    weight = torch.randn(V, H, device="cuda") * 0.1
    target = torch.randint(0, V, (BT,), device="cuda")

    (loss_ref, gi_ref, gw_ref, _), (loss_fused, gi_fused, gw_fused, _) = _run_both(
        input, weight, target, reduction="sum"
    )
    torch.testing.assert_close(loss_fused, loss_ref, atol=5e-2, rtol=2e-2)
    torch.testing.assert_close(gi_fused, gi_ref, atol=5e-2, rtol=2e-2)
    torch.testing.assert_close(gw_fused, gw_ref, atol=5e-2, rtol=2e-2)


def test_all_features_combined():
    """label_smoothing + ce_weight + softcap + ignore_index + bias all at
    once -- the interaction between these is exactly what the dx_y
    true-class correction in the kernel has to get right simultaneously,
    not just each feature in isolation."""
    torch.manual_seed(7)
    BT, H, V = 24, 40, 96
    input = torch.randn(BT, H, device="cuda") * 0.1
    weight = torch.randn(V, H, device="cuda") * 0.1
    bias = torch.randn(V, device="cuda") * 0.1
    ce_weight = torch.rand(V, device="cuda") + 0.5
    target = torch.randint(0, V, (BT,), device="cuda")
    target[::5] = -100

    (loss_ref, gi_ref, gw_ref, gb_ref), (loss_fused, gi_fused, gw_fused, gb_fused) = _run_both(
        input, weight, target, bias=bias, ce_weight=ce_weight,
        label_smoothing=0.1, softcap=15.0, ignore_index=-100,
    )
    torch.testing.assert_close(loss_fused, loss_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(gi_fused, gi_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(gw_fused, gw_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(gb_fused, gb_ref, atol=3e-2, rtol=3e-2)
