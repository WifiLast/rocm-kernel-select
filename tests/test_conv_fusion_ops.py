"""Tests for amd_tuned_torch.conv_fusion_ops.

Split by what each part needs:

  * The MODULE SURGERY (replace_conv_blocks / unfuse_conv_blocks / which
    Sequentials match) is pure nn.Module bookkeeping and runs anywhere,
    on CPU, with no Triton and no GPU. That is most of this file.
  * The KERNEL itself needs both, so those tests skip when either is
    missing. They check NUMERICS against stock, not speed.

NOTHING HERE BENCHMARKS. This card's clocks swing by up to 4x between
runs -- the same kernel on the same shape measured 0.18ms and 0.73ms
minutes apart with no other GPU process -- so a timing assertion would be
a coin flip in CI. The performance claims in conv_fusion_ops' docstring
come from an interleaved A/B harness run by hand; see that docstring's
"MEASURING THIS YOURSELF" note for why one-then-the-other timing produced
"3x slower" for a change that is slightly faster.

Run with:

    python -m pytest source/cmp_ext_turing/tests
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import amd_tuned_torch.conv_fusion_ops as cf

# References use torch.group_norm (the ATen builtin) rather than
# F.group_norm: enable() replaces F.group_norm, and this repo's conftest
# backs the native extension with MagicMocks, so F.group_norm can return
# a MagicMock here. The builtin is never patched.


needs_kernel = pytest.mark.skipif(
    not cf.available(), reason="needs Triton and a GPU")


def _block(c=64, k=64, groups=8):
    return nn.Sequential(nn.GroupNorm(groups, c), nn.SiLU(), nn.Conv2d(c, k, 3, padding=1))


# ---------------------------------------------------------------------------
# Module surgery -- no GPU, no Triton
# ---------------------------------------------------------------------------

class TestPatternMatch:
    def test_matches_the_sdxl_resblock_shape(self):
        assert cf._is_target(_block())

    def test_rejects_wrong_order(self):
        # SiLU before the norm is a different computation; fusing it would
        # silently change results.
        assert not cf._is_target(
            nn.Sequential(nn.SiLU(), nn.GroupNorm(8, 64), nn.Conv2d(64, 64, 3)))

    def test_rejects_batchnorm_and_layernorm(self):
        assert not cf._is_target(
            nn.Sequential(nn.BatchNorm2d(64), nn.SiLU(), nn.Conv2d(64, 64, 3)))

    def test_rejects_other_activations(self):
        for act in (nn.ReLU(), nn.GELU(), nn.Identity()):
            assert not cf._is_target(
                nn.Sequential(nn.GroupNorm(8, 64), act, nn.Conv2d(64, 64, 3)))

    def test_rejects_non_affine_group_norm(self):
        # The kernel reads weight/bias; a non-affine GroupNorm has none.
        assert not cf._is_target(
            nn.Sequential(nn.GroupNorm(8, 64, affine=False), nn.SiLU(),
                          nn.Conv2d(64, 64, 3)))

    def test_rejects_wrong_length_and_conv1d(self):
        assert not cf._is_target(nn.Sequential(nn.GroupNorm(8, 64), nn.SiLU()))
        assert not cf._is_target(
            nn.Sequential(nn.GroupNorm(8, 64), nn.SiLU(), nn.Conv1d(64, 64, 3)))
        assert not cf._is_target(nn.Conv2d(64, 64, 3))


class TestReplacement:
    def _model(self):
        m = nn.Module()
        m.a = _block()
        m.nested = nn.Module()
        m.nested.b = _block()
        m.nested.untouched = nn.Sequential(nn.ReLU(), nn.Conv2d(64, 64, 1))
        m.plain = nn.Conv2d(3, 8, 1)
        return m

    def test_replaces_recursively_and_counts(self, monkeypatch):
        monkeypatch.setattr(cf, "enabled", lambda: True)
        m = self._model()
        assert cf.replace_conv_blocks(m) == 2
        assert isinstance(m.a, cf.FusedGroupNormSiLUConv2d)
        assert isinstance(m.nested.b, cf.FusedGroupNormSiLUConv2d)
        # Everything else is untouched.
        assert isinstance(m.nested.untouched, nn.Sequential)
        assert isinstance(m.plain, nn.Conv2d)

    def test_reuses_the_same_parameter_objects(self, monkeypatch):
        """Not copies -- an optimizer already holding these Parameters, or a
        model already training, must keep working across the swap."""
        monkeypatch.setattr(cf, "enabled", lambda: True)
        m = self._model()
        gn_w, gn_b = m.a[0].weight, m.a[0].bias
        cv_w = m.a[2].weight
        cf.replace_conv_blocks(m)
        assert m.a.norm.weight is gn_w
        assert m.a.norm.bias is gn_b
        assert m.a.conv.weight is cv_w

    def test_state_dict_keys_are_unchanged(self, monkeypatch):
        """The fused module names its children "0" and "2" so a checkpoint
        written before the swap still loads after it."""
        monkeypatch.setattr(cf, "enabled", lambda: True)
        m = self._model()
        before = set(m.state_dict())
        cf.replace_conv_blocks(m)
        assert set(m.state_dict()) == before

    def test_is_idempotent(self, monkeypatch):
        monkeypatch.setattr(cf, "enabled", lambda: True)
        m = self._model()
        assert cf.replace_conv_blocks(m) == 2
        assert cf.replace_conv_blocks(m) == 0

    def test_unfuse_restores_sequential_and_parameters(self, monkeypatch):
        monkeypatch.setattr(cf, "enabled", lambda: True)
        m = self._model()
        gn_w, cv_w = m.a[0].weight, m.a[2].weight
        cf.replace_conv_blocks(m)
        assert cf.unfuse_conv_blocks(m) == 2
        assert isinstance(m.a, nn.Sequential) and len(m.a) == 3
        assert isinstance(m.a[1], nn.SiLU)
        assert m.a[0].weight is gn_w and m.a[2].weight is cv_w
        assert set(m.state_dict()) == set(self._model().state_dict())

    def test_disabled_changes_nothing(self, monkeypatch):
        monkeypatch.setattr(cf, "enabled", lambda: False)
        m = self._model()
        assert cf.replace_conv_blocks(m) == 0
        assert isinstance(m.a, nn.Sequential)


class TestGating:
    def test_env_var_disables(self, monkeypatch):
        monkeypatch.setattr(cf, "available", lambda: True)
        monkeypatch.setenv(cf._ENV, "0")
        assert cf.enabled() is False
        monkeypatch.setenv(cf._ENV, "1")
        assert cf.enabled() is True

    def test_unavailable_is_never_enabled(self, monkeypatch):
        monkeypatch.setattr(cf, "available", lambda: False)
        monkeypatch.delenv(cf._ENV, raising=False)
        assert cf.enabled() is False

    def test_describe_never_raises(self):
        assert isinstance(cf.describe(), str)

    def test_group_norm_silu_falls_back_on_cpu(self):
        """A CPU tensor must take the stock path rather than reaching a
        Triton kernel that cannot run on it."""
        x = torch.randn(2, 8, 4, 4)
        w, b = torch.randn(8), torch.randn(8)
        got = cf.group_norm_silu(x, 2, w, b)
        ref = F.silu(torch.group_norm(x, 2, w, b))
        assert torch.allclose(got, ref, atol=1e-6)


# ---------------------------------------------------------------------------
# The kernel -- needs Triton + GPU. Numerics only.
# ---------------------------------------------------------------------------

@needs_kernel
@pytest.mark.parametrize("shape,groups", [
    ((2, 1280, 64, 32), 32),    # the two most-called SDXL UNet shapes
    ((2, 640, 128, 64), 32),
    ((2, 320, 256, 128), 32),   # biggest group: ~82K elements, the fp16 trap
    ((1, 32, 8, 8), 32),        # one channel per group
    ((2, 128, 16, 16), 8),
])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_fused_matches_stock(shape, groups, dtype):
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda", dtype=dtype)
    w = torch.randn(shape[1], device="cuda", dtype=dtype)
    b = torch.randn(shape[1], device="cuda", dtype=dtype)
    got = cf._fused_group_norm_silu(x, groups, w, b, 1e-5)
    ref = F.silu(torch.group_norm(x, groups, w, b))
    assert torch.isfinite(got).all(), "non-finite output -- fp32 accumulation broken?"
    tol = 3e-2 if dtype is torch.float16 else 1e-4
    assert (got.float() - ref.float()).abs().max().item() < tol


@needs_kernel
def test_large_group_does_not_overflow_fp16():
    """The regression this kernel's fp32 accumulators exist for.

    A GroupNorm group at SDXL sizes is 40 channels x 256x128 = 1.3M
    elements. Accumulating x*x in fp16 passes 65,504 almost immediately and
    the whole output becomes NaN. Values are scaled up here so a fp16
    accumulator would certainly overflow, not just might."""
    x = torch.full((1, 320, 256, 128), 8.0, device="cuda", dtype=torch.float16)
    x += torch.randn_like(x)
    w = torch.ones(320, device="cuda", dtype=torch.float16)
    b = torch.zeros(320, device="cuda", dtype=torch.float16)
    got = cf._fused_group_norm_silu(x, 32, w, b, 1e-5)
    assert torch.isfinite(got).all()
    ref = F.silu(torch.group_norm(x, 32, w, b))
    assert (got.float() - ref.float()).abs().max().item() < 3e-2


@needs_kernel
def test_contest_result_is_correct_whichever_side_wins():
    """group_norm_silu must return the right answer regardless of which
    candidate the contest picks -- including the shapes where the fused
    kernel loses and stock is selected."""
    for shape in ((2, 320, 256, 128), (2, 1280, 32, 16)):
        x = torch.randn(*shape, device="cuda", dtype=torch.float16)
        w = torch.randn(shape[1], device="cuda", dtype=torch.float16)
        b = torch.randn(shape[1], device="cuda", dtype=torch.float16)
        got = cf.group_norm_silu(x, 32, w, b)
        ref = F.silu(torch.group_norm(x, 32, w, b))
        assert (got.float() - ref.float()).abs().max().item() < 3e-2


@needs_kernel
def test_fused_module_matches_eager_block():
    """The reference is built from unpatched ops rather than by running the
    nn.Sequential: nn.GroupNorm.forward calls F.group_norm, which
    enable() replaces and which conftest backs with a MagicMock, so running
    the Sequential here fails inside the REFERENCE and tells us nothing
    about the fused module."""
    torch.manual_seed(0)
    seq = _block(c=64, k=64, groups=8).cuda().half().eval()
    gn, conv = seq[0], seq[2]
    x = torch.randn(2, 64, 32, 32, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        ref = conv(F.silu(torch.group_norm(x, gn.num_groups, gn.weight, gn.bias, gn.eps)))
        got = cf.FusedGroupNormSiLUConv2d(gn, conv)(x)
    assert (got.float() - ref.float()).abs().max().item() < 3e-2


@needs_kernel
def test_gradient_path_falls_back_to_stock():
    """Forward-only: with a live gradient the stock path must run, so no
    autograd graph is ever built through the Triton kernel."""
    x = torch.randn(2, 64, 16, 16, device="cuda", dtype=torch.float32,
                    requires_grad=True)
    w = torch.randn(64, device="cuda", dtype=torch.float32, requires_grad=True)
    b = torch.randn(64, device="cuda", dtype=torch.float32, requires_grad=True)
    out = cf.group_norm_silu(x, 8, w, b)
    assert out.requires_grad and out.grad_fn is not None
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# The automatic patch: _dispatch installs it on nn.Sequential.forward, so
# SD's ResBlock blocks fuse without anyone calling replace_conv_blocks.
# ---------------------------------------------------------------------------

class TestSequentialPatch:
    """These drive _patched_sequential_forward directly rather than going
    through enable(): enable() installs a dozen other patches, and a test
    that needs all of them to work is testing the wrong thing."""

    def _install(self, monkeypatch):
        """Install the patch and return (_dispatch, true_original).

        Never captures the patch as its own original -- doing that makes
        _patched_sequential_forward call itself forever, which is how this
        class first failed and the bug it found in enable().

        The patch may ALREADY be installed when this runs: conftest sets
        AMD_TUNED_TORCH_AUTOPATCH with os.environ.setdefault, so a shell
        that already exports AMD_TUNED_TORCH_AUTOPATCH=1 (this machine
        does) keeps enable() running at import and defeats that isolation.
        In that case the true original lives in _ORIGINALS."""
        from amd_tuned_torch import _dispatch
        current = nn.Sequential.forward
        if current is _dispatch._patched_sequential_forward:
            current = _dispatch._ORIGINALS[(nn.Sequential, "forward")]
        assert current is not _dispatch._patched_sequential_forward
        monkeypatch.setattr(_dispatch, "_SEQ_FORWARD_ORIG", current)
        monkeypatch.setattr(nn.Sequential, "forward",
                            _dispatch._patched_sequential_forward)
        return _dispatch, current

    def test_declines_every_non_matching_sequential(self, monkeypatch):
        """The decline path runs for EVERY nn.Sequential in the process, so
        it must return byte-identical results for anything it does not
        fuse."""
        _dispatch, stock_forward = self._install(monkeypatch)
        cases = [
            (nn.Sequential(nn.Linear(8, 8), nn.ReLU()), torch.randn(4, 8)),
            (nn.Sequential(nn.ReLU(), nn.Tanh(), nn.Sigmoid()), torch.randn(4, 8)),
            (nn.Sequential(nn.BatchNorm2d(4), nn.SiLU(), nn.Conv2d(4, 4, 3, padding=1)),
             torch.randn(2, 4, 8, 8)),
            (nn.Sequential(nn.GroupNorm(2, 4), nn.ReLU(), nn.Conv2d(4, 4, 3, padding=1)),
             torch.randn(2, 4, 8, 8)),
            (nn.Sequential(nn.GroupNorm(2, 4), nn.SiLU(), nn.Conv1d(4, 4, 3, padding=1)),
             torch.randn(2, 4, 8)),
            (nn.Sequential(nn.Conv2d(4, 4, 1)), torch.randn(2, 4, 8, 8)),
        ]
        for seq, x in cases:
            seq.eval()
            with torch.no_grad():
                patched = seq(x)
                stock = stock_forward(seq, x)
            assert torch.equal(patched, stock), f"declined path changed {seq}"

    def test_non_affine_group_norm_is_declined(self, monkeypatch):
        """The kernel reads weight/bias; a non-affine GroupNorm has neither,
        and must not reach it."""
        _dispatch, stock_forward = self._install(monkeypatch)
        seq = nn.Sequential(nn.GroupNorm(2, 4, affine=False), nn.SiLU(),
                            nn.Conv2d(4, 4, 3, padding=1)).eval()
        x = torch.randn(2, 4, 8, 8)
        with torch.no_grad():
            assert torch.equal(seq(x), stock_forward(seq, x))

    def test_cpu_matching_block_still_correct(self, monkeypatch):
        """A block that MATCHES the pattern but runs on CPU: group_norm_silu
        falls back to stock, so the patch must still return the right
        answer rather than reaching a Triton kernel."""
        _dispatch, stock_forward = self._install(monkeypatch)
        torch.manual_seed(0)
        seq = nn.Sequential(nn.GroupNorm(2, 4), nn.SiLU(),
                            nn.Conv2d(4, 4, 3, padding=1)).eval()
        x = torch.randn(2, 4, 8, 8)
        with torch.no_grad():
            got = seq(x)
            ref = stock_forward(seq, x)
        assert torch.allclose(got, ref, atol=1e-5)

    def test_decision_is_not_cached_on_the_instance(self, monkeypatch):
        """Mutating a Sequential's children after it has been called must
        change the decision -- a cached flag would go stale and keep fusing
        (or keep declining) the wrong module."""
        _dispatch, stock_forward = self._install(monkeypatch)
        seq = nn.Sequential(nn.GroupNorm(2, 4), nn.SiLU(),
                            nn.Conv2d(4, 4, 3, padding=1)).eval()
        x = torch.randn(2, 4, 8, 8)
        with torch.no_grad():
            seq(x)                      # first call: matches
            seq[1] = nn.ReLU()          # no longer the fused pattern
            got = seq(x)
            ref = stock_forward(seq, x)
        assert torch.equal(got, ref)
        # And no bookkeeping attribute was left on the module.
        assert not any(k.startswith("_amd_tuned") for k in seq.__dict__)

    @needs_kernel
    def test_matching_block_matches_unpatched_reference(self, monkeypatch):
        _dispatch, stock_forward = self._install(monkeypatch)
        torch.manual_seed(0)
        seq = _block(c=64, k=64, groups=8).cuda().half().eval()
        gn, conv = seq[0], seq[2]
        x = torch.randn(2, 64, 32, 32, device="cuda", dtype=torch.float16)
        with torch.no_grad():
            ref = conv(F.silu(torch.group_norm(x, gn.num_groups, gn.weight,
                                               gn.bias, gn.eps)))
            got = seq(x)
        assert (got.float() - ref.float()).abs().max().item() < 3e-2

    def test_enable_disable_round_trip_restores_and_never_recurses(self):
        """enable() installs through _install, so disable() unwinds it with
        everything else. Also the regression guard for the recursion bug:
        enable() -> disable() -> enable() must not capture the patched
        function as its own original."""
        import amd_tuned_torch
        from amd_tuned_torch import _dispatch, conv_fusion_ops

        amd_tuned_torch.disable()
        stock = nn.Sequential.forward
        assert stock is not _dispatch._patched_sequential_forward

        for _ in range(3):
            amd_tuned_torch.enable()
            if conv_fusion_ops.enabled():
                assert nn.Sequential.forward is _dispatch._patched_sequential_forward
                assert _dispatch._SEQ_FORWARD_ORIG is not \
                    _dispatch._patched_sequential_forward, "would recurse forever"
                # And it actually runs, rather than blowing the stack.
                seq = nn.Sequential(nn.Linear(4, 4), nn.ReLU()).eval()
                with torch.no_grad():
                    seq(torch.randn(2, 4))
            amd_tuned_torch.disable()
            assert nn.Sequential.forward is stock


@needs_kernel
def test_triton_declines_channels_last():
    """The Triton kernel indexes NCHW. It used to accept channels_last by
    calling .contiguous(), which transposes the whole tensor and measured
    SLOWER than stock on every shape tried. It must decline so the contest
    hands the layout to CK instead."""
    x = torch.randn(2, 64, 16, 16, device="cuda", dtype=torch.float16) \
        .to(memory_format=torch.channels_last)
    w = torch.randn(64, device="cuda", dtype=torch.float16)
    b = torch.randn(64, device="cuda", dtype=torch.float16)
    assert cf._fused_group_norm_silu(x, 8, w, b, 1e-5) is None
    # Contiguous NCHW is still served.
    assert cf._fused_group_norm_silu(x.contiguous(), 8, w, b, 1e-5) is not None


@needs_kernel
@pytest.mark.parametrize("channels_last", [False, True])
def test_group_norm_silu_correct_in_both_layouts(channels_last):
    torch.manual_seed(0)
    x = torch.randn(2, 64, 16, 16, device="cuda", dtype=torch.float16)
    if channels_last:
        x = x.to(memory_format=torch.channels_last)
    w = torch.randn(64, device="cuda", dtype=torch.float16)
    b = torch.randn(64, device="cuda", dtype=torch.float16)
    got = cf.group_norm_silu(x, 8, w, b)
    ref = F.silu(torch.group_norm(x, 8, w, b))
    assert torch.isfinite(got).all()
    assert (got.float() - ref.float()).abs().max().item() < 3e-2


def test_contest_key_distinguishes_memory_format(monkeypatch):
    """Which candidate wins flips entirely between NCHW and channels_last,
    so the cache key must separate them -- otherwise one layout's winner is
    applied to the other, which is exactly backwards."""
    seen = []

    def fake_pick(kind, key, candidates, tolerance=None):
        seen.append(key)
        return candidates[-1][1]()

    monkeypatch.setattr(cf.kernel_select, "pick_key", fake_pick)
    monkeypatch.setattr(cf, "enabled", lambda: True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        pytest.skip("needs a GPU: the CPU path short-circuits before the contest")
    x = torch.randn(2, 64, 8, 8, device=dev, dtype=torch.float16)
    w = torch.randn(64, device=dev, dtype=torch.float16)
    b = torch.randn(64, device=dev, dtype=torch.float16)
    cf.group_norm_silu(x, 8, w, b)
    cf.group_norm_silu(x.to(memory_format=torch.channels_last), 8, w, b)
    assert len(seen) == 2 and seen[0] != seen[1]
    assert seen[0][-1] == "nchw" and seen[1][-1] == "nhwc"
