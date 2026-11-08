"""Tests for amd_tuned_torch.fusion_profiler.

The observation pass is pure Python bookkeeping over forward hooks and two
watched functions, so nearly all of this runs on CPU with no Triton and no
GPU. Only apply() needs a device, and even that is tested for WHERE it
points torch.compile rather than for speed -- what compiling buys is
measured by hand (see the module docstring), not asserted here, because
this machine's clocks make a timing assertion a coin flip.

Run with:

    python -m pytest source/cmp_ext_turing/tests
"""
from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import amd_tuned_torch.fusion_profiler as fp


class VAEStyle(nn.Module):
    """sgm's VAE ResnetBlock shape: inline, no Sequential, and the
    activation is a hand-written sigmoid-multiply rather than F.silu."""

    def __init__(self, c=64, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, c)
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        h = self.norm1(x)
        h = h * torch.sigmoid(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = h * torch.sigmoid(h)
        h = self.conv2(h)
        return x + h


class UNetStyle(nn.Module):
    """SDXL UNet ResBlock shape: nn.Sequential with a real nn.SiLU."""

    def __init__(self, c=64, groups=8):
        super().__init__()
        self.in_layers = nn.Sequential(
            nn.GroupNorm(groups, c), nn.SiLU(), nn.Conv2d(c, c, 3, padding=1))

    def forward(self, x):
        return self.in_layers(x)


class FunctionalSiLU(nn.Module):
    """GroupNorm followed by F.silu called directly, not via a module."""

    def __init__(self, c=64, groups=8):
        super().__init__()
        self.norm = nn.GroupNorm(groups, c)
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return self.conv(F.silu(self.norm(x)))


class Decoy(nn.Module):
    """GroupNorm NOT followed by an activation -- must never be reported."""

    def __init__(self, c=64, groups=8):
        super().__init__()
        self.norm = nn.GroupNorm(groups, c)
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return self.conv(torch.tanh(self.norm(x)))


def _model():
    m = nn.Module()
    m.vae = VAEStyle()
    m.unet = UNetStyle()
    m.func = FunctionalSiLU()
    m.decoy = Decoy()
    return m.eval()


@pytest.fixture(autouse=True)
def clean():
    fp.reset()
    yield
    fp.reset()


def _run(m, x):
    with torch.no_grad():
        return [m.vae(x), m.unet(x), m.func(x), m.decoy(x)]


class TestObservation:
    def test_finds_inline_sigmoid_multiply_sites(self):
        """The VAE case: no Sequential, and the activation is
        x*torch.sigmoid(x). This is the blind spot the profiler exists
        for -- conv_fusion_ops.replace_conv_blocks finds none of these."""
        m, x = _model(), torch.randn(2, 64, 8, 8)
        with fp.observing(m):
            _run(m, x)
        found = fp.sites(m)
        assert "vae.norm1" in found and "vae.norm2" in found

    def test_finds_sequential_and_functional_sites_too(self):
        m, x = _model(), torch.randn(2, 64, 8, 8)
        with fp.observing(m):
            _run(m, x)
        found = fp.sites(m)
        assert "unet.in_layers.0" in found      # nn.SiLU module
        assert "func.norm" in found             # F.silu call

    def test_excludes_group_norms_with_no_activation(self):
        m, x = _model(), torch.randn(2, 64, 8, 8)
        with fp.observing(m):
            _run(m, x)
        assert "decoy.norm" not in fp.sites(m)

    def test_observation_is_bit_identical(self):
        """Profiling must not change a single output value -- it records
        tensor identities and nothing else."""
        torch.manual_seed(0)
        m, x = _model(), torch.randn(2, 64, 8, 8)
        before = [t.clone() for t in _run(m, x)]
        with fp.observing(m):
            _run(m, x)
        after = _run(m, x)
        assert all(torch.equal(a, b) for a, b in zip(before, after))

    def test_restores_the_functions_it_watched(self):
        m, x = _model(), torch.randn(2, 64, 8, 8)
        sigmoid, silu = torch.sigmoid, F.silu
        with fp.observing(m):
            _run(m, x)
        assert torch.sigmoid is sigmoid
        assert F.silu is silu

    def test_restores_even_when_the_model_raises(self):
        class Boom(nn.Module):
            def forward(self, x):
                raise ValueError("boom")

        sigmoid, silu = torch.sigmoid, F.silu
        with pytest.raises(ValueError):
            with fp.observing(Boom()):
                Boom()(torch.randn(1))
        assert torch.sigmoid is sigmoid and F.silu is silu

    def test_removes_its_hooks(self):
        m, x = _model(), torch.randn(2, 64, 8, 8)
        with fp.observing(m):
            _run(m, x)
        for mod in m.modules():
            if isinstance(mod, nn.GroupNorm):
                assert not mod._forward_hooks

    def test_nothing_recorded_outside_the_window(self):
        """Ids are reused once a tensor is freed, so a stale _pending entry
        could mis-attribute a later sigmoid. The window must be closed."""
        m, x = _model(), torch.randn(2, 64, 8, 8)
        with fp.observing(m):
            pass                       # no forward at all
        _run(m, x)                     # runs with observation off
        assert fp.sites(m) == []
        assert fp._pending == {}

    def test_reset_clears_findings(self):
        m, x = _model(), torch.randn(2, 64, 8, 8)
        with fp.observing(m):
            _run(m, x)
        assert fp.sites(m)
        fp.reset()
        assert fp.sites(m) == []


class TestOwners:
    def test_owner_is_the_enclosing_module(self):
        m, x = _model(), torch.randn(2, 64, 8, 8)
        with fp.observing(m):
            _run(m, x)
        found = fp.owners(m)
        assert "vae" in found                 # parent of vae.norm1/norm2
        assert "unet.in_layers" in found      # parent of unet.in_layers.0
        assert "func" in found
        assert "decoy" not in found

    def test_owners_are_deduplicated(self):
        """vae.norm1 and vae.norm2 share one owner; it must appear once."""
        m, x = _model(), torch.randn(2, 64, 8, 8)
        with fp.observing(m):
            _run(m, x)
        found = fp.owners(m)
        assert found.count("vae") == 1


class TestPersistence:
    def test_round_trip(self, tmp_path):
        m, x = _model(), torch.randn(2, 64, 8, 8)
        with fp.observing(m):
            _run(m, x)
        path = tmp_path / "sites.json"
        fp.save(m, str(path))
        assert json.loads(path.read_text())["owners"] == fp.owners(m)
        assert fp.load(str(path)) == fp.owners(m)

    def test_load_missing_file_is_empty(self, tmp_path):
        assert fp.load(str(tmp_path / "nope.json")) == []

    def test_load_corrupt_file_is_empty(self, tmp_path):
        p = tmp_path / "sites.json"
        p.write_text("{not json")
        assert fp.load(str(p)) == []


class TestApply:
    def test_compiles_only_the_owners(self, monkeypatch):
        compiled = []

        def fake_compile(mod, **kw):
            compiled.append(mod)
            return mod

        monkeypatch.setattr(torch, "compile", fake_compile)
        m, x = _model(), torch.randn(2, 64, 8, 8)
        with fp.observing(m):
            _run(m, x)
        n = fp.apply(m)
        assert n == len(fp.owners(m))
        assert m.decoy not in compiled       # never the decoy

    def test_apply_with_no_findings_does_nothing(self, monkeypatch, tmp_path):
        monkeypatch.setenv(fp._CACHE_ENV, str(tmp_path / "none.json"))
        monkeypatch.setattr(torch, "compile",
                            lambda *a, **k: pytest.fail("should not compile"))
        assert fp.apply(_model()) == 0

    def test_uses_disk_findings_when_this_process_observed_nothing(
            self, monkeypatch, tmp_path):
        """Run 1 and run 2 are usually different processes -- a webui loads
        a model, generates, and exits."""
        path = tmp_path / "sites.json"
        path.write_text(json.dumps({"sites": ["vae.norm1"], "owners": ["vae"]}))
        monkeypatch.setenv(fp._CACHE_ENV, str(path))
        compiled = []
        monkeypatch.setattr(torch, "compile",
                            lambda mod, **kw: compiled.append(mod) or mod)
        m = _model()
        assert fp.sites(m) == []          # nothing observed in THIS process
        assert fp.apply(m) == 1
        assert compiled == [m.vae]

    def test_unknown_paths_are_skipped_not_fatal(self, monkeypatch):
        """A profile from a different model must not crash apply()."""
        monkeypatch.setattr(torch, "compile", lambda mod, **kw: mod)
        assert fp.apply(_model(), owner_names=["does.not.exist"]) == 0

    def test_describe_never_raises(self):
        assert isinstance(fp.describe(), str)
        assert isinstance(fp.describe(_model()), str)


def test_profile_is_the_same_whether_or_not_enable_ran():
    """The profile must describe the MODEL, not the dispatch state.

    _dispatch's nn.Sequential patch routes a matching block straight to
    conv_fusion_ops, bypassing nn.GroupNorm.forward -- so without the
    stand-down in observing() the profiler reports different sites
    depending on whether enable() happened to have run first, which makes
    a saved profile non-reproducible."""
    import amd_tuned_torch

    m, x = _model(), torch.randn(2, 64, 8, 8)

    amd_tuned_torch.disable()
    fp.reset()
    with fp.observing(m):
        _run(m, x)
    without_enable = fp.sites(m)

    amd_tuned_torch.enable()
    try:
        fp.reset()
        with fp.observing(m):
            _run(m, x)
        with_enable = fp.sites(m)
    finally:
        amd_tuned_torch.disable()

    assert without_enable == with_enable
    assert "unet.in_layers.0" in with_enable


def test_sequential_patch_is_reinstalled_after_observing():
    import amd_tuned_torch
    from amd_tuned_torch import _dispatch, conv_fusion_ops

    amd_tuned_torch.enable()
    try:
        if not conv_fusion_ops.enabled():
            pytest.skip("conv fusion unavailable, patch never installed")
        assert nn.Sequential.forward is _dispatch._patched_sequential_forward
        m, x = _model(), torch.randn(2, 64, 8, 8)
        with fp.observing(m):
            assert nn.Sequential.forward is not _dispatch._patched_sequential_forward
            _run(m, x)
        assert nn.Sequential.forward is _dispatch._patched_sequential_forward
    finally:
        amd_tuned_torch.disable()
