"""Tests for amd_tuned_torch.torch_compile -- the torch.compile(mode="max-autotune"
[-no-cudagraphs], freezing, static shapes) wrapper.

No GPU, aiter, or TransformerEngine required: every test here uses plain
CPU tensors and passes backend="eager" (skips Dynamo's own default Inductor
backend selection but still exercises Dynamo tracing + the grad-safety
dispatch wrapper), same style as test_compile_ops.py's
TestTorchCompileFullGraph. Passing backend="eager" also means _resolve_mode
leaves `mode` unset (an Inductor-only concept -- see torch_compile.py's
_resolve_mode), so these tests don't depend on Inductor/Triton actually
being usable on the machine running them.

Run with:

    python -m pytest source/cmp_ext_turing/tests
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import amd_tuned_torch.torch_compile as torch_compile


class TestAvailable:
    def test_available_reflects_torch_compile_presence(self):
        assert torch_compile.available() is hasattr(torch, "compile")


class TestResolveMode:
    def test_defaults_to_inductor_mode_when_backend_unspecified(self, monkeypatch):
        monkeypatch.setattr(torch_compile, "_configure_inductor", lambda **kw: "max-autotune-no-cudagraphs")
        mode = torch_compile._resolve_mode(None, no_cudagraphs=True, freeze=True, compile_kwargs={})
        assert mode == "max-autotune-no-cudagraphs"

    def test_leaves_mode_unset_for_non_inductor_backend(self):
        mode = torch_compile._resolve_mode(None, no_cudagraphs=True, freeze=True,
                                            compile_kwargs={"backend": "eager"})
        assert mode is None

    def test_explicit_mode_always_wins(self):
        mode = torch_compile._resolve_mode("reduce-overhead", no_cudagraphs=True, freeze=True,
                                            compile_kwargs={"backend": "eager"})
        assert mode == "reduce-overhead"


class TestConfigureInductor:
    def test_returns_no_cudagraphs_mode_by_default(self):
        assert torch_compile._configure_inductor(no_cudagraphs=True, freeze=False) == "max-autotune-no-cudagraphs"

    def test_returns_cudagraphs_mode_when_disabled(self):
        assert torch_compile._configure_inductor(no_cudagraphs=False, freeze=False) == "max-autotune"

    def test_sets_freezing_only_when_requested(self):
        try:
            import torch._inductor.config as inductor_config
        except ImportError:
            pytest.skip("torch._inductor.config not available on this build")
        inductor_config.freezing = False
        torch_compile._configure_inductor(no_cudagraphs=True, freeze=False)
        assert inductor_config.freezing is False
        torch_compile._configure_inductor(no_cudagraphs=True, freeze=True)
        assert inductor_config.freezing is True
        inductor_config.freezing = False  # leave global state as found

    def test_bumps_but_never_lowers_dynamo_cache_size_limit(self):
        try:
            import torch._dynamo.config as dynamo_config
        except ImportError:
            pytest.skip("torch._dynamo.config not available on this build")
        original = dynamo_config.cache_size_limit
        try:
            dynamo_config.cache_size_limit = 1000
            torch_compile._configure_inductor(no_cudagraphs=True, freeze=False)
            assert dynamo_config.cache_size_limit == 1000  # not lowered
            dynamo_config.cache_size_limit = 1
            torch_compile._configure_inductor(no_cudagraphs=True, freeze=False)
            assert dynamo_config.cache_size_limit >= 64  # bumped
        finally:
            dynamo_config.cache_size_limit = original


class TestGradSafeDispatch:
    """The wrapper returned by compile_fn/compile_module must fall back to
    the uncompiled original whenever autograd is actually live for this
    call's tensors -- freeze=True bakes parameters in as constants, which
    would silently go stale under a real backward pass otherwise."""

    def test_uses_compiled_path_when_grad_not_needed(self):
        calls = {"orig": 0, "compiled": 0}

        def fn(x):
            calls["orig"] += 1
            return x * 2

        wrapped = torch_compile._wrap(fn, lambda x: (calls.__setitem__("compiled", calls["compiled"] + 1), fn(x))[1])
        with torch.no_grad():
            wrapped(torch.ones(2))
        assert calls["compiled"] == 1

    def test_falls_back_to_original_when_grad_is_live(self):
        calls = {"orig": 0, "compiled": 0}

        def fn(x):
            calls["orig"] += 1
            return x * 2

        def compiled(x):
            calls["compiled"] += 1
            return fn(x)

        wrapped = torch_compile._wrap(fn, compiled)
        x = torch.ones(2, requires_grad=True)
        wrapped(x)
        assert calls["orig"] == 1
        assert calls["compiled"] == 0

    def test_grad_enabled_but_no_tensor_requires_grad_still_uses_compiled(self):
        calls = {"compiled": 0}

        def fn(x):
            return x * 2

        def compiled(x):
            calls["compiled"] += 1
            return fn(x)

        wrapped = torch_compile._wrap(fn, compiled)
        wrapped(torch.ones(2))  # grad mode on by default, but tensor doesn't require_grad
        assert calls["compiled"] == 1


class TestCompileFn:
    def test_returns_uncompiled_when_torch_compile_unavailable(self, monkeypatch):
        monkeypatch.setattr(torch_compile, "available", lambda: False)

        def fn(x):
            return x + 1

        result = torch_compile.compile_fn(fn)
        assert result is fn

    def test_compiled_fn_matches_eager_output(self):
        def fn(x, y):
            return x @ y + 1

        wrapped = torch_compile.compile_fn(fn, backend="eager", dynamic=False)
        x, y = torch.randn(3, 4), torch.randn(4, 5)
        with torch.no_grad():
            expected = fn(x, y)
            actual = wrapped(x, y)
        assert torch.allclose(actual, expected)

    def test_fullgraph_env_default_reads_flag(self, monkeypatch):
        monkeypatch.setenv("AMD_TUNED_TORCH_COMPILE_FULLGRAPH", "1")
        captured = {}
        real_compile = torch.compile

        def spy_compile(fn, **kwargs):
            captured.update(kwargs)
            return real_compile(fn, **kwargs)

        monkeypatch.setattr(torch, "compile", spy_compile)
        torch_compile.compile_fn(lambda x: x, backend="eager")
        assert captured["fullgraph"] is True


class TestCompileModule:
    def test_returns_uncompiled_when_torch_compile_unavailable(self, monkeypatch):
        monkeypatch.setattr(torch_compile, "available", lambda: False)
        linear = nn.Linear(4, 2)
        orig_forward = linear.forward
        result = torch_compile.compile_module(linear)
        assert result is linear
        assert linear.forward is orig_forward

    def test_replaces_forward_and_preserves_output(self):
        linear = nn.Linear(4, 2)
        with torch.no_grad():
            linear.weight.fill_(1.0)
            linear.bias.fill_(0.0)
        torch_compile.compile_module(linear, backend="eager", dynamic=False)
        with torch.no_grad():
            out = linear(torch.ones(1, 4))
        assert torch.equal(out, torch.full((1, 2), 4.0))

    def test_falls_back_to_eager_module_under_grad(self):
        linear = nn.Linear(4, 2)
        torch_compile.compile_module(linear, backend="eager", dynamic=False)
        x = torch.ones(1, 4, requires_grad=True)
        out = linear(x)  # must not raise, and must still be a real forward pass
        assert out.shape == (1, 2)
        assert out.requires_grad
