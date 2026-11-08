"""Tests for amd_tuned_torch.te_ops -- the TransformerEngine-backed
replacements amd_tuned_torch monkeypatches over layer_norm/rms_norm/gelu/silu/
scaled_dot_product_attention.

Like the rest of tests/, these run with NO TransformerEngine and no GPU. What
is under test is the adapter layer: which `tex` binding each autograd
Function reaches for, in what argument order, how tensors are reshaped on the
way in and back out, and how the attention wrapper maps PyTorch's
(batch, heads, seq, head_dim) + is_causal/attn_mask calling convention onto
TE's bshd layout + named attn_mask_type strings. `tex` and
`transformer_engine.pytorch` are MagicMocks throughout, so a swapped argument
or a dropped transpose is caught here rather than on hardware nobody has
lying around.

The argument orders asserted below are the ones in
third_party/TransformerEngine/transformer_engine/pytorch/csrc/extensions/pybind.cpp:

    layernorm_fwd(input, weight, bias, eps, ln_out, quantizer, otype,
                  sm_margin, zero_centered_gamma) -> (out, mu, rsigma)
    layernorm_bwd(dz, x, mu, rsigma, gamma, sm_margin, zero_centered_gamma)
                                                 -> (dx, dgamma, dbeta)
    rmsnorm_fwd(input, weight, eps, ln_out, quantizer, otype, sm_margin,
                zero_centered_gamma)             -> (out, None, rsigma)
    rmsnorm_bwd(dz, x, rsigma, gamma, sm_margin, zero_centered_gamma)
                                                 -> (dx, dgamma)
    gelu/silu(input, quantizer);  dgelu/dsilu(grad, fwd_input, quantizer)

These are positional calls in te_ops.py, so a TE-side signature change is
exactly the kind of breakage that otherwise only shows up as a segfault or
silent garbage on a machine with TE installed.

What these tests deliberately do NOT assert is numerics: every kernel
involved is a TE kernel, and re-deriving LayerNorm or softmax in the test
would only test the reimplementation. Real-hardware numerical checks belong
in tests_hardware/.

The env-var gate (AMD_TUNED_TORCH_ENABLE_TE) and is_bottom_right_causal_mask's
pattern matching are covered in tests/test_amd_tuned_torch_monkeypatch.py
(TestTeDisabledByDefault / TestIsBottomRightCausalMask) and are not repeated here.

Run with:

    python -m pytest source/cmp_ext_turing/tests/test_te_ops.py
"""
from __future__ import annotations

import pathlib
from unittest.mock import MagicMock

import pytest
import torch

import amd_tuned_torch.te_ops as te_ops


# transformer_engine.pytorch.constants.TE_DType, minus the fp8 entries te_ops
# never reaches: a plain dict keyed by torch dtype, so TE_DType[dtype] in the
# wrappers resolves to something comparable rather than a MagicMock that
# compares equal to nothing.
FAKE_TE_DTYPE = {
    torch.float32: "kFloat32",
    torch.float16: "kFloat16",
    torch.bfloat16: "kBFloat16",
}


@pytest.fixture
def tex(monkeypatch):
    """Make te_ops believe TE is present, with a MagicMock `tex`.

    Every wrapper resolves `tex`/`TE_DType` from module globals at call time
    (they are module-level names, bound once at import), so substituting them
    here reaches the autograd Functions without reimporting anything.
    """
    fake = MagicMock(name="tex")
    monkeypatch.setattr(te_ops, "_TE_AVAILABLE", True)
    monkeypatch.setattr(te_ops, "tex", fake)
    monkeypatch.setattr(te_ops, "TE_DType", FAKE_TE_DTYPE)
    return fake


@pytest.fixture
def te(monkeypatch):
    """MagicMock stand-in for `transformer_engine.pytorch`, with a
    DotProductAttention that behaves like the real one shape-wise: takes
    bshd q/k/v, returns (batch, seq, heads * head_dim).

    Each constructed module records the kwargs it was built with and the
    (q, k, v, kwargs) of its last forward call, so both the per-config cache
    and the per-call attn_mask_type override can be asserted.
    """
    fake = MagicMock(name="te")

    def factory(**kwargs):
        module = MagicMock(name="DotProductAttention")
        module.init_kwargs = kwargs

        def forward(q, k, v, **fwd_kwargs):
            module.fwd_args = (q, k, v)
            module.fwd_kwargs = fwd_kwargs
            batch, seq, heads, head_dim = q.shape
            # Head-major packing inside the last dim, the layout TE's
            # DotProductAttention actually returns for qkv_format="bshd".
            return torch.arange(
                batch * seq * heads * head_dim, dtype=torch.float32
            ).view(batch, seq, heads * head_dim)

        module.side_effect = forward
        return module

    fake.DotProductAttention.side_effect = factory
    monkeypatch.setattr(te_ops, "_TE_AVAILABLE", True)
    monkeypatch.setattr(te_ops, "te", fake)
    # A fresh cache per test: _attention_cache is module-level and would
    # otherwise carry MagicMock modules across tests.
    monkeypatch.setattr(te_ops, "_attention_cache", {})
    return fake


class TestImportGate:
    """te_ops' central safety property: `import transformer_engine` must be
    reachable only under the AMD_TUNED_TORCH_ENABLE_TE branch. An
    ABI-mismatched TE build segfaults inside its native module-init, which no
    try/except can catch -- so an unconditional top-level import would make
    `import amd_tuned_torch` itself crashable.

    Asserted against the source's own AST rather than sys.modules: by the time
    this runs, other tests in the session have stubbed TE modules into
    sys.modules, so that would measure the suite, not this module. (The
    runtime behaviour of the gate itself is covered by TestTeDisabledByDefault in
    test_amd_tuned_torch_monkeypatch.py, which reloads the module under each
    env-var state.)
    """

    def test_te_is_never_imported_at_module_top_level(self):
        import ast

        tree = ast.parse(pathlib.Path(te_ops.__file__).read_text())
        top_level = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_level += [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                top_level.append(node.module or "")
        assert not [name for name in top_level if "transformer_engine" in name]


class TestLayerNorm:
    def test_forward_binding_and_argument_order(self, tex):
        x = torch.randn(2, 3, 8)
        w = torch.randn(8)
        b = torch.randn(8)
        tex.layernorm_fwd.return_value = (
            torch.zeros(6, 8), torch.zeros(6), torch.zeros(6),
        )

        te_ops.layer_norm(x, (8,), w, b, eps=1e-3)

        args = tex.layernorm_fwd.call_args.args
        assert len(args) == 9
        assert torch.equal(args[0], x.view(-1, 8))
        assert torch.equal(args[1], w)
        assert torch.equal(args[2], b)
        assert args[3] == 1e-3
        # ln_out and quantizer: no preallocated output, no FP8 quantizer.
        assert args[4] is None and args[5] is None
        assert args[6] == FAKE_TE_DTYPE[torch.float32]
        # sm_margin=0 (no SMs reserved for comm overlap), zero_centered_gamma
        # off -- PyTorch's F.layer_norm scales by gamma, not (1 + gamma).
        assert args[7] == 0 and args[8] is False

    def test_input_is_flattened_to_2d_and_output_restored(self, tex):
        x = torch.randn(2, 3, 4, 8)
        tex.layernorm_fwd.return_value = (
            torch.zeros(24, 8), torch.zeros(24), torch.zeros(24),
        )

        out = te_ops.layer_norm(x, (8,), torch.randn(8), torch.randn(8))

        assert tex.layernorm_fwd.call_args.args[0].shape == (24, 8)
        assert out.shape == (2, 3, 4, 8)

    def test_non_contiguous_input_is_made_contiguous(self, tex):
        x = torch.randn(8, 6).t()          # (6, 8), non-contiguous
        assert not x.is_contiguous()
        tex.layernorm_fwd.return_value = (
            torch.zeros(6, 8), torch.zeros(6), torch.zeros(6),
        )

        te_ops.layer_norm(x, (8,), torch.randn(8), torch.randn(8))

        passed = tex.layernorm_fwd.call_args.args[0]
        assert passed.is_contiguous()
        assert torch.equal(passed, x.contiguous())

    def test_normalized_shape_is_ignored_in_favour_of_weight_numel(self, tex):
        """te_ops.layer_norm keeps F.layer_norm's signature but derives the
        inner dim from the weight, since that is what TE's binding needs.
        The positional `normalized_shape` is accepted and dropped."""
        x = torch.randn(4, 8)
        tex.layernorm_fwd.return_value = (
            torch.zeros(4, 8), torch.zeros(4), torch.zeros(4),
        )

        te_ops.layer_norm(x, "ignored", torch.randn(8), torch.randn(8))

        assert tex.layernorm_fwd.call_args.args[0].shape == (4, 8)

    def test_out_dtype_follows_the_weight_not_the_input(self, tex):
        """Mixed precision: an fp32 master-weight LayerNorm over bf16
        activations must produce the weight's dtype tag, matching what the
        module's autocast wrapper expects back."""
        x = torch.randn(4, 8, dtype=torch.bfloat16)
        tex.layernorm_fwd.return_value = (
            torch.zeros(4, 8), torch.zeros(4), torch.zeros(4),
        )

        te_ops.layer_norm(x, (8,), torch.randn(8), torch.randn(8))

        assert tex.layernorm_fwd.call_args.args[6] == FAKE_TE_DTYPE[torch.float32]

    def test_backward_binding_argument_order_and_grad_wiring(self, tex):
        x = torch.randn(2, 3, 8, requires_grad=True)
        w = torch.randn(8, requires_grad=True)
        b = torch.randn(8, requires_grad=True)
        mu, rsigma = torch.zeros(6), torch.ones(6)
        tex.layernorm_fwd.return_value = (torch.zeros(6, 8), mu, rsigma)
        dx, dw, db = torch.full((6, 8), 2.0), torch.full((8,), 3.0), torch.full((8,), 4.0)
        tex.layernorm_bwd.return_value = (dx, dw, db)

        out = te_ops.layer_norm(x, (8,), w, b)
        grad = torch.randn(2, 3, 8)
        out.backward(grad)

        args = tex.layernorm_bwd.call_args.args
        assert len(args) == 7
        assert torch.equal(args[0], grad.view(-1, 8))       # dz, flattened
        assert torch.equal(args[1], x.detach().view(-1, 8))  # the saved 2-D x
        assert torch.equal(args[2], mu)
        assert torch.equal(args[3], rsigma)
        assert torch.equal(args[4], w.detach())
        assert args[5] == 0 and args[6] is False
        # dx comes back 2-D from TE and must be restored to the input's shape,
        # or autograd rejects it.
        assert torch.equal(x.grad, dx.view(2, 3, 8))
        assert torch.equal(w.grad, dw)
        assert torch.equal(b.grad, db)


class TestRMSNorm:
    def test_forward_binding_and_argument_order(self, tex):
        x = torch.randn(2, 3, 8)
        w = torch.randn(8)
        tex.rmsnorm_fwd.return_value = (torch.zeros(6, 8), None, torch.zeros(6))

        te_ops.rms_norm(x, (8,), w, eps=1e-3)

        args = tex.rmsnorm_fwd.call_args.args
        assert len(args) == 8
        assert torch.equal(args[0], x.view(-1, 8))
        assert torch.equal(args[1], w)
        assert args[2] == 1e-3
        assert args[3] is None and args[4] is None
        assert args[5] == FAKE_TE_DTYPE[torch.float32]
        assert args[6] == 0 and args[7] is False

    def test_none_mean_from_te_is_tolerated(self, tex):
        """rmsnorm_fwd returns py::none() in the mu slot (RMSNorm has no
        mean). Saving that into ctx would raise, so it must be dropped, not
        stashed."""
        tex.rmsnorm_fwd.return_value = (torch.zeros(4, 8), None, torch.zeros(4))

        out = te_ops.rms_norm(torch.randn(4, 8), (8,), torch.randn(8))

        assert out.shape == (4, 8)

    def test_eps_defaults_to_the_input_dtypes_epsilon(self, tex):
        tex.rmsnorm_fwd.return_value = (torch.zeros(4, 8), None, torch.zeros(4))

        te_ops.rms_norm(torch.randn(4, 8, dtype=torch.float16), (8,), torch.randn(8))

        assert tex.rmsnorm_fwd.call_args.args[2] == torch.finfo(torch.float16).eps

    def test_explicit_eps_of_zero_is_not_replaced_by_the_default(self, tex):
        """`if eps is None`, not `if not eps` -- eps=0.0 is a legitimate (if
        unwise) caller choice and must reach TE unchanged."""
        tex.rmsnorm_fwd.return_value = (torch.zeros(4, 8), None, torch.zeros(4))

        te_ops.rms_norm(torch.randn(4, 8), (8,), torch.randn(8), eps=0.0)

        assert tex.rmsnorm_fwd.call_args.args[2] == 0.0

    def test_backward_binding_argument_order_and_grad_wiring(self, tex):
        x = torch.randn(2, 3, 8, requires_grad=True)
        w = torch.randn(8, requires_grad=True)
        rsigma = torch.ones(6)
        tex.rmsnorm_fwd.return_value = (torch.zeros(6, 8), None, rsigma)
        dx, dw = torch.full((6, 8), 2.0), torch.full((8,), 3.0)
        tex.rmsnorm_bwd.return_value = (dx, dw)

        out = te_ops.rms_norm(x, (8,), w)
        grad = torch.randn(2, 3, 8)
        out.backward(grad)

        args = tex.rmsnorm_bwd.call_args.args
        assert len(args) == 6
        assert torch.equal(args[0], grad.view(-1, 8))
        assert torch.equal(args[1], x.detach().view(-1, 8))
        assert torch.equal(args[2], rsigma)     # no mu slot, unlike layernorm_bwd
        assert torch.equal(args[3], w.detach())
        assert args[4] == 0 and args[5] is False
        assert torch.equal(x.grad, dx.view(2, 3, 8))
        assert torch.equal(w.grad, dw)


class TestActivations:
    """gelu/silu take (input, quantizer); their backwards take
    (grad, fwd_input, quantizer) -- grad FIRST, which is the easy one to get
    backwards since forward's own argument is the input."""

    @pytest.mark.parametrize("fn_name,fwd_binding", [("gelu", "gelu"), ("silu", "silu")])
    def test_forward_binding(self, tex, fn_name, fwd_binding):
        x = torch.randn(4, 8)
        getattr(tex, fwd_binding).return_value = torch.zeros(4, 8)

        getattr(te_ops, fn_name)(x)

        args = getattr(tex, fwd_binding).call_args.args
        assert len(args) == 2
        assert torch.equal(args[0], x)
        assert args[1] is None      # no FP8 quantizer

    @pytest.mark.parametrize(
        "fn_name,fwd_binding,bwd_binding",
        [("gelu", "gelu", "dgelu"), ("silu", "silu", "dsilu")],
    )
    def test_backward_passes_grad_before_input(self, tex, fn_name, fwd_binding, bwd_binding):
        x = torch.randn(4, 8, requires_grad=True)
        getattr(tex, fwd_binding).return_value = torch.zeros(4, 8)
        dx = torch.full((4, 8), 7.0)
        getattr(tex, bwd_binding).return_value = dx

        out = getattr(te_ops, fn_name)(x)
        grad = torch.randn(4, 8)
        out.backward(grad)

        args = getattr(tex, bwd_binding).call_args.args
        assert len(args) == 3
        assert torch.equal(args[0], grad)           # dy
        assert torch.equal(args[1], x.detach())     # the saved forward input
        assert args[2] is None
        assert torch.equal(x.grad, dx)

    @pytest.mark.parametrize("fn_name,binding", [("gelu", "gelu"), ("silu", "silu")])
    def test_non_contiguous_input_is_made_contiguous(self, tex, fn_name, binding):
        x = torch.randn(8, 4).t()
        getattr(tex, binding).return_value = torch.zeros(4, 8)

        getattr(te_ops, fn_name)(x)

        assert getattr(tex, binding).call_args.args[0].is_contiguous()


class TestAttentionModuleCache:
    def test_module_is_built_with_the_expected_static_config(self, te):
        q = k = v = torch.randn(2, 4, 6, 32)

        te_ops.scaled_dot_product_attention(q, k, v, scale=0.125)

        kwargs = te.DotProductAttention.call_args.kwargs
        assert kwargs["num_attention_heads"] == 4
        assert kwargs["kv_channels"] == 32
        assert kwargs["attention_dropout"] == 0.0
        # bshd is what the transposes below feed it; "no_mask" is only the
        # construction-time default, overridden per forward() call.
        assert kwargs["qkv_format"] == "bshd"
        assert kwargs["attn_mask_type"] == "no_mask"
        assert kwargs["softmax_scale"] == 0.125

    def test_same_config_reuses_one_module(self, te):
        q = k = v = torch.randn(2, 4, 6, 32)

        te_ops.scaled_dot_product_attention(q, k, v)
        te_ops.scaled_dot_product_attention(q, k, v)

        assert te.DotProductAttention.call_count == 1

    def test_causal_and_non_causal_share_one_module(self, te):
        """attn_mask_type is a per-forward() override, so it must not be part
        of the cache key -- otherwise a model alternating prefill and decode
        would build a second module for no reason."""
        q = k = v = torch.randn(2, 4, 6, 32)

        te_ops.scaled_dot_product_attention(q, k, v, is_causal=True)
        te_ops.scaled_dot_product_attention(q, k, v, is_causal=False)

        assert te.DotProductAttention.call_count == 1

    @pytest.mark.parametrize(
        "second",
        [
            {"heads": 8},        # different head count
            {"head_dim": 64},    # different head dim
            {"scale": 0.5},      # different softmax scale
        ],
    )
    def test_differing_config_builds_a_second_module(self, te, second):
        heads = second.get("heads", 4)
        head_dim = second.get("head_dim", 32)
        scale = second.get("scale", None)
        base = torch.randn(2, 4, 6, 32)
        other = torch.randn(2, heads, 6, head_dim)

        te_ops.scaled_dot_product_attention(base, base, base)
        te_ops.scaled_dot_product_attention(other, other, other, scale=scale)

        assert te.DotProductAttention.call_count == 2


class TestAttentionForward:
    def test_qkv_are_transposed_into_bshd_and_contiguous(self, te):
        q = torch.randn(2, 4, 6, 32)
        k = torch.randn(2, 4, 6, 32)
        v = torch.randn(2, 4, 6, 32)

        te_ops.scaled_dot_product_attention(q, k, v)

        module = te_ops._attention_cache[(4, 32, None)]
        passed_q, passed_k, passed_v = module.fwd_args
        for passed, original in ((passed_q, q), (passed_k, k), (passed_v, v)):
            assert passed.shape == (2, 6, 4, 32)
            assert passed.is_contiguous()
            assert torch.equal(passed, original.transpose(1, 2).contiguous())

    def test_output_is_unpacked_back_to_bhsd(self, te):
        q = k = v = torch.randn(2, 4, 6, 32)

        out = te_ops.scaled_dot_product_attention(q, k, v)

        assert out.shape == (2, 4, 6, 32)
        module = te_ops._attention_cache[(4, 32, None)]
        te_out = torch.arange(2 * 6 * 4 * 32, dtype=torch.float32).view(2, 6, 4 * 32)
        # TE packs heads inside the last dim (head-major); each (batch, head,
        # seq) row must come back as that head's slice of the packed output.
        for head in range(4):
            assert torch.equal(out[1, head, 3], te_out[1, 3, head * 32:(head + 1) * 32])
        assert module.fwd_kwargs["attn_mask_type"] == "no_mask"

    @pytest.mark.parametrize(
        "kwargs,expected",
        [
            ({}, "no_mask"),
            ({"is_causal": True}, "causal"),
            ({"attn_mask": torch.tril(torch.ones(6, 6, dtype=torch.bool))},
             "causal_bottom_right"),
            # A mask wins over is_causal: the caller only passes a mask that
            # already matched the bottom-right pattern, which is the stricter
            # (correct) of the two when q_len < kv_len.
            ({"attn_mask": torch.tril(torch.ones(6, 6, dtype=torch.bool)), "is_causal": True},
             "causal_bottom_right"),
        ],
    )
    def test_attn_mask_type_selection(self, te, kwargs, expected):
        q = k = v = torch.randn(2, 4, 6, 32)

        te_ops.scaled_dot_product_attention(q, k, v, **kwargs)

        module = te_ops._attention_cache[(4, 32, None)]
        assert module.fwd_kwargs["attn_mask_type"] == expected

    def test_the_mask_tensor_itself_is_never_forwarded_to_te(self, te):
        """Forwarding a tensor would force attn_mask_type="arbitrary", which
        disables TE's fused backends entirely -- the whole point of the
        named-string mapping above."""
        q = k = v = torch.randn(2, 4, 6, 32)
        mask = torch.tril(torch.ones(6, 6, dtype=torch.bool))

        te_ops.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        module = te_ops._attention_cache[(4, 32, None)]
        assert "attn_mask" not in module.fwd_kwargs
        assert not any(
            isinstance(a, torch.Tensor) and a.dtype == torch.bool
            for a in module.fwd_args
        )

    def test_kv_longer_than_q_keeps_the_querys_sequence_length(self, te):
        """KV-cache decode: q_len=1 against kv_len=9. The output is one token
        per query position, not per key."""
        q = torch.randn(2, 4, 1, 32)
        k = v = torch.randn(2, 4, 9, 32)

        out = te_ops.scaled_dot_product_attention(q, k, v, is_causal=True)

        module = te_ops._attention_cache[(4, 32, None)]
        assert module.fwd_args[0].shape == (2, 1, 4, 32)
        assert module.fwd_args[1].shape == (2, 9, 4, 32)
        assert out.shape == (2, 4, 1, 32)

    def test_gqa_module_is_sized_from_the_querys_head_count(self, te):
        """Grouped-query attention: k/v carry fewer heads than q. TE expands
        them internally, so num_attention_heads/kv_channels come from the
        query."""
        q = torch.randn(2, 8, 6, 32)
        k = v = torch.randn(2, 2, 6, 32)

        out = te_ops.scaled_dot_product_attention(q, k, v)

        kwargs = te.DotProductAttention.call_args.kwargs
        assert kwargs["num_attention_heads"] == 8
        assert out.shape == (2, 8, 6, 32)


class TestCausalReferenceCache:
    """is_bottom_right_causal_mask's own matching is covered in
    test_amd_tuned_torch_monkeypatch.py; what is checked here is that the
    reference pattern is built once per (q_len, kv_len, device), since the
    predicate runs on every single patched SDPA call."""

    def test_reference_is_cached_per_shape_and_device(self, monkeypatch):
        monkeypatch.setattr(te_ops, "_causal_reference_cache", {})
        first = te_ops._bottom_right_causal_reference(4, 6, torch.device("cpu"))
        second = te_ops._bottom_right_causal_reference(4, 6, torch.device("cpu"))
        assert second is first
        assert len(te_ops._causal_reference_cache) == 1

        te_ops._bottom_right_causal_reference(4, 7, torch.device("cpu"))
        assert len(te_ops._causal_reference_cache) == 2

    def test_reference_is_bottom_right_aligned(self, monkeypatch):
        monkeypatch.setattr(te_ops, "_causal_reference_cache", {})
        ref = te_ops._bottom_right_causal_reference(2, 5, torch.device("cpu"))
        # q_len=2, kv_len=5: query 0 sees keys 0..3, query 1 sees keys 0..4.
        assert ref.tolist() == [
            [True, True, True, True, False],
            [True, True, True, True, True],
        ]

    def test_repeated_matching_does_not_grow_the_cache(self, monkeypatch):
        monkeypatch.setattr(te_ops, "_causal_reference_cache", {})
        mask = torch.tril(torch.ones(8, 8, dtype=torch.bool))
        for _ in range(5):
            assert te_ops.is_bottom_right_causal_mask(mask) is True
        assert len(te_ops._causal_reference_cache) == 1
