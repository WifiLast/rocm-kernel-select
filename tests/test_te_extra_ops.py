"""Tests for amd_tuned_torch.te_extra_ops -- the TransformerEngine kernels
this package exposes as an explicit API rather than as monkeypatches.

Like the rest of tests/, these run with NO TransformerEngine and no GPU: what
is under test is the Python layer -- which tex binding each entry point
reaches for, in what argument order, and the shape/dtype rules that decide
whether the fused kernel is eligible at all. `tex` is a MagicMock throughout,
so a wrong argument order is caught here rather than on hardware nobody has
lying around.

The one thing these tests deliberately do NOT assert is numerics: every
kernel involved is a TE kernel, and re-deriving softmax or Adam in the test
would only test the reimplementation. Real-hardware numerical checks belong
in tests_hardware/.

Run with:

    python -m pytest source/cmp_ext_turing/tests/test_te_extra_ops.py
"""
from __future__ import annotations

import pathlib
import sys
import types
from unittest.mock import MagicMock

import pytest
import torch

import amd_tuned_torch.te_extra_ops as te_extra_ops
import amd_tuned_torch.te_ops as te_ops


@pytest.fixture
def fake_tex(monkeypatch):
    """Make te_extra_ops believe TE is present, with a MagicMock `tex`.

    te_extra_ops reads `te_ops.tex` through its own `_tex()` on every call
    rather than capturing it at import, precisely so this kind of
    substitution works without reimporting anything.
    """
    tex = MagicMock()
    monkeypatch.setattr(te_ops, "_TE_AVAILABLE", True)
    monkeypatch.setattr(te_ops, "tex", tex)
    return tex


class TestAvailability:
    """te_extra_ops must never open a second TE import gate -- it defers to
    te_ops so AMD_TUNED_TORCH_ENABLE_TE governs both modules at once."""

    def test_available_mirrors_te_ops(self, monkeypatch):
        monkeypatch.setattr(te_ops, "_TE_AVAILABLE", False)
        assert te_extra_ops.available() is False
        monkeypatch.setattr(te_ops, "_TE_AVAILABLE", True)
        assert te_extra_ops.available() is True

    def test_entry_points_raise_a_useful_error_without_te(self, monkeypatch):
        monkeypatch.setattr(te_ops, "_TE_AVAILABLE", False)
        with pytest.raises(RuntimeError, match="AMD_TUNED_TORCH_ENABLE_TE"):
            te_extra_ops.multi_tensor_scale([torch.zeros(2)], [torch.zeros(2)], 1.0)

    def test_the_module_has_no_import_time_te_reference(self):
        """te_extra_ops is imported at package import time (__init__.py), so a
        module-level `import transformer_engine` would defeat te_ops' whole
        opt-in design -- the crash that gate exists to prevent happens inside
        TE's native module-init, which no try/except can catch.

        Asserted against the source's own AST rather than sys.modules: by the
        time this runs, other tests in the session have stubbed TE modules
        into sys.modules, so that would measure the suite, not this module.
        The only permitted TE import is the function-local one inside
        copy_to_kv_cache.
        """
        import ast

        tree = ast.parse(pathlib.Path(te_extra_ops.__file__).read_text())
        top_level_imports = [
            alias.name
            for node in tree.body
            for alias in (node.names if isinstance(node, (ast.Import, ast.ImportFrom)) else [])
        ]
        top_level_imports += [
            node.module
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module
        ]
        assert not any(
            name and name.startswith("transformer_engine") for name in top_level_imports
        ), f"te_extra_ops must not import TE at module scope; found {top_level_imports}"


class TestMultiTensorListHandling:
    """Every multi-tensor kernel indexes all of its lists with one shared
    chunk index, so the lists have to line up and empty slots have to go."""

    def test_empty_numel_tensors_are_dropped_from_every_list(self, fake_tex):
        grads = [torch.zeros(4), torch.zeros(0), torch.zeros(4)]
        params = [torch.zeros(4), torch.zeros(0), torch.zeros(4)]
        m = [torch.zeros(4), torch.zeros(0), torch.zeros(4)]
        v = [torch.zeros(4), torch.zeros(0), torch.zeros(4)]

        te_extra_ops.multi_tensor_adam(grads, params, m, v, lr=1e-3, step=1)

        tensor_lists = fake_tex.multi_tensor_adam.call_args[0][2]
        assert all(len(lst) == 2 for lst in tensor_lists), (
            "the zero-element slot must be dropped from all four lists together"
        )

    def test_misaligned_lists_are_rejected(self, fake_tex):
        with pytest.raises(RuntimeError, match="aligned multi-tensor lists"):
            te_extra_ops.multi_tensor_adam(
                [torch.zeros(4)], [torch.zeros(4), torch.zeros(4)],
                [torch.zeros(4)], [torch.zeros(4)], lr=1e-3, step=1,
            )

    def test_an_all_empty_group_is_a_no_op(self, fake_tex):
        te_extra_ops.multi_tensor_adam([], [], [], [], lr=1e-3, step=1)
        fake_tex.multi_tensor_adam.assert_not_called()

    def test_noop_flag_is_cached_zero_int32(self, fake_tex):
        device = torch.zeros(1).device
        first = te_extra_ops._noop_flag(device)
        second = te_extra_ops._noop_flag(device)
        assert first is second, "a per-step allocation would undo part of the win"
        assert first.dtype == torch.int32
        assert int(first.item()) == 0


class TestMultiTensorAdam:
    """Argument order here is positional in TE's kernel and matches
    FusedAdam.step()'s tensor_lists: grads, params, exp_avg, exp_avg_sq."""

    def test_tensor_list_order_and_scalar_arguments(self, fake_tex):
        grads = [torch.zeros(4)]
        params = [torch.ones(4)]
        exp_avg = [torch.zeros(4)]
        exp_avg_sq = [torch.zeros(4)]

        te_extra_ops.multi_tensor_adam(
            grads, params, exp_avg, exp_avg_sq,
            lr=1e-3, beta1=0.8, beta2=0.95, eps=1e-6, step=7,
            adam_w_mode=True, bias_correction=True, weight_decay=0.01,
        )

        args = fake_tex.multi_tensor_adam.call_args[0]
        assert args[0] == te_extra_ops._CHUNK_SIZE
        tensor_lists = args[2]
        assert [t[0] for t in tensor_lists] == [grads[0], params[0], exp_avg[0], exp_avg_sq[0]]
        assert args[3:] == (1e-3, 0.8, 0.95, 1e-6, 7, 1, 1, 0.01)

    def test_adam_w_mode_and_bias_correction_become_ints(self, fake_tex):
        te_extra_ops.multi_tensor_adam(
            [torch.zeros(4)], [torch.zeros(4)], [torch.zeros(4)], [torch.zeros(4)],
            lr=1e-3, step=1, adam_w_mode=False, bias_correction=False,
        )
        args = fake_tex.multi_tensor_adam.call_args[0]
        assert args[8] == 0, "adam_w_mode=False is TE's mode 0 (L2 regularization)"
        assert args[9] == 0

    def test_master_params_become_a_fifth_list(self, fake_tex):
        master = [torch.zeros(4, dtype=torch.float32)]
        te_extra_ops.multi_tensor_adam(
            [torch.zeros(4)], [torch.zeros(4)], [torch.zeros(4)], [torch.zeros(4)],
            lr=1e-3, step=1, master_params=master,
        )
        tensor_lists = fake_tex.multi_tensor_adam.call_args[0][2]
        assert len(tensor_lists) == 5
        assert tensor_lists[4][0] is master[0]


class TestMultiTensorSgd:
    def test_list_order_is_grads_params_momentums(self, fake_tex):
        grads, params, momentums = [torch.zeros(4)], [torch.ones(4)], [torch.zeros(4)]
        te_extra_ops.multi_tensor_sgd(
            grads, params, momentums,
            lr=0.1, momentum=0.9, dampening=0.0, weight_decay=1e-4,
            nesterov=True, first_run=True, wd_after_momentum=True, scale=0.5,
        )
        args = fake_tex.multi_tensor_sgd.call_args[0]
        assert [t[0] for t in args[2]] == [grads[0], params[0], momentums[0]]
        assert args[3:] == (1e-4, 0.9, 0.0, 0.1, True, True, True, 0.5)


class TestMultiTensorNormAndScale:
    def test_l2norm_without_per_tensor_returns_none_for_the_second_slot(self, fake_tex):
        fake_tex.multi_tensor_l2norm.return_value = (torch.tensor(2.0), torch.zeros(3))
        total, per_tensor = te_extra_ops.multi_tensor_l2norm([torch.zeros(4)])
        assert per_tensor is None
        assert float(total) == 2.0

    def test_l2norm_with_per_tensor_passes_it_through(self, fake_tex):
        expected = torch.zeros(3)
        fake_tex.multi_tensor_l2norm.return_value = (torch.tensor(2.0), expected)
        _, per_tensor = te_extra_ops.multi_tensor_l2norm([torch.zeros(4)], per_tensor=True)
        assert per_tensor is expected
        assert fake_tex.multi_tensor_l2norm.call_args[0][3] is True

    def test_inv_scale_routes_to_the_unscale_variant(self, fake_tex):
        """The AMP case: report the true gradient norm while the stored
        gradients are still loss-scaled, without writing them back."""
        fake_tex.multi_tensor_unscale_l2norm.return_value = (torch.tensor(1.0), None)
        inv_scale = torch.tensor([0.5])
        te_extra_ops.multi_tensor_l2norm([torch.zeros(4)], inv_scale=inv_scale)
        fake_tex.multi_tensor_l2norm.assert_not_called()
        assert fake_tex.multi_tensor_unscale_l2norm.call_args[0][3] is inv_scale

    def test_scale_passes_input_and_output_lists_in_order(self, fake_tex):
        src, dst = [torch.ones(4)], [torch.zeros(4)]
        te_extra_ops.multi_tensor_scale(src, dst, 0.25)
        args = fake_tex.multi_tensor_scale.call_args[0]
        assert [t[0] for t in args[2]] == [src[0], dst[0]]
        assert args[3] == 0.25

    def test_a_caller_supplied_noop_flag_is_used_verbatim(self, fake_tex):
        """An AMP scaler reads its own buffer back after the call to learn
        whether an overflow was seen -- it has to be the same tensor."""
        flag = torch.zeros(1, dtype=torch.int32)
        te_extra_ops.multi_tensor_scale([torch.ones(4)], [torch.zeros(4)], 2.0,
                                        noop_flag=flag)
        assert fake_tex.multi_tensor_scale.call_args[0][1] is flag


class TestSoftmaxBatchPerBlock:
    """Pure launch-geometry arithmetic, reproduced from TE's
    get_batch_per_block. Wrong values here silently make kernel_available
    reject shapes the kernel handles fine (or accept ones it doesn't)."""

    @pytest.mark.parametrize(
        "key_seq_len,expected",
        [
            (16, 2 * (128 // 16)),    # pow2=16 < 32 -> warp_size 16, 2 batches/warp
            (128, 2 * (128 // 32)),   # pow2=128 -> warp_size 32, still 2 batches/warp
            (256, 1 * (128 // 32)),   # pow2 > 128 -> 1 batch/warp
            (2048, 1 * (128 // 32)),
        ],
    )
    def test_matches_tes_formula(self, key_seq_len, expected):
        assert te_extra_ops.batch_per_block(key_seq_len) == expected


class TestSoftmaxKernelAvailability:
    """The bindings validate with AT_ASSERTM, which aborts into a C++
    exception rather than returning a status -- so every constraint has to be
    caught here, before the call. Mirrors TE's own is_kernel_available."""

    def _scores(self, b=2, np_=4, sq=128, sk=128, dtype=torch.float16):
        return torch.zeros(b, np_, sq, sk, dtype=dtype)

    def test_accepts_a_well_formed_shape(self):
        assert te_extra_ops.kernel_available(self._scores()) is True

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_rejects_non_half_dtypes(self, dtype):
        assert te_extra_ops.kernel_available(self._scores(dtype=dtype)) is False

    def test_rejects_non_4d_input(self):
        assert te_extra_ops.kernel_available(torch.zeros(4, 128, 128, dtype=torch.float16)) is False

    @pytest.mark.parametrize("sk", [16, 16384, 20000])
    def test_rejects_key_lengths_outside_the_supported_range(self, sk):
        assert te_extra_ops.kernel_available(self._scores(sk=sk)) is False

    def test_rejects_key_length_not_divisible_by_eight(self):
        assert te_extra_ops.kernel_available(self._scores(sk=124)) is False

    def test_rejects_single_query_row(self):
        """sq == 1 is the decode step -- the kernel asserts sq > 1, so a
        decode-shaped call has to go to torch."""
        assert te_extra_ops.kernel_available(self._scores(sq=1)) is False

    def test_rejects_top_left_causal_on_a_non_square_score_matrix(self):
        assert te_extra_ops.kernel_available(
            self._scores(sq=128, sk=256), attn_mask_type="causal"
        ) is False

    def test_accepts_bottom_right_causal_on_a_non_square_score_matrix(self):
        """causal_bottom_right is exactly the KV-cache shape (sq < sk), and
        unlike plain causal it carries no square requirement."""
        assert te_extra_ops.kernel_available(
            self._scores(sq=128, sk=256), attn_mask_type="causal_bottom_right"
        ) is True

    def test_rejects_query_length_not_divisible_by_four(self):
        assert te_extra_ops.kernel_available(self._scores(sq=126)) is False

    def test_rejects_attn_batches_not_divisible_by_four(self):
        assert te_extra_ops.kernel_available(self._scores(b=1, np_=2)) is False

    def test_rejects_query_length_not_divisible_by_batch_per_block(self):
        # sk=2048 -> batch_per_block 4; sq=4 is divisible by 4 but the point
        # is that the rule is applied at all, so pick an sq that fails it.
        sk = 256  # batch_per_block == 4
        assert te_extra_ops.batch_per_block(sk) == 4
        assert te_extra_ops.kernel_available(self._scores(sq=4, sk=sk)) is True
        assert te_extra_ops.kernel_available(self._scores(sq=6, sk=sk)) is False

    def test_padding_requires_a_correctly_shaped_mask(self):
        scores = self._scores()
        assert te_extra_ops.kernel_available(scores, None, "padding") is False
        good = torch.zeros(2, 1, 128, 128, dtype=torch.bool)
        assert te_extra_ops.kernel_available(scores, good, "padding") is True
        broadcast = torch.zeros(1, 1, 128, 128, dtype=torch.bool)
        assert te_extra_ops.kernel_available(scores, broadcast, "padding") is True
        wrong = torch.zeros(2, 4, 128, 128, dtype=torch.bool)
        assert te_extra_ops.kernel_available(scores, wrong, "padding") is False

    def test_never_raises_on_a_nonsense_shape(self):
        """A False here is a routing decision, not an error -- callers use it
        to choose a path, so it must not throw on anything."""
        assert te_extra_ops.kernel_available(torch.zeros(0), None, "arbitrary") is False


class TestSoftmaxDispatch:
    def test_mask_plus_mask_type_selects_the_masked_kernel(self, fake_tex):
        scores = torch.zeros(2, 4, 128, 128, dtype=torch.float16)
        mask = torch.zeros(2, 1, 128, 128, dtype=torch.bool)
        fake_tex.scaled_masked_softmax_forward.return_value = scores
        te_extra_ops.scaled_softmax(scores, mask, 0.5, "padding")
        fake_tex.scaled_masked_softmax_forward.assert_called_once_with(scores, mask, 0.5)
        fake_tex.scaled_softmax_forward.assert_not_called()

    def test_no_mask_selects_the_plain_kernel(self, fake_tex):
        scores = torch.zeros(2, 4, 128, 128, dtype=torch.float16)
        fake_tex.scaled_softmax_forward.return_value = scores
        te_extra_ops.scaled_softmax(scores, None, 0.5, "no_mask")
        fake_tex.scaled_softmax_forward.assert_called_once_with(scores, 0.5)

    def test_causal_does_not_reach_the_aligned_kernel_by_default(self, fake_tex):
        """TE itself commented that dispatch out ("Disable for now until
        unalignment bug is fixed"), so mirroring TE means NOT routing causal
        there until that is resolved upstream."""
        scores = torch.zeros(2, 4, 128, 128, dtype=torch.float16)
        fake_tex.scaled_softmax_forward.return_value = scores
        te_extra_ops.scaled_softmax(scores, None, 1.0, "causal")
        fake_tex.scaled_aligned_causal_masked_softmax_forward.assert_not_called()
        fake_tex.scaled_softmax_forward.assert_called_once()

    def test_the_aligned_kernel_is_reachable_on_explicit_opt_in(self, fake_tex):
        scores = torch.zeros(2, 4, 128, 128, dtype=torch.float16)
        fake_tex.scaled_aligned_causal_masked_softmax_forward.return_value = scores
        te_extra_ops.scaled_softmax(scores, None, 1.0, "causal",
                                    use_aligned_causal_kernel=True)
        fake_tex.scaled_aligned_causal_masked_softmax_forward.assert_called_once_with(
            scores, 1.0
        )

    def test_upper_triang_entry_point_is_separate(self, fake_tex):
        """It takes a 3D (b*np, s, s) tensor, unlike every other variant --
        folding it into the 4D dispatch would hide a reshape."""
        scores = torch.zeros(8, 128, 128, dtype=torch.float16)
        fake_tex.scaled_upper_triang_masked_softmax_forward.return_value = scores
        te_extra_ops.upper_triang_causal_softmax(scores, 0.25)
        fake_tex.scaled_upper_triang_masked_softmax_forward.assert_called_once_with(
            scores, 0.25
        )


class TestSoftmaxAutograd:
    """The backward kernels take the softmax OUTPUT, not the input -- the
    Jacobian is expressible in terms of the result. Passing the input would
    be silently wrong, so it is asserted."""

    def test_backward_passes_the_saved_output_not_the_input(self, fake_tex):
        scores = torch.randn(2, 4, 128, 128, dtype=torch.float16, requires_grad=True)
        out = torch.rand(2, 4, 128, 128, dtype=torch.float16)
        fake_tex.scaled_softmax_forward.return_value = out
        fake_tex.scaled_softmax_backward.return_value = torch.zeros_like(out)

        y = te_extra_ops.scaled_softmax(scores, None, 0.5, "no_mask")
        y.sum().backward()

        bwd_args = fake_tex.scaled_softmax_backward.call_args[0]
        assert torch.equal(bwd_args[1], out)
        assert bwd_args[2] == 0.5

    def test_masked_backward_does_not_need_the_mask(self, fake_tex):
        """Masked positions are already exactly zero in the output, so their
        gradient falls out as zero -- TE's backward takes no mask argument."""
        scores = torch.randn(2, 4, 128, 128, dtype=torch.float16, requires_grad=True)
        mask = torch.zeros(2, 1, 128, 128, dtype=torch.bool)
        out = torch.rand(2, 4, 128, 128, dtype=torch.float16)
        fake_tex.scaled_masked_softmax_forward.return_value = out
        fake_tex.scaled_masked_softmax_backward.return_value = torch.zeros_like(out)

        te_extra_ops.scaled_softmax(scores, mask, 1.0, "padding").sum().backward()
        assert len(fake_tex.scaled_masked_softmax_backward.call_args[0]) == 3


class TestSoftmaxFallbackWrapper:
    """scaled_softmax_or_torch is the routing wrapper -- it must never raise
    for an ineligible shape, only cost an eligibility check."""

    def test_falls_back_to_torch_when_te_is_absent(self, monkeypatch):
        monkeypatch.setattr(te_ops, "_TE_AVAILABLE", False)
        scores = torch.randn(2, 4, 8, 8, dtype=torch.float16)
        out = te_extra_ops.scaled_softmax_or_torch(scores, scale=0.5)
        expected = torch.softmax(scores.float() * 0.5, dim=-1).to(scores.dtype)
        assert torch.allclose(out, expected, atol=1e-3)

    def test_falls_back_for_an_ineligible_shape_even_with_te(self, fake_tex):
        scores = torch.randn(2, 4, 1, 64, dtype=torch.float16)  # sq == 1
        te_extra_ops.scaled_softmax_or_torch(scores, scale=1.0)
        fake_tex.scaled_softmax_forward.assert_not_called()

    def test_uses_the_kernel_when_eligible(self, fake_tex):
        scores = torch.randn(2, 4, 128, 128, dtype=torch.float16)
        fake_tex.scaled_softmax_forward.return_value = scores
        te_extra_ops.scaled_softmax_or_torch(scores, scale=1.0)
        fake_tex.scaled_softmax_forward.assert_called_once()

    def test_fallback_computes_in_fp32_and_casts_back(self):
        """Matches TE's forward_torch_softmax: accumulating a long row in
        fp16 loses precision the fused kernel does not."""
        scores = torch.randn(2, 4, 8, 8, dtype=torch.float16)
        out = te_extra_ops.scaled_softmax_or_torch(scores)
        assert out.dtype == torch.float16


class TestDropout:
    def test_is_the_identity_when_not_training(self, fake_tex):
        x = torch.randn(4, 4)
        assert te_extra_ops.dropout(x, 0.5, training=False) is x
        fake_tex.dropout_fwd.assert_not_called()

    def test_is_the_identity_at_probability_zero(self, fake_tex):
        x = torch.randn(4, 4)
        assert te_extra_ops.dropout(x, 0.0) is x
        fake_tex.dropout_fwd.assert_not_called()

    @pytest.mark.parametrize("p", [-0.1, 1.0, 1.5])
    def test_rejects_out_of_range_probabilities(self, p, fake_tex):
        with pytest.raises(ValueError, match="dropout probability"):
            te_extra_ops.dropout(torch.randn(4, 4), p)

    def test_forward_saves_the_packed_mask_for_backward(self, fake_tex):
        """TE's mask is a bit per element, not a full-width tensor -- it is
        only meaningful to tex.dropout_bwd, and has to survive to it."""
        x = torch.randn(4, 4, requires_grad=True)
        out = torch.randn(4, 4)
        mask = torch.zeros(2, dtype=torch.uint8)
        fake_tex.dropout_fwd.return_value = (out, mask)
        fake_tex.dropout_bwd.return_value = torch.zeros(4, 4)

        te_extra_ops.dropout(x, 0.1).sum().backward()

        bwd_args = fake_tex.dropout_bwd.call_args[0]
        assert torch.equal(bwd_args[1], mask)
        assert bwd_args[2] == 0.1


class TestSequenceLayout:
    def test_thd_to_bshd_derives_batch_size_from_cu_seqlens(self, fake_tex):
        """cu_seqlens has batch_size + 1 entries; getting that off by one
        would mis-shape the whole output."""
        thd = torch.zeros(10, 4, 8)
        cu_seqlens = torch.tensor([0, 4, 10], dtype=torch.int32)
        fake_tex.convert_thd_to_bshd.return_value = torch.zeros(2, 6, 4, 8)

        te_extra_ops.thd_to_bshd(thd, cu_seqlens, 6)

        args = fake_tex.convert_thd_to_bshd.call_args[0]
        assert args[2] == 2, "batch_size = len(cu_seqlens) - 1"
        assert args[3] == 6

    def test_thd_to_bshd_backward_is_the_inverse_conversion(self, fake_tex):
        thd = torch.zeros(10, 4, 8, requires_grad=True)
        cu_seqlens = torch.tensor([0, 4, 10], dtype=torch.int32)
        fake_tex.convert_thd_to_bshd.return_value = torch.zeros(2, 6, 4, 8)
        fake_tex.convert_bshd_to_thd.return_value = torch.zeros(10, 4, 8)

        te_extra_ops.thd_to_bshd(thd, cu_seqlens, 6).sum().backward()

        args = fake_tex.convert_bshd_to_thd.call_args[0]
        assert args[2] == 10, "the token count must come back from the forward"

    def test_bshd_to_thd_reads_the_token_count_from_cu_seqlens(self, fake_tex):
        bshd = torch.zeros(2, 6, 4, 8)
        cu_seqlens = torch.tensor([0, 4, 10], dtype=torch.int32)
        fake_tex.convert_bshd_to_thd.return_value = torch.zeros(10, 4, 8)

        te_extra_ops.bshd_to_thd(bshd, cu_seqlens)

        assert fake_tex.convert_bshd_to_thd.call_args[0][2] == 10

    def test_pad_and_unpad_forward_plain_int_lists(self, fake_tex):
        """The binding takes std::vector<size_t>; a tensor or a generator
        would not convert."""
        src, dst = torch.zeros(6, 4), torch.zeros(8, 4)
        te_extra_ops.pad_rows(src, dst, (2, 4), (4, 4))
        args = fake_tex.fused_multi_row_padding.call_args[0]
        assert args[2] == [2, 4] and args[3] == [4, 4]

        te_extra_ops.unpad_rows(dst, src, (4, 4), (2, 4))
        args = fake_tex.fused_multi_row_unpadding.call_args[0]
        assert args[2] == [4, 4] and args[3] == [2, 4]

    def test_swap_first_dims_forwards_the_out_argument(self, fake_tex):
        x = torch.zeros(2, 3, 4)
        out = torch.zeros(3, 2, 4)
        te_extra_ops.swap_first_dims(x, out)
        fake_tex.swap_first_dims.assert_called_once_with(x, out)


class TestCopyToKvCache:
    def test_rejects_an_unknown_qkv_format_before_importing_te_internals(self, fake_tex):
        """Checked ahead of the TE submodule import so a plain typo is
        reported as a typo, not as an ImportError from TE's attention code."""
        with pytest.raises(ValueError, match="bshd/sbhd/thd"):
            te_extra_ops.copy_to_kv_cache(
                torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1),
                torch.zeros(1), torch.zeros(1), torch.zeros(1),
                "bhsd", 1, 1, 1,
            )

    def test_translates_the_format_string_and_preserves_argument_order(
        self, fake_tex, monkeypatch
    ):
        """A thin passthrough on purpose -- the argument order is TE's, and
        reimplementing a cache manager on top would be a second source of
        truth for page-table layout."""
        qkv_format_enum = {"bshd": 0, "sbhd": 1, "thd": 2}
        fused_attn = types.ModuleType("transformer_engine.pytorch.cpp_extensions.fused_attn")
        fused_attn.QKVFormat = qkv_format_enum
        for name in (
            "transformer_engine",
            "transformer_engine.pytorch",
            "transformer_engine.pytorch.cpp_extensions",
        ):
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        monkeypatch.setitem(
            sys.modules, "transformer_engine.pytorch.cpp_extensions.fused_attn", fused_attn
        )

        new_k, new_v = torch.zeros(1, 2, 4, 8), torch.zeros(1, 2, 4, 8)
        k_cache, v_cache = torch.zeros(1, 16, 4, 8), torch.zeros(1, 16, 4, 8)
        page_table = torch.zeros(1, dtype=torch.int32)
        cu_new, cu_cached = torch.zeros(2, dtype=torch.int32), torch.zeros(2, dtype=torch.int32)

        te_extra_ops.copy_to_kv_cache(
            new_k, new_v, k_cache, v_cache, page_table, cu_new, cu_cached,
            "bshd", 1, 2, 16, max_pages_per_seq=1, is_non_paged=True,
        )

        args = fake_tex.copy_to_kv_cache.call_args[0]
        assert args[:5] == (new_k, new_v, k_cache, v_cache, page_table)
        assert args[7] == 0, "the format string must become TE's enum value"
        assert args[8:] == (1, 2, 16, 1, True)
