"""CPU tests for einsum_ops (the einsum -> permute/reshape + bmm planner) and
the _dispatch._patched_einsum wrapper's routing. The planner is pure PyTorch,
so its numbers are checked against stock torch.einsum here on CPU; the GPU
side (patched torch.bmm underneath, kernel_select contest) is
tests_hardware/test_einsum_ops.py."""
from __future__ import annotations

import pytest
import torch

import amd_tuned_torch
from amd_tuned_torch import einsum_ops, kernel_select

from conftest import force_eligible

STOCK = einsum_ops._STOCK_EINSUM   # torch.einsum may already be patched

CASES = [
    ("bij,bjk->bik", [(3, 4, 5), (3, 5, 6)]),
    ("bhqd,bhkd->bhqk", [(2, 3, 7, 8), (2, 3, 9, 8)]),
    ("bhqk,bhkd->bhqd", [(2, 3, 7, 9), (2, 3, 9, 8)]),
    ("bnc,cd->bnd", [(2, 6, 5), (5, 4)]),
    ("ij,jk->ik", [(4, 5), (5, 6)]),
    ("ij,jk", [(4, 5), (5, 6)]),                      # implicit output
    ("i,i->", [(7,), (7,)]),                          # dot product
    ("i,j->ij", [(3,), (4,)]),                        # outer product
    ("ij->ji", [(3, 4)]),                             # single operand transpose
    ("ijk->i", [(3, 4, 5)]),                          # single operand reduce
    ("bij,bjk,bkl->bil", [(2, 3, 4), (2, 4, 5), (2, 5, 6)]),   # three operands
    ("abc,cd->abd", [(1, 4, 5), (5, 3)]),
    ("bij,bjk->bik", [(1, 4, 5), (3, 5, 6)]),         # size-1 broadcast batch
    ("ijk,ik->ij", [(3, 4, 5), (3, 5)]),              # batch dim not leading
    ("ij,kj->ikj", [(3, 4), (5, 4)]),                 # kept index, no contraction
    ("ab,bc->", [(3, 4), (4, 5)]),                    # full reduction
]


@pytest.mark.parametrize("eq, shapes", CASES)
def test_planner_matches_stock(eq, shapes):
    torch.manual_seed(0)
    ops = [torch.randn(s, dtype=torch.float64) for s in shapes]
    assert einsum_ops.supported(eq, ops) is None
    got = einsum_ops.einsum(eq, ops, bmm=torch.bmm)
    ref = STOCK(eq, *ops)
    assert got.shape == ref.shape
    torch.testing.assert_close(got, ref)


@pytest.mark.parametrize("eq, shapes, reason", [
    ("...ij,...jk->...ik", [(2, 3, 4), (2, 4, 5)], "ellipsis"),
    ("ii->i", [(3, 3)], "repeated index in one operand"),
    ("ij,jk->iz", [(3, 4), (4, 5)], "bad output"),
    ("ij,jk->ik", [(3, 4)], "operand count"),
    ("ijk,jk->ik", [(3, 4), (4, 5)], "rank mismatch"),
    ("ij,jk->ik", [(3, 4), (5, 6)], "size mismatch"),
])
def test_unsupported_reasons(eq, shapes, reason):
    ops = [torch.randn(s) for s in shapes]
    assert einsum_ops.supported(eq, ops) == reason


def test_reference_is_stock_einsum():
    a, b = torch.randn(3, 4), torch.randn(4, 5)
    torch.testing.assert_close(einsum_ops.reference("ij,jk->ik", a, b), a @ b)
    assert einsum_ops.available() is True


def test_planner_uses_the_bmm_it_is_given():
    calls = []

    def bmm(a, b):
        calls.append((tuple(a.shape), tuple(b.shape)))
        return torch.bmm(a, b)
    einsum_ops.einsum("bhqd,bhkd->bhqk", [torch.randn(2, 3, 7, 8), torch.randn(2, 3, 9, 8)], bmm=bmm)
    assert calls == [((6, 7, 8), (6, 8, 9))]


# ---------------------------------------------------------------------------
# _patched_einsum routing (CPU tensors, eligibility forced)

@pytest.fixture
def wrapper(monkeypatch):
    """_patched_einsum with stock einsum registered as the original and a
    recording pick_key/cached_key in place of a real GPU contest."""
    stock_calls = []

    def stock(*args, **kwargs):
        stock_calls.append(args[0] if args else None)
        return STOCK(*args, **kwargs)
    monkeypatch.setitem(amd_tuned_torch._dispatch._ORIGINALS, (torch, "einsum"), stock)
    monkeypatch.setattr(kernel_select, "_ENABLED", True)
    amd_tuned_torch._dispatch.einsum_memo_clear()
    cache: dict = {}
    contests = []

    def pick_key(kind, key, candidates, tolerance=None):
        if kind == "einsum":     # torch.bmm may be patched too (other tests enable()); its contests aren't ours
            contests.append((kind, key, [n for n, _ in candidates]))
        cache[(kind,) + tuple(key)] = candidates[0][0]
        return candidates[0][1]()          # pretend the rewrite won
    monkeypatch.setattr(kernel_select, "pick_key", pick_key)
    monkeypatch.setattr(kernel_select, "cached_key", lambda kind, key: cache.get((kind,) + tuple(key)))
    return type("W", (), {"fn": staticmethod(amd_tuned_torch._dispatch._patched_einsum),
                          "stock_calls": stock_calls, "contests": contests})


def test_wrapper_contests_rewrite_against_stock(monkeypatch, wrapper):
    force_eligible(monkeypatch)
    a, b = torch.randn(3, 4, 5), torch.randn(3, 5, 6)
    out = wrapper.fn("bij,bjk->bik", a, b)
    torch.testing.assert_close(out, STOCK("bij,bjk->bik", a, b))
    assert wrapper.contests == [("einsum", (torch.float32, "bij,bjk->bik", ((3, 4, 5), (3, 5, 6))),
                                 ["bmm_plan", "stock"])]
    # second call: cached winner, no new contest (same key, other spacing)
    wrapper.fn("bij, bjk -> bik", a, b)
    assert len(wrapper.contests) == 1


def test_memo_still_checks_grad(monkeypatch, wrapper):
    force_eligible(monkeypatch)
    a, b = torch.randn(4, 5), torch.randn(5, 6)
    wrapper.fn("ij,jk->ik", a, b)                       # decided: bmm_plan
    monkeypatch.setattr(amd_tuned_torch._dispatch, "_grad_safe", lambda *x, **k: False)
    monkeypatch.setattr(einsum_ops, "einsum", lambda *x, **k: pytest.fail("rewrite under autograd"))
    wrapper.fn("ij,jk->ik", a, b)


def test_wrapper_accepts_operand_list(monkeypatch, wrapper):
    force_eligible(monkeypatch)
    a, b = torch.randn(4, 5), torch.randn(5, 6)
    torch.testing.assert_close(wrapper.fn("ij,jk->ik", [a, b]), a @ b)
    assert len(wrapper.contests) == 1


def test_wrapper_stock_for_unsupported_equation(monkeypatch, wrapper):
    force_eligible(monkeypatch)
    x = torch.randn(3, 3)
    torch.testing.assert_close(wrapper.fn("ii->i", x), torch.diagonal(x))
    assert wrapper.contests == [] and wrapper.stock_calls == ["ii->i"]


def test_wrapper_stock_for_sublist_format(monkeypatch, wrapper):
    force_eligible(monkeypatch)
    a, b = torch.randn(4, 5), torch.randn(5, 6)
    torch.testing.assert_close(wrapper.fn(a, [0, 1], b, [1, 2], [0, 2]), a @ b)
    assert wrapper.contests == []


def test_wrapper_stock_for_mixed_dtypes(monkeypatch, wrapper):
    force_eligible(monkeypatch)
    with pytest.raises(RuntimeError):       # stock einsum's own error, not ours
        wrapper.fn("ij,jk->ik", torch.randn(4, 5), torch.randn(5, 6, dtype=torch.float64))
    assert wrapper.contests == [] and len(wrapper.stock_calls) == 1


def test_wrapper_stock_for_cpu_tensors(wrapper):
    # no force_eligible: real _usable rejects CPU tensors
    wrapper.fn("ij,jk->ik", torch.randn(4, 5), torch.randn(5, 6))
    assert wrapper.contests == [] and len(wrapper.stock_calls) == 1


def test_wrapper_stock_when_kernel_select_off(monkeypatch, wrapper):
    force_eligible(monkeypatch)
    monkeypatch.setattr(kernel_select, "_ENABLED", False)
    wrapper.fn("ij,jk->ik", torch.randn(4, 5), torch.randn(5, 6))
    assert wrapper.contests == []


def test_wrapper_stock_when_grad_needed(monkeypatch, wrapper):
    force_eligible(monkeypatch)
    monkeypatch.setattr(amd_tuned_torch._dispatch, "_grad_safe", lambda *a, **k: False)
    a = torch.randn(4, 5, requires_grad=True)
    wrapper.fn("ij,jk->ik", a, torch.randn(5, 6)).sum().backward()
    assert wrapper.contests == [] and a.grad is not None


def test_cached_stock_winner_skips_rewrite(monkeypatch, wrapper):
    force_eligible(monkeypatch)
    monkeypatch.setattr(kernel_select, "cached_key", lambda kind, key: "stock")
    monkeypatch.setattr(einsum_ops, "einsum", lambda *a, **k: pytest.fail("rewrite must not run"))
    wrapper.fn("ij,jk->ik", torch.randn(4, 5), torch.randn(5, 6))
    assert len(wrapper.stock_calls) == 1
