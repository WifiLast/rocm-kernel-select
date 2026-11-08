"""Analyse a MIOpen log captured by stable-diffusion-webui-amdgpu's
start.sh --debug: conv inventory, solver/algorithm distribution, and a
specific, evidence-based verdict on whether CONV FUSION is happening.

RELATIONSHIP TO tools/bench_conv_from_miopen_log.py. That script takes the
shapes out of a log and BENCHMARKS amd_tuned_torch's conv tiers on them,
so it needs torch, a GPU, and the compiled extension. This one only reads
the log: pure Python, no torch import, no device, runs in a second on a
16MB capture. Use this first to see what happened, that one to decide what
to do about it.

WHAT "DID FUSION WORK" ACTUALLY ASKS. MIOpen can fuse convolution with
bias and activation, but only through its FUSION API -- miopenCreateFusionPlan,
miopenCreateOpConvForward/OpBiasForward/OpActivationForward,
miopenCompileFusionPlan, miopenExecuteFusionPlan. The fused solvers that
MIOPEN_DEBUG_CONV_CK_IGEMM_FWD_BIAS_ACTIV and friends gate are reachable
ONLY from that path. A plain miopenConvolutionForward call cannot select
them, because the problem it describes has no bias or activation attached
to fuse.

So this script answers the question with four independent pieces of
evidence rather than one grep:

  1. Did any fusion-API call appear in the log?
  2. Was any fused solver named in the log?
  3. Does the PyTorch build even REFERENCE the fusion API? (--torch-lib)
     This is the decisive one: if libtorch never imports those symbols, no
     environment variable can make fusion happen, and a log showing no
     fusion is not evidence of a misconfiguration.
  4. Was the capture taken with the fusion knobs on at all? A log from
     `start.sh --debug` (the `conv` profile) never had them set, so
     "no fusion" from such a log says nothing either way.

BEWARE THE OBVIOUS GREP. "diffusion" contains "fusion". A case-insensitive
search for /fusion/ over a stable-diffusion log matches model filenames,
config paths and warning text, and will report dozens of hits on a run
with no fusion whatsoever. Every pattern here is anchored to MIOpen's own
log lines and uses word boundaries.

USAGE

    python tools/analyse_miopen_log.py LOG
    python tools/analyse_miopen_log.py LOG --top 20
    python tools/analyse_miopen_log.py LOG --fusion-only
    python tools/analyse_miopen_log.py LOG --json report.json
    python tools/analyse_miopen_log.py LOG \
        --torch-lib /mnt/data/stable-diffusion-webui-amdgpu/venv/lib/python3.10/site-packages/torch/lib/libtorch_hip.so
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

# ---------------------------------------------------------------------------
# Patterns. All anchored to MIOpen's own log lines ("MIOpen(HIP) ...") so
# webui chatter cannot contribute -- see the "diffusion" warning above.
# ---------------------------------------------------------------------------

_MIOPEN_LINE = re.compile(r"^MIOpen\(")

# The driver-equivalent command MIOPEN_ENABLE_LOGGING_CMD emits, e.g.
#   ./bin/MIOpenDriver convfp16 -n 2 -c 640 -H 128 -W 64 -k 640 -y 3 -x 3
#                      -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
# -p/-q pad, -u/-v stride, -l/-j dilation, -g groups, -F direction.
_CMD = re.compile(
    r"conv(?P<dt>fp16|bfp16|fp32|fp64)\s+-n (?P<n>\d+) -c (?P<c>\d+) "
    r"-H (?P<H>\d+) -W (?P<W>\d+) -k (?P<k>\d+) -y (?P<y>\d+) -x (?P<x>\d+) "
    r"-p (?P<p>\d+) -q (?P<q>\d+) -u (?P<u>\d+) -v (?P<v>\d+) "
    r"-l (?P<l>\d+) -j (?P<j>\d+).*?-g (?P<g>\d+) -F (?P<F>\d+)")

# Which command tag a driver line carries. LogCmdConvolution is a real
# convolution being executed; LogCmdFindConvolution is MIOpen PROBING for an
# algorithm and executes no user work -- counting the two together inflates
# both the call count and the FLOP total (72 of 2720 lines in the capture
# this was written against).
_CMD_TAG = re.compile(r"\[(LogCmdConvolution|LogCmdFindConvolution)\]")

_KERNEL = re.compile(r"kernel_name = ([A-Za-z0-9_]+)")
_ALGO = re.compile(r"\b(miopenConvolutionFwdAlgo[A-Za-z0-9]+|"
                   r"miopenConvolutionBwdDataAlgo[A-Za-z0-9]+|"
                   r"miopenConvolutionBwdWeightsAlgo[A-Za-z0-9]+)\b")
# The bracketed tag MIOpen puts on each line: "Info2 [ProcessPredictions] ..."
_TAG = re.compile(r"^MIOpen\([^)]*\)\s+[\d.]*:?\s*\w*\s*\[([A-Za-z0-9_]+)\]")
_ELAPSED = re.compile(r"^MIOpen\([^)]*\)\s+([\d.]+):")
_SOLVER_NAME = re.compile(r"\b(Conv[A-Za-z0-9_]{3,}|Gemm[A-Za-z0-9_]{3,})\b")
_TUNANET = re.compile(r"TunaNet Result: (.+)$")
_ARCH = re.compile(r"\bgfx[0-9a-f]+[a-z]*\b")
# MIOpen reports some of its own settings, e.g.
#   MIOpen(HIP)   0.005: Info [] MIOPEN_FIND_MODE = FAST(2)
# That is authoritative for what MIOpen actually used, unlike start.sh's
# echo of what the shell exported, so it wins where both are present.
_MIOPEN_SETTING = re.compile(r"Info \[\] ([A-Z_0-9]+) = (.+?)\s*$")

# MIOpen's fusion API, as it appears in a log line. Word-anchored.
_FUSION_API = re.compile(
    r"\bmiopen(CreateFusionPlan|DestroyFusionPlan|CompileFusionPlan|"
    r"ExecuteFusionPlan|FusionPlanGetWorkSpaceSize|FusionPlanConvolutionGetAlgo|"
    r"CreateOpConvForward|CreateOpBiasForward|CreateOpActivationForward|"
    r"CreateOpBatchNormInference|CreateOperatorArgs|SetOpArgs\w*)\b")

# Solvers that only the fusion path can select.
_FUSED_SOLVER = re.compile(
    r"\b(ConvBiasActiv\w*|ConvCKIgemmFwdBiasActiv\w*|ConvCKIgemmFwdBiasResAddActiv\w*|"
    r"ConvBinWinogradRxSFused\w*|ConvWinogradFused\w*|\w*FusedWinograd\w*|"
    r"BnFwdInferActivationFused\w*|BnFwdTrgActivationFused\w*|"
    r"BnBwdTrgActivationFused\w*)\b")

# The env knobs start.sh's `fusion` profile sets. Their presence in the log
# header (start.sh echoes the environment) says the capture was taken with
# fusion enabled.
_FUSION_KNOBS = (
    "MIOPEN_DEBUG_CONV_CK_IGEMM_FWD_BIAS_ACTIV",
    "MIOPEN_DEBUG_CONV_CK_IGEMM_FWD_BIAS_RES_ADD_ACTIV",
    "MIOPEN_DEBUG_CONV_CK_IGEMM_GRP_FWD_ACTIV",
    "MIOPEN_DEBUG_CONV_CK_IGEMM_GRP_FWD_BIAS_ACTIV",
    "MIOPEN_DEBUG_AMD_FUSED_WINOGRAD",
)

# Torch symbols that would have to be present for fusion to be reachable.
_TORCH_FUSION_SYMBOLS = (
    b"miopenCreateFusionPlan",
    b"miopenCompileFusionPlan",
    b"miopenExecuteFusionPlan",
    b"miopenCreateOpConvForward",
    b"miopenCreateOpBiasForward",
    b"miopenCreateOpActivationForward",
)

_DIRECTION = {"1": "forward", "2": "backward-data", "4": "backward-weights"}


class Conv:
    __slots__ = ("dt", "n", "c", "H", "W", "k", "y", "x", "p", "q", "u", "v",
                 "l", "j", "g", "F")

    def __init__(self, m):
        d = m.groupdict()
        self.dt = d["dt"]
        for f in ("n", "c", "H", "W", "k", "y", "x", "p", "q", "u", "v", "l", "j", "g", "F"):
            setattr(self, f, int(d[f]))

    def key(self):
        return (self.dt, self.n, self.c, self.H, self.W, self.k, self.y, self.x,
                self.p, self.q, self.u, self.v, self.l, self.j, self.g, self.F)

    def out_hw(self):
        oh = (self.H + 2 * self.p - self.l * (self.y - 1) - 1) // self.u + 1
        ow = (self.W + 2 * self.q - self.j * (self.x - 1) - 1) // self.v + 1
        return oh, ow

    def flops(self):
        oh, ow = self.out_hw()
        if oh <= 0 or ow <= 0:
            return 0
        return 2 * self.n * self.k * oh * ow * (self.c // max(self.g, 1)) * self.y * self.x

    def describe(self):
        oh, ow = self.out_hw()
        return (f"{self.dt} n{self.n} c{self.c} {self.H}x{self.W} -> "
                f"k{self.k} {oh}x{ow}  {self.y}x{self.x}"
                f"{' s%d' % self.u if self.u != 1 else ''}"
                f"{' d%d' % self.l if self.l != 1 else ''}"
                f"{' g%d' % self.g if self.g != 1 else ''}"
                f"  [{_DIRECTION.get(str(self.F), 'F=%d' % self.F)}]")


def parse(path):
    """One pass over the log. Returns a dict of everything the report needs."""
    convs = collections.Counter()          # executed convolutions only
    find_convs = collections.Counter()     # algorithm probes (no user work)
    conv_objs = {}
    miopen_settings = {}
    kernels = collections.Counter()
    algos = collections.Counter()
    tags = collections.Counter()
    solvers_mentioned = collections.Counter()
    fusion_api_hits = []
    fused_solver_hits = []
    tunanet_lists = collections.Counter()
    arches = collections.Counter()
    env_seen = {}
    miopen_lines = 0
    total_lines = 0
    # Attribute each kernel launch to the most recently logged conv command --
    # MIOpen emits the command first, then the launches it caused.
    current = None
    per_shape_kernels = collections.defaultdict(collections.Counter)

    with open(path, "r", errors="replace") as fh:
        for line in fh:
            total_lines += 1

            # start.sh echoes the environment it set, before python starts.
            if "=" in line and line.lstrip().startswith(("MIOPEN_", "AMD_TUNED_TORCH_")):
                k, _, v = line.strip().partition("=")
                env_seen.setdefault(k.strip(), v.strip())

            if not _MIOPEN_LINE.match(line):
                continue
            miopen_lines += 1

            mt = _TAG.match(line)
            if mt:
                tags[mt.group(1)] += 1

            ms_set = _MIOPEN_SETTING.search(line)
            if ms_set:
                miopen_settings[ms_set.group(1)] = ms_set.group(2)

            m = _CMD.search(line)
            if m:
                conv = Conv(m)
                key = conv.key()
                conv_objs.setdefault(key, conv)
                tag = _CMD_TAG.search(line)
                if tag and tag.group(1) == "LogCmdFindConvolution":
                    find_convs[key] += 1
                else:
                    convs[key] += 1
                    # Only an executed conv owns the kernel lines that follow;
                    # attributing them to a find probe would mis-credit solvers.
                    current = key

            mk = _KERNEL.search(line)
            if mk:
                kernels[mk.group(1)] += 1
                if current is not None:
                    per_shape_kernels[current][mk.group(1)] += 1

            for ma in _ALGO.finditer(line):
                algos[ma.group(1)] += 1

            for ms in _SOLVER_NAME.finditer(line):
                solvers_mentioned[ms.group(1)] += 1

            if _FUSION_API.search(line):
                fusion_api_hits.append(line.rstrip())
            if _FUSED_SOLVER.search(line):
                fused_solver_hits.append(line.rstrip())

            mtn = _TUNANET.search(line)
            if mtn:
                tunanet_lists[mtn.group(1).strip()[:200]] += 1

            for ma in _ARCH.finditer(line):
                arches[ma.group(0)] += 1

    return dict(
        convs=convs, find_convs=find_convs, miopen_settings=miopen_settings,
        conv_objs=conv_objs, kernels=kernels, algos=algos,
        tags=tags, solvers_mentioned=solvers_mentioned,
        fusion_api_hits=fusion_api_hits, fused_solver_hits=fused_solver_hits,
        tunanet_lists=tunanet_lists, arches=arches, env_seen=env_seen,
        miopen_lines=miopen_lines, total_lines=total_lines,
        per_shape_kernels=per_shape_kernels,
    )


def inspect_torch_lib(path):
    """Does this libtorch reference MIOpen's fusion API?

    Method: a byte search of the whole shared object for each symbol name.
    An undefined symbol that libtorch imports has its name in .dynstr, so
    the name IS present in the file when torch calls it and absent when it
    does not. That makes a plain byte scan a sound test here and avoids
    needing nm/objdump. It cannot distinguish an imported symbol from the
    same text appearing in, say, a debug string -- so a PRESENT result is
    weaker evidence than an ABSENT one, and the report says so."""
    if not path:
        return None
    if not os.path.exists(path):
        return {"path": path, "error": "not found"}
    try:
        with open(path, "rb") as fh:
            blob = fh.read()
    except OSError as exc:
        return {"path": path, "error": str(exc)}
    found = {sym.decode(): (sym in blob) for sym in _TORCH_FUSION_SYMBOLS}
    return {"path": path, "symbols": found, "any": any(found.values())}


def fusion_verdict(data, torch_info):
    """Four pieces of evidence -> one verdict. Returns (verdict, lines)."""
    api = len(data["fusion_api_hits"])
    fused = len(data["fused_solver_hits"])
    knobs_on = [k for k in _FUSION_KNOBS if data["env_seen"].get(k) == "1"]
    out = []

    out.append(f"1. Fusion-API calls in the log            : {api}")
    out.append(f"2. Fused solvers named in the log         : {fused}")
    if torch_info is None:
        out.append("3. PyTorch references the fusion API      : not checked "
                   "(pass --torch-lib to settle this)")
    elif "error" in torch_info:
        out.append(f"3. PyTorch references the fusion API      : could not check "
                   f"({torch_info['error']})")
    else:
        missing = [s for s, present in torch_info["symbols"].items() if not present]
        if torch_info["any"]:
            out.append("3. PyTorch references the fusion API      : YES (at least one symbol)")
        else:
            out.append("3. PyTorch references the fusion API      : NO -- none of "
                       f"{len(missing)} fusion symbols appear in")
            out.append(f"                                            {torch_info['path']}")
    if knobs_on:
        out.append(f"4. Capture had the fusion knobs enabled   : yes ({len(knobs_on)}/"
                   f"{len(_FUSION_KNOBS)} set)")
    else:
        out.append("4. Capture had the fusion knobs enabled   : NO -- this log was not "
                   "taken with start.sh --debug=fusion")

    torch_says_no = (torch_info is not None and "error" not in torch_info
                     and not torch_info["any"])

    if api > 0 or fused > 0:
        verdict = "FUSION IS HAPPENING"
    elif torch_says_no:
        verdict = "FUSION CANNOT HAPPEN -- PyTorch never calls MIOpen's fusion API"
    elif not knobs_on:
        verdict = "INCONCLUSIVE -- no fusion seen, but the capture never enabled it"
    else:
        verdict = "NO FUSION -- knobs were on and nothing fused"
    return verdict, out


def fusion_opportunities(data, top):
    """Conv shapes whose surrounding pattern suggests a fusable epilogue.

    This is a HEURISTIC over shapes, not a claim about the graph: a MIOpen
    log records convolutions, not the elementwise ops around them, so
    nothing here proves a bias or activation follows. What it does show is
    where fusion would pay off IF the framework offered it -- a 1x1 conv
    issued at the same input shape as a 3x3 is the residual-skip projection
    of a ResBlock, and in SD/SDXL that pair is always conv -> (bias) ->
    SiLU/GroupNorm -> conv, i.e. exactly the epilogue MIOpen's
    conv+bias+activation solvers exist for."""
    by_input = collections.defaultdict(list)
    for key, count in data["convs"].items():
        conv = data["conv_objs"][key]
        if conv.F != 1:
            continue
        by_input[(conv.dt, conv.n, conv.c, conv.H, conv.W)].append((conv, count))

    pairs = []
    for sig, entries in by_input.items():
        threes = [(c, n) for c, n in entries if c.y > 1]
        ones = [(c, n) for c, n in entries if c.y == 1 and c.x == 1]
        if threes and ones:
            calls = sum(n for _, n in threes) + sum(n for _, n in ones)
            fl = sum(c.flops() * n for c, n in threes) + sum(c.flops() * n for c, n in ones)
            pairs.append((calls, fl, sig, threes, ones))
    pairs.sort(reverse=True, key=lambda t: t[0])
    return pairs[:top]


def report(data, torch_info, args):
    convs = data["convs"]
    total_calls = sum(convs.values())
    arch = data["arches"].most_common(1)
    arch = arch[0][0] if arch else "unknown"

    print("=" * 78)
    print(f"MIOpen log: {args.log}")
    print(f"  {data['total_lines']:,} lines, {data['miopen_lines']:,} from MIOpen"
          f"   |   arch: {arch}")
    print("=" * 78)

    verdict, evidence = fusion_verdict(data, torch_info)
    print()
    print("CONV FUSION")
    print("-" * 78)
    for line in evidence:
        print("  " + line)
    print()
    print(f"  VERDICT: {verdict}")
    if data["fusion_api_hits"][:3]:
        print("  first fusion-API lines:")
        for line in data["fusion_api_hits"][:3]:
            print("    " + line[:150])

    if args.fusion_only:
        _print_opportunities(data, args)
        return

    print()
    print("CONVOLUTIONS")
    print("-" * 78)
    print(f"  {total_calls:,} EXECUTED calls over {len(convs)} distinct shapes")
    finds = sum(data["find_convs"].values())
    if finds:
        print(f"  {finds:,} algorithm-search probes (LogCmdFindConvolution) -- "
              "excluded from")
        print("    the counts and FLOPs below; they select a kernel, they do not "
              "do the work")
    fwd = sum(n for k, n in convs.items() if data["conv_objs"][k].F == 1)
    print(f"  forward: {fwd:,}   other directions: {total_calls - fwd:,}")
    total_flops = sum(data["conv_objs"][k].flops() * n for k, n in convs.items())
    print(f"  total conv FLOPs: {total_flops / 1e12:.2f} TFLOP")

    print()
    print("  Algorithms chosen:")
    algo_total = sum(data["algos"].values()) or 1
    for name, n in data["algos"].most_common(10):
        print(f"    {n:7,}  {100.0 * n / algo_total:5.1f}%  {name}")

    print()
    print("  Kernels actually launched:")
    for name, n in data["kernels"].most_common(10):
        print(f"    {n:7,}  {name}")
    launches = sum(data["kernels"].values())
    if launches and launches < total_calls:
        gemm = sum(n for a, n in data["algos"].items() if a.endswith("AlgoGEMM"))
        print(f"    ({launches:,} launches for {total_calls:,} convs -- convs on the")
        print(f"     GEMM algorithm ({gemm:,} here) are served by rocBLAS via")
        print("     CallGemm*, which emits no MIOpen kernel_name line.)")

    print()
    print(f"TOP {args.top} CONV SHAPES BY CALL COUNT")
    print("-" * 78)
    for key, n in convs.most_common(args.top):
        conv = data["conv_objs"][key]
        ks = data["per_shape_kernels"].get(key)
        kname = ks.most_common(1)[0][0] if ks else "(no kernel line)"
        print(f"  {n:5,}x  {conv.describe()}")
        print(f"          {conv.flops() * n / 1e12:7.3f} TFLOP total   {kname}")

    _print_opportunities(data, args)

    print()
    print("NOTES")
    print("-" * 78)
    xdlops = sum(n for s, n in data["solvers_mentioned"].items() if "Xdlops" in s)
    if xdlops and arch.startswith("gfx11"):
        print(f"  * {xdlops:,} mentions of *Xdlops* solvers on {arch}. RDNA3 has no")
        print("    XDLOPS (MFMA) units -- these are TunaNet predicting CDNA solvers")
        print("    that this card cannot run, and they get discarded. Not an error,")
        print("    but it means the heuristic's top pick is regularly unusable and")
        print("    the fallback is what really chose the kernel.")
    if data["tags"].get("GetSolutionsFallback"):
        print(f"  * {data['tags']['GetSolutionsFallback']:,} GetSolutionsFallback calls "
              "-- immediate-mode fallback,")
        print("    i.e. no find-db entry for the problem. Consider MIOPEN_FIND_ENFORCE=3")
        print("    (DB_UPDATE) on a warm-up run to populate it.")
    find_mode = (data["miopen_settings"].get("MIOPEN_FIND_MODE")
                 or data["env_seen"].get("MIOPEN_FIND_MODE"))
    if find_mode and not find_mode.startswith(("NORMAL", "1")):
        print(f"  * MIOPEN_FIND_MODE = {find_mode} during capture -- not a full search.")
        print("    MIOpen reuses find-db/heuristic answers instead of evaluating every")
        print("    solver, so the solver set in this log is narrower than the hardware")
        print("    supports. Re-capture with MIOPEN_FIND_MODE=1 to see the real set.")
    for name, value in sorted(data["miopen_settings"].items()):
        if name != "MIOPEN_FIND_MODE":
            print(f"  * MIOpen reported {name} = {value}")


def _print_opportunities(data, args):
    pairs = fusion_opportunities(data, args.top)
    if not pairs:
        return
    print()
    print("WHERE FUSION WOULD PAY (heuristic -- see fusion_opportunities())")
    print("-" * 78)
    print("  ResBlock-shaped sites: a 1x1 projection issued at the same input")
    print("  shape as a 3x3, which in SD/SDXL always carries a bias+activation")
    print("  epilogue between them.")
    print()
    for calls, fl, sig, threes, ones in pairs:
        dt, n, c, H, W = sig
        print(f"  {calls:5,} calls  {fl / 1e12:7.3f} TFLOP   {dt} n{n} c{c} {H}x{W}")
        for conv, cnt in sorted(threes + ones, key=lambda t: -t[1]):
            print(f"           {cnt:5,}x  {conv.y}x{conv.x} -> k{conv.k}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("log")
    ap.add_argument("--top", type=int, default=12,
                    help="how many shapes to list (default 12)")
    ap.add_argument("--fusion-only", action="store_true",
                    help="print only the fusion verdict and opportunities")
    ap.add_argument("--json", metavar="PATH", help="also write the report as JSON")
    ap.add_argument("--torch-lib", metavar="PATH",
                    default=os.environ.get("ANALYSE_MIOPEN_TORCH_LIB"),
                    help="libtorch_hip.so to check for fusion-API references; "
                         "this is what makes the fusion verdict decisive")
    args = ap.parse_args(argv)

    if not os.path.exists(args.log):
        ap.error(f"no such log: {args.log}")

    # Default to the webui's own libtorch when the log came from there and
    # the caller did not name one -- that is the build that produced the log.
    if not args.torch_lib:
        guess = os.path.join(
            os.path.dirname(os.path.abspath(args.log)), "..", "..",
            "venv", "lib", "python3.10", "site-packages", "torch", "lib",
            "libtorch_hip.so")
        guess = os.path.normpath(guess)
        if os.path.exists(guess):
            args.torch_lib = guess

    data = parse(args.log)
    torch_info = inspect_torch_lib(args.torch_lib)
    report(data, torch_info, args)

    if args.json:
        verdict, evidence = fusion_verdict(data, torch_info)
        out = {
            "log": os.path.abspath(args.log),
            "arch": (data["arches"].most_common(1) or [("unknown", 0)])[0][0],
            "fusion": {
                "verdict": verdict,
                "evidence": evidence,
                "api_calls": len(data["fusion_api_hits"]),
                "fused_solvers": len(data["fused_solver_hits"]),
                "torch_lib": torch_info,
            },
            "conv_calls": sum(data["convs"].values()),
            "find_probes": sum(data["find_convs"].values()),
            "miopen_settings": data["miopen_settings"],
            "distinct_shapes": len(data["convs"]),
            "algorithms": dict(data["algos"]),
            "kernels": dict(data["kernels"]),
            "shapes": [
                {
                    "count": n,
                    "dtype": data["conv_objs"][k].dt,
                    "n": data["conv_objs"][k].n, "c": data["conv_objs"][k].c,
                    "H": data["conv_objs"][k].H, "W": data["conv_objs"][k].W,
                    "k": data["conv_objs"][k].k,
                    "y": data["conv_objs"][k].y, "x": data["conv_objs"][k].x,
                    "stride": [data["conv_objs"][k].u, data["conv_objs"][k].v],
                    "pad": [data["conv_objs"][k].p, data["conv_objs"][k].q],
                    "dilation": [data["conv_objs"][k].l, data["conv_objs"][k].j],
                    "groups": data["conv_objs"][k].g,
                    "direction": _DIRECTION.get(str(data["conv_objs"][k].F)),
                    "tflop_total": data["conv_objs"][k].flops() * n / 1e12,
                    "kernel": (data["per_shape_kernels"][k].most_common(1)[0][0]
                               if data["per_shape_kernels"].get(k) else None),
                }
                for k, n in data["convs"].most_common()
            ],
        }
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
