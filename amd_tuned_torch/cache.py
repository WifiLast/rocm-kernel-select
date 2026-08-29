"""amd_tuned_torch.cache -- a generic similarity-gated memoization cache, extending
the mechanism behind amd_tuned_torch.teacache (itself a port of TeaCache,
arXiv:2411.19108) beyond diffusion transformers to any callable whose
primary input tends to change smoothly across consecutive calls.

Unlike amd_tuned_torch.magcache (needs a precomputed, schedule-specific per-step
table -- meaningless outside a fixed denoising schedule) and
amd_tuned_torch.teacache (ties skip/reuse to a diffusion-specific residual, plus
forced full recompute on the first/last of a fixed step count), this drops
every diffusion-specific assumption: no step counter, no calibration table,
no residual structure required. Just: if this call's primary argument is
close enough (relative L1 distance) to the previous call's, skip calling
the real function and reuse its last output.

THIS IS NOT SAFE TO APPLY EVERYWHERE. It's only correct where consecutive
calls are expected to have similar inputs *and* it's acceptable for the
output to lag slightly behind an exact recomputation when that holds --
diffusion timesteps, video frames, iterative refinement loops. It is NOT
safe for e.g. independent batch items, unrelated calls that happen to
share a call site, or anything where "similar input" doesn't imply
"reusing the last output is an acceptable approximation." That's exactly
why none of this is auto-installed by amd_tuned_torch's own enable()/disable() --
it's gated by its own environment flag on top of that, AMD_TUNED_TORCH_ENABLE_
SIMILARITY_CACHE, which unlike every other flag in this project controls
something genuinely risky by default when set. See "Global module-level
caching" further down before turning it on.

Usage (explicit, at a call site you've judged appropriate -- always safe
regardless of the environment flag, since you control exactly what gets
wrapped):

    import amd_tuned_torch

    @amd_tuned_torch.cache.similarity_cached(thresh=0.1)
    def run_expensive_step(hidden_states, *aux_args):
        return transformer_blocks(hidden_states, *aux_args)

    # or, for explicit control instead of a decorator:
    cache = amd_tuned_torch.cache.SimilarityCache(thresh=0.1)
    output = cache.call(run_expensive_step, hidden_states, *aux_args)

Controlled by AMD_TUNED_TORCH_ENABLE_SIMILARITY_CACHE (default off, "0"): every
SimilarityCache constructed without an explicit `enabled=` argument reads
this once, at construction time. Left unset (or "0"), a SimilarityCache is
a pure passthrough -- fn is always called, nothing is ever skipped, and
the wrapped call is behaviorally identical to calling fn directly, at the
extra (cheap) cost of an L1-distance check per call. Same opt-in-only
posture as AMD_TUNED_TORCH_ENABLE_TE.

Caching returns the *same* output object across every cache hit until the
next real call, not a fresh copy -- do not mutate a returned value in
place if you're relying on the cache (same caveat amd_tuned_torch.magcache/
teacache's cached_residual already carries; this isn't a new risk caching
introduces beyond the ordinary one of any tensor returned from any
function, except now that tensor may be handed out more than once).

Not every reuse pattern is "close enough to the previous call" -- see
KeyedCache further down for the other one: exact, identity-keyed
memoization ("has this *exact* logical query already been computed"),
correct for workloads where the right answer depends on precise input
identity rather than tolerating drift, and wrong to force into
SimilarityCache by tightening thresh (a coarse aggregate-similarity
metric can still coincidentally match unrelated inputs no matter how
tight the tolerance -- tightening thresh shrinks that risk, it can't
eliminate it, because the failure is in what's being measured, not the
threshold on it).

Pairing with amd_tuned_torch.torch_compile
-----------------------------------
`compile=True` makes a SimilarityCache torch.compile the *real* calls, not
the skipped ones -- the two optimizations stack rather than compete:
skipping cuts how often `fn` runs at all, compiling cuts the cost of the
runs that still happen. This pairing is more than "why not": a hit
requires `x.shape == self.previous_input.shape` (see _is_close_enough), so
the population of calls that ever reach `fn` is already naturally
clustered around a small number of distinct shapes -- exactly what
torch_compile's default dynamic=False (recompile-per-exact-shape, no
shape-generic fallback kernel) wants to see.

    @amd_tuned_torch.cache.similarity_cached(thresh=0.1, compile=True)
    def run_expensive_step(hidden_states, *aux_args):
        return transformer_blocks(hidden_states, *aux_args)

`fn` is compiled at most once per SimilarityCache instance, on its first
real (cache-miss) call -- not once per call -- and that compiled callable
is reused for every later miss regardless of which `fn` *object* is passed
to future .call() invocations, as long as it's behaviorally the same
function. That matters for the module-level auto-cache below: each call to
_cached_module_call builds a fresh `_forward` closure, but every one of
those closures for a given module instance does the same thing (invoke
that module's real forward), so reusing the first-compiled version for it
is correct, not stale. See SimilarityCache.compile/compile_kwargs and
AMD_TUNED_TORCH_SIMILARITY_CACHE_COMPILE below for the module-level toggle -- and
its caveat: compiling every module process-wide, one at a time, is usually
not what you want (see that section).

Compiling is a standing optimization, not stale cached data -- unlike
previous_input/previous_output, `reset()` deliberately does NOT clear the
compiled callable. There's nothing to invalidate between unrelated runs;
the whole point is to pay the compile cost once and amortize it across
every run for that cache's lifetime.

Global module-level caching
----------------------------
When AMD_TUNED_TORCH_ENABLE_SIMILARITY_CACHE is set (to anything but "0"/""/
"false"/"False"), importing amd_tuned_torch ALSO patches torch.nn.Module.__call__
process-wide: every nn.Module instance gets its own private SimilarityCache
(keyed by that instance's identity, so different layers/modules never
compare against each other's inputs), applied automatically to its first
positional call argument. No code changes needed -- this is the literal
"use it in torch by default" behavior the flag name promises.

This is meaningfully riskier than the explicit decorator usage above,
because amd_tuned_torch has no way to know whether a given module is actually
called in a "smoothly drifting consecutive inputs" pattern (safe -- e.g. a
top-level diffusion transformer invoked once per external sampling-loop
timestep) or called many times per single forward pass with logically
*unrelated* inputs each time (unsafe -- e.g. a block called once per
token, once per expert, once per recurrent step -- see the nested-call
bypass below, which now catches most of this shape automatically -- or
shared/reused weights called from multiple *unrelated top-level call
sites*, which nesting alone can't detect since neither call is nested
inside the other). In the unsafe case, this can silently reuse a
completely wrong previous output instead of merely approximating -- not a
quality/speed tradeoff, a correctness bug. Safety valves:

  - Grad-safety: a module is only ever cached while _grad_safe() would
    also gate amd_tuned_torch's kernel-swap ops (no_grad/inference_mode, or none
    of its call args require_grad) -- reusing a stale output tensor from a
    different forward pass would otherwise corrupt autograd. Training runs
    are effectively exempt regardless of this flag.
  - Nested-call bypass: while executing one module's real forward, any
    nn.Module.__call__ reached from inside it is never independently
    wrapped or checked -- only the outermost module actually invoked from
    outside amd_tuned_torch's own machinery ever gets a SimilarityCache. See
    AMD_TUNED_TORCH_SIMILARITY_CACHE_SKIP_NESTED below.
  - Opt-out: set `module._amd_tuned_torch_no_cache = True` on any specific module
    instance (or its class) to exclude it even while the global flag is on.
  - `amd_tuned_torch.cache.disable_module_cache()` restores stock
    torch.nn.Module.__call__ and forgets every per-module cache.

AMD_TUNED_TORCH_SIMILARITY_CACHE_THRESH (default "0.1") sets the relative-L1
threshold used for every auto-created per-module cache; pass
`thresh=` to `enable_module_cache()` directly to override it for a call
you make yourself instead of relying on the import-time auto-install.

AMD_TUNED_TORCH_SIMILARITY_CACHE_MAX_BYTES (default "67108864", i.e. 64 MiB) caps
how large a module's primary-argument tensor can be before that module's
auto-created cache stops caching it (see SimilarityCache's max_bytes for
the exact bypass semantics). Set to "0" (or any value <= 0) for no limit.
Unlike AMD_TUNED_TORCH_SIMILARITY_CACHE_THRESH, this one defaults to something
other than "off" *for module-level caching specifically*, because that's
the case where a cache is auto-created for every nn.Module in a process
and kept alive for the module's whole lifetime -- large image/activation
tensors (a VAE decoder's feature maps, say) sitting in that cache forever
is a standing VRAM cost, not just a bounded working set, and hits exactly
the kind of module (called once per unrelated request, not smoothly
drifting per-timestep input) this flag's own safety writeup above already
flags as the unsafe case. SimilarityCache itself (explicit decorator/
direct-construction usage) still defaults max_bytes to None (no limit) --
you're opting in there and presumably already judged the workload safe.

AMD_TUNED_TORCH_SIMILARITY_CACHE_MIN_BYTES (default "4096", 4 KiB) is max_bytes's
mirror image: caps how *small* a module's primary-argument tensor can be
before that module's auto-created cache stops caching it (see
SimilarityCache's min_bytes for the exact bypass semantics -- same
passthrough as max_bytes, just triggered from the other direction). This
one defaults to a real floor for module-level caching too, for a different
reason than max_bytes's memory-safety story: below a few KiB, the fixed
per-call overhead -- _is_close_enough's elementwise subtract + two
reductions + a .item() sync, and (with compile=True) torch.compile's own
guard/dispatch overhead -- stops being negligible next to just calling the
tiny op itself, so caching genuinely small tensors (a timestep embedding
lookup, a small gating scalar, a class-token add) can be a net slowdown
rather than the speedup this module exists to provide. 4 KiB is
conservative: comfortably below a typical hidden-state/activation tensor
(e.g. a single-token 4096-dim fp16 hidden state is already 8 KiB) so it
shouldn't disable caching for anything that would actually benefit from
it, while still excluding the clearly-too-small cases above. Set to "0"
(or any value <= 0) for no floor; SimilarityCache's own min_bytes= (explicit
decorator/direct-construction usage) still defaults to None -- no floor --
since you're opting into that usage yourself and can judge it directly.

AMD_TUNED_TORCH_SIMILARITY_CACHE_MAX_MISSES (default "5") is a circuit breaker:
once a module's auto-created cache sees this many misses in a row, it
auto-disables itself -- every later call on that module skips
_is_close_enough entirely (see SimilarityCache's max_consecutive_misses
for the exact mechanics) until reset(). Defaults to a real limit here for
a case min/max_bytes above can't catch: a module invoked repeatedly with
inputs that are simply never similar to each other regardless of size --
e.g. a geometry decoder called once per spatial chunk while decoding a 3D
volume, where every chunk queries a different, unrelated region of space.
Measured in practice on exactly that workload (Hunyuan3D-2's volume
decoding): every one of the decoder's ~16 internal submodule calls paid
_is_close_enough's mandatory .item() sync -- a GPU->CPU round trip that
stalls the async kernel-launch pipeline -- on every one of ~2000+ chunks,
for a skip that essentially never happened, roughly halving throughput.
Set to "0" (or any value <= 0) to never auto-disable; SimilarityCache's
own max_consecutive_misses= (explicit decorator/direct-construction usage)
still defaults to None -- never auto-disable -- since you're opting into
that usage yourself.

AMD_TUNED_TORCH_SIMILARITY_CACHE_MIN_HIT_RATE (default "0.2") is a second,
independent circuit breaker -- rate-based instead of streak-based -- for a
gap MAX_MISSES leaves open: a "never similar" workload measured by strict
consecutive misses can be defeated by even rare *coincidental* hits, and
for volume decoding specifically that's exactly what happens. Query chunks
come from a dense, regularly-spaced 3D grid (see generate_dense_grid_points
in hy3dgen's volume_decoders.py) -- consecutive chunks are spatially
*adjacent* regions, not independently-random batches, and
_is_close_enough's similarity signal is a single coarse number (mean
absolute difference over the *entire* tensor, relative to its mean
magnitude) -- for a large batch of quasi-uniformly-distributed coordinates,
that aggregate statistic can land within thresh of the previous chunk's by
sheer coincidence even though every individual query point differs. Each
such coincidence both (a) resets max_consecutive_misses's streak counter
back to 0, indefinitely postponing that breaker, and (b) is itself a
correctness bug -- silently returning a *different* chunk's decoded output
for this chunk's actual query points, not merely a stale approximation.

This breaker sidesteps (a) by tracking hit/miss outcomes over a rolling
*time* window (see SimilarityCache.min_hit_rate/hit_rate_window_seconds/
hit_rate_min_samples) instead of requiring an unbroken run -- a handful of
scattered coincidental hits can no longer indefinitely block a trip the
way they defeat a strict streak, because the trip decision is "what
fraction of calls in the last hit_rate_window_seconds actually hit", not
"how many misses in an unbroken row". It doesn't fix (b) -- a low-enough
hit rate to trip this breaker still means every hit along the way served
wrong output for that call -- it only bounds how much wall-clock time gets
wasted (both on wrong answers and on the sync cost of finding them) before
giving up. For a module where even occasional hits are unacceptable
regardless of rate, the reliable fix is still the per-module opt-out
(`module._amd_tuned_torch_no_cache = True`), not either circuit breaker -- see
"Global module-level caching" above.

Deliberately NOT a per-run reset the way previous_input/max_consecutive_
misses are: this breaker's window is wall-clock time, continuous across
whatever reset() calls happen to land inside it, because the property it's
measuring ("is checking still paying for itself, recently") doesn't reset
just because a new generation/video/run started -- reset() does still
clear it (see SimilarityCache.reset), since starting a genuinely different
workload on the same cache instance is exactly when stale rate history
should be discarded, same reasoning as previous_input.

Set AMD_TUNED_TORCH_SIMILARITY_CACHE_MIN_HIT_RATE to "0" (or any value <= 0) to
disable this breaker; SimilarityCache's own min_hit_rate= (explicit
decorator/direct-construction usage) still defaults to None -- disabled --
same asymmetry as every other module-cache-only default above.
hit_rate_window_seconds (default 2.0) and hit_rate_min_samples (default
20) aren't exposed as their own environment variables -- pass them to
enable_module_cache() directly if the default window/sample-size doesn't
suit your call frequency.

AMD_TUNED_TORCH_SIMILARITY_CACHE_AUTO_EXCLUDE (default off, "0"): once a
per-module cache's circuit breaker trips (either max_consecutive_misses or
min_hit_rate) and it has no compile=True speedup worth preserving,
permanently exclude that module from the wrapper entirely -- setting its
`_amd_tuned_torch_no_cache = True`, the same attribute the manual opt-out above
uses -- instead of continuing to route every future call through
SimilarityCache.call() just to hit its already-cheap _auto_disabled
fast-path. Real savings: even the fast-path still pays a weakref dict
lookup, the grad-safety check over call args, and building a fresh
`_forward` closure, every call, forever, for a module that's proven it
never benefits. Measured on the same volume-decoding workload the other
breakers were tuned against: once both breakers were tripping quickly
(within the first ~12 of 2122 chunks), throughput barely improved further
without this -- the remaining gap wasn't the distance check anymore, it
was this fixed per-call dispatch cost, paid ~34,000 times (~16 submodules
x ~2100 remaining chunks) for calls that were already going to be
passed straight through.

Off by default because it's a real semantic change, not just an
optimization: reset() only clears a SimilarityCache's own internal state
(previous_input, streak, rate history) -- it has no reference to the
module instance it's attached to, so it can't un-set an attribute on it.
A module auto-excluded this way stays excluded for the rest of the
process, across every reset() call, and even across disable_module_cache()
+ re-enable (the attribute lives on the module object, not in this
module's bookkeeping) -- clear `module._amd_tuned_torch_no_cache` yourself if you
need a module to become eligible again. Fine -- often exactly what you
want -- for a workload you've already confirmed (via debug_summary()/
trip_reason) never benefits from caching regardless of run boundaries;
wrong for one where different runs might genuinely call for different
skip decisions. Set to "1" to enable; SimilarityCache's explicit decorator/
direct-construction API has no equivalent (there's no "module" for a
free-standing cache to exclude -- its own _auto_disabled fast-path is
already the cheapest available skip for that usage).

AMD_TUNED_TORCH_SIMILARITY_CACHE_CLASS_COOLDOWN (default unset, i.e. disabled), a
number of seconds: when any one module instance's circuit breaker trips,
put every OTHER instance of that same class on a temporary cooldown too --
for that many seconds, any instance of the class skips straight through
with no distance check and no per-instance bookkeeping at all, whether or
not it has its own SimilarityCache yet. Complements AUTO_EXCLUDE rather
than replacing it: AUTO_EXCLUDE waits for one specific instance to prove
itself, permanently; this generalizes that instance's trip to its whole
class immediately, temporarily. Useful for a deep stack of many instances
of the same normalization/projection class (e.g. every RMSNorm/LayerNorm
in a decoder), all seeing their own slice of the same never-similar
per-chunk activations -- whichever instance happens to trip first (by
chance the least fortunate in dodging coincidental hits) puts the rest on
cooldown immediately, instead of each independently re-discovering the
same conclusion over its own next several calls. Self-correcting if the
trip turns out not to generalize: the cooldown simply expires and
per-instance tracking resumes for the other instances, which were never
reset during the cooldown -- just not advanced, since they were skipped
entirely. No correctness cost beyond what any breaker already carries:
cooldown'd calls still run `fn` for real every time, they just skip
checking first -- this only ever widens which calls skip the *check*,
never serves a stale *cached output* to a call that wouldn't otherwise
have gotten one. Set to a positive number of seconds (e.g. "20") to
enable; SimilarityCache's explicit decorator/direct-construction API has
no equivalent (there's no "class" of free-standing caches to generalize
across -- each is independent by construction).

AMD_TUNED_TORCH_SIMILARITY_CACHE_SKIP_NESTED (default ON, "1"): while the
module-level auto-cache is executing one module's real forward (a miss,
or a breaker/size bypass -- anything that actually calls through), any
nn.Module.__call__ reached from inside that forward is passed straight to
stock __call__, never independently wrapped in its own SimilarityCache.
Only the outermost module actually invoked from outside amd_tuned_torch's own
machinery -- i.e. never itself reached while another wrapped call's real
forward is on the stack -- gets checked/cached at all. Tracked with a
threading.local() call-depth counter (not a plain module global) so
depth never leaks between threads in a server handling concurrent
requests.

Exists because, unlike every other knob in this section, this defaults to
ON: nesting is usually pure waste, not a tradeoff. A block buried inside
an already-wrapped top-level forward -- one Linear/LayerNorm/attention
sub-block among many in a transformer stack -- almost never sees inputs as
similar as its parent's, because each layer's nonlinearity amplifies
whatever drift the parent's input already had; caching it independently
just pays _is_close_enough's mandatory .item() sync on the way to
discovering that, same story as AMD_TUNED_TORCH_SIMILARITY_CACHE_MAX_MISSES's
volume-decoding measurement, but for every leaf submodule instead of one
decoder. Measured in practice on Hunyuan3D-2's shape-generation DiT
sampling loop: over a single 30-step, ~4-second run, 777 independently
auto-wrapped inner Linear/RMSNorm/GELU/QKNorm/LayerNorm instances each
separately warmed up and gave up, tripping their own max_consecutive_
misses breaker 158 times combined, spread across nearly the whole loop --
while the one module actually invoked once per external sampling-loop
timestep (the top-level DiT itself, exactly this file's own canonical safe
example) never got an uncontested chance to show whether *it* was
cacheable, since none of that overhead or bookkeeping has anything to do
with it. Skipping nested calls entirely collapses that down to checking
only the top-level instance.

Set to "0" (or "false"/"False") to cache every nested call independently
instead, the pre-existing behavior -- worth reaching for only if you've
confirmed a specific architecture genuinely benefits from caching an inner
block whose own input drifts smoothly even when its parent's doesn't (the
opposite of the common case above); reach for the per-instance/per-class
opt-out (`module._amd_tuned_torch_no_cache = True`) instead if it's only a handful
of modules you want nested caching disabled for, rather than turning this
off globally. `enable_module_cache(skip_nested=...)` overrides the
environment variable for a call you make yourself, same pattern as every
other module-cache knob above. SimilarityCache's own explicit decorator/
direct-construction API has no equivalent -- there's no ambient notion of
"nested" for a free-standing cache you construct and call yourself.

AMD_TUNED_TORCH_SIMILARITY_CACHE_COMPILE (default off, "0") makes every
auto-created per-module cache torch.compile its module's forward on that
module's first real (cache-miss) call -- see "Pairing with
amd_tuned_torch.torch_compile" above for the mechanics. Defaults to OFF here,
unlike AMD_TUNED_TORCH_SIMILARITY_CACHE_MAX_BYTES, for a reason that's much
narrower than it used to be: with AMD_TUNED_TORCH_SIMILARITY_CACHE_SKIP_NESTED at
its own default (ON), only the outermost module actually invoked from
outside amd_tuned_torch's own machinery ever gets a SimilarityCache at all -- so
turning this on now compiles that same small set of top-level modules
(a UNet, a VAE, a DiT) amd_tuned_torch.torch_compile.compile_module() would target
manually anyway, not hundreds of leaf modules individually. It's still off
by default because it's a bigger behavior change to opt into blindly
(first-call max-autotune latency on whichever modules turn out to be
top-level, sight unseen) than a knob that's merely "usually fine" -- but
if you've confirmed via debug_summary() which modules the auto-cache is
actually wrapping and they're the right handful, this flag is a
reasonable way to compile them without touching call sites. The old
concern -- compiling every leaf module (every nn.Linear, every norm)
individually, a well-known torch.compile anti-pattern that captures far
less fusion than compiling one larger containing module -- only applies
if you've also set skip_nested=False (see that flag above), since that's
what restores per-submodule wrapping. Prefer calling
amd_tuned_torch.torch_compile.compile_module() yourself on a handful of top-level
submodules instead if you'd rather pick them explicitly than trust
whichever modules happen to be invoked from outside amd_tuned_torch.

Debugging
---------
Every SimilarityCache (explicit or module-level auto-cache) tracks plain
hit/miss/bypass counters regardless of any debug flag -- call
amd_tuned_torch.cache.debug_summary() to print a per-module table of them for the
module-level auto-cache (hits, misses, size_bypassed, breaker_bypassed,
auto_disabled, trip_reason), sorted by call volume, right after the pass
you're investigating. trip_reason says WHICH breaker fired and why -- a
nonzero hits count next to a hit-rate trip_reason is the signature of the
coincidental-hit problem AMD_TUNED_TORCH_SIMILARITY_CACHE_MIN_HIT_RATE exists for
(see that section below): real hits happened, just not often enough to be
worth the check. AMD_TUNED_TORCH_SIMILARITY_CACHE_DEBUG (default off, "0") turns on
additional lifecycle logging through the standard `logging` module (logger
"amd_tuned_torch.cache") -- cache creation, and the circuit breaker actually
tripping -- self-contained: setting the env var alone is enough, it
attaches its own handler so output appears even if the host application
never configures Python logging itself.

AMD_TUNED_TORCH_SIMILARITY_CACHE_DEBUG_FILE (default unset), if set to a path,
additionally (or instead -- see below) writes that same lifecycle logging
to a file via logging.FileHandler (append mode, so nothing is lost across
restarts of a long-running process -- delete the file yourself for a clean
slate). Setting this alone is enough to turn debug logging on, same as
AMD_TUNED_TORCH_SIMILARITY_CACHE_DEBUG -- you don't need both set just to get file
output. Exists because stderr is easy to lose track of in a long-running
Gradio/server process (scrollback limits, a supervisor that discards it,
simply not having a terminal attached at the moment something interesting
happens) -- a file survives all of that and can be tail -f'd or grepped
after the fact. If AMD_TUNED_TORCH_SIMILARITY_CACHE_DEBUG is ALSO explicitly set,
you get both the stderr handler and the file handler; if only
AMD_TUNED_TORCH_SIMILARITY_CACHE_DEBUG_FILE is set, you get the file only (no
extra stderr noise). Each log line includes a timestamp
(%(asctime)s) specifically for this file case -- correlating a trip event
against, say, a request's wall-clock time in your own application logs.

If a fix here doesn't seem to be taking effect in a running process, check
`amd_tuned_torch.cache.__file__` first -- a stale/separately-installed copy of
amd_tuned_torch (e.g. under a conda env, `pip install`ed rather than the editable
`-e` install pointing at this checkout) is a more common cause than the
logic being wrong.
"""
from __future__ import annotations

import collections
import functools
import logging
import os
import threading
import time
import weakref
from typing import Any, Callable, Optional

import torch

log = logging.getLogger("amd_tuned_torch.cache")

# AMD_TUNED_TORCH_SIMILARITY_CACHE_DEBUG (default off, "0"): when set, lifecycle
# events (cache created, circuit breaker tripped, reset) are logged at
# INFO/DEBUG through the standard `logging` module (logger "amd_tuned_torch.cache")
# instead of staying silent. Read once at import time -- purely a logging
# verbosity switch, does not change any caching/skip/compile behavior.
#
# Self-contained on purpose: setting the env var alone is enough to see
# output, even if the host application (e.g. a Gradio app that never calls
# logging.basicConfig()) does zero logging setup of its own -- when this is
# on, a StreamHandler is attached directly to the "amd_tuned_torch.cache" logger
# below (not the root logger, so this never changes what any other logger
# in the process does). Without that, AMD_TUNED_TORCH_SIMILARITY_CACHE_DEBUG=1
# would only make cache.py *call* logging.debug()/.info() -- Python's
# logging drops everything below WARNING by default when nothing has
# configured a handler, so those calls would silently go nowhere, which is
# exactly what was happening before this was added.
#
# Off by default so it never adds per-call overhead (a disabled logger call
# is a cheap early-return, but this also skips even constructing most of
# the log message args) for anyone not actively debugging cache behavior.
_DEBUG_ENV_SET = os.environ.get("AMD_TUNED_TORCH_SIMILARITY_CACHE_DEBUG", "0") not in ("0", "", "false", "False")
# AMD_TUNED_TORCH_SIMILARITY_CACHE_DEBUG_FILE (default unset): a path, not a
# boolean -- see the module docstring. Setting this alone is enough to
# turn debug logging on (no need to also set AMD_TUNED_TORCH_SIMILARITY_CACHE_
# DEBUG=1 just to get file output).
_DEBUG_FILE = os.environ.get("AMD_TUNED_TORCH_SIMILARITY_CACHE_DEBUG_FILE", "")
_DEBUG = _DEBUG_ENV_SET or bool(_DEBUG_FILE)

if _DEBUG:
    log.setLevel(logging.DEBUG)
    _formatter = logging.Formatter("[amd_tuned_torch.cache] %(asctime)s %(levelname)s %(message)s")
    # Avoid duplicate handlers if this module reloads -- checked per
    # handler type since a plain stderr handler and a file handler are
    # independent opt-ins that can both be present at once.
    if _DEBUG_ENV_SET and not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in log.handlers
    ):
        _stream_handler = logging.StreamHandler()
        _stream_handler.setFormatter(_formatter)
        log.addHandler(_stream_handler)
    if _DEBUG_FILE and not any(isinstance(h, logging.FileHandler) for h in log.handlers):
        _file_handler = logging.FileHandler(_DEBUG_FILE)  # default mode="a": append, never truncate
        _file_handler.setFormatter(_formatter)
        log.addHandler(_file_handler)


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) not in ("0", "", "false", "False")


class SimilarityCache:
    """Skip calling `fn` and reuse its last output whenever the current
    call's primary tensor argument is close enough (relative L1 distance)
    to the previous call's primary argument. Only the *first* positional
    argument drives the skip decision -- any further positional/keyword
    arguments are assumed not to affect it (e.g. static config passed every
    call) and are simply forwarded to `fn` unchanged, cache hit or not.

    thresh: max relative L1 distance between consecutive primary-argument
        tensors for the previous output to be considered close enough to
        reuse. 0 disables skipping entirely (equivalent to enabled=False,
        but still pays the distance-check cost every call).
    enabled: explicit on/off, overriding AMD_TUNED_TORCH_ENABLE_SIMILARITY_CACHE.
        None (the default) reads that environment variable once, at
        construction time -- see the module docstring.
    max_bytes: if the primary argument's tensor is larger than this many
        bytes (numel() * element_size()), the call bypasses caching
        entirely -- fn is called directly and neither previous_input nor
        previous_output is touched, so an oversized call never evicts a
        smaller cached pair and never gets held onto itself. None (the
        default) means no size limit. Exists because previous_input/
        previous_output are held for the cache's entire lifetime (until
        the next call or an explicit reset()) -- for a long-lived cache
        (e.g. one per nn.Module instance under enable_module_cache(), kept
        alive for the whole process) that turns large, otherwise-ephemeral
        activations (a VAE decoder's image-sized tensors, say) into a
        standing VRAM cost, not just a bounded working set. See
        enable_module_cache()'s max_bytes default for module-level caching.
    min_bytes: the mirror-image bound -- if the primary argument's tensor
        is *smaller* than this many bytes, the call likewise bypasses
        caching entirely (same passthrough semantics as max_bytes: fn is
        called directly, previous_input/previous_output untouched). None
        (the default here -- explicit decorator/direct-construction usage)
        means no lower limit; the module-level auto-cache defaults this to
        4 KiB instead (see enable_module_cache()'s min_bytes and
        AMD_TUNED_TORCH_SIMILARITY_CACHE_MIN_BYTES below), for a different reason
        than max_bytes's memory-safety story: below a few KiB, the fixed
        per-call overhead of the distance check itself (an elementwise
        subtract + two reductions + a .item() sync in _is_close_enough) --
        and, with compile=True, torch.compile's own per-call dispatch/guard
        overhead -- stops being negligible next to just calling `fn` again,
        so caching a genuinely small tensor can be a net slowdown.
    max_consecutive_misses: circuit breaker -- if this many calls in a row
        are NOT close enough to skip (misses), the cache auto-disables
        itself: every later call bypasses the distance check entirely
        (still runs `fn`, or the compiled version of it if compile=True,
        just without ever computing _is_close_enough again) until reset()
        is called. None (the default here) means never auto-disable. This
        exists for the same reason as min_bytes but addresses a case
        min_bytes/max_bytes can't: a callable that's repeatedly invoked
        with inputs that are simply never similar to each other -- e.g. a
        geometry decoder called once per spatial chunk during volume
        decoding, where every chunk queries a different, unrelated region
        and a "hit" essentially never happens regardless of tensor size.
        In that case _is_close_enough's mandatory .item() sync (a GPU->CPU
        round trip that stalls whatever async kernel-launch pipelining
        would otherwise happen) is pure overhead paid on every call for
        zero skip benefit -- measured to roughly halve throughput on
        exactly that workload. Once the miss streak proves the pattern
        doesn't hold, there's no reason to keep paying for the check.
        Consecutive-miss count resets to 0 on every hit (a hit is evidence
        the cache IS providing value, worth continuing to check for) and
        via reset() (a fresh "run" deserves a fresh chance, same as
        previous_input/previous_output -- unlike the compiled callable,
        which reset() deliberately leaves alone). See
        enable_module_cache()'s max_consecutive_misses default (5) and
        AMD_TUNED_TORCH_SIMILARITY_CACHE_MAX_MISSES for the module-level auto-cache.
    min_hit_rate: a second, independent circuit breaker -- rate-based
        instead of streak-based. Once at least hit_rate_min_samples calls
        have landed within the trailing hit_rate_window_seconds, if the
        fraction of those that were hits is below min_hit_rate, the cache
        auto-disables itself the same way max_consecutive_misses does
        (same _auto_disabled flag, same distance-check bypass). None (the
        default here) disables this breaker entirely -- no deque is even
        allocated, zero overhead, when unset.

        Exists for a gap max_consecutive_misses leaves open: a workload
        with occasional *coincidental* hits (not genuinely similar inputs,
        just aggregate statistics that happen to land within thresh) resets
        the streak counter every time one occurs, which can indefinitely
        postpone that breaker no matter how rarely real skip value shows
        up. This is exactly what happens decoding a dense 3D query grid --
        consecutive chunks are spatially adjacent, not independently
        random, so _is_close_enough's single coarse aggregate-mean-
        difference metric over a whole large batch can coincidentally read
        as "close enough" between two chunks whose individual query points
        are all different -- silently serving the wrong chunk's decoded
        output, not just staying stale. See AMD_TUNED_TORCH_SIMILARITY_CACHE_
        MIN_HIT_RATE in the module docstring for the full writeup,
        including why that's a correctness problem this breaker bounds the
        cost of but doesn't fix -- the reliable fix for "even occasional
        wrong hits are unacceptable" is the per-module opt-out
        (`module._amd_tuned_torch_no_cache = True`), not either circuit breaker.

        Unlike previous_input/max_consecutive_misses, this tracks over
        wall-clock TIME, not call count and not per-run: the property being
        measured (is checking still paying for itself, recently) doesn't
        care how many calls happened, only how much of the recent past was
        wasted -- a bursty workload and a slow one should trip at the same
        *elapsed time* of poor value, not the same call count. reset()
        still clears the tracked history (see reset()'s docstring).
    hit_rate_window_seconds: how far back (wall-clock seconds) min_hit_rate
        looks when computing the recent hit rate. Default 2.0 -- short
        enough to react quickly once a workload's actual behavior settles
        (e.g. right after the first few chunks of a volume decode), long
        enough to smooth over call-to-call noise. Only meaningful when
        min_hit_rate is set.
    hit_rate_min_samples: minimum number of calls that must have landed
        within hit_rate_window_seconds before min_hit_rate is trusted
        enough to act on. Default 20 -- avoids tripping (or deciding not
        to) on a handful of calls right after construction/reset(), before
        the window has enough data to be statistically meaningful. Only
        meaningful when min_hit_rate is set.
    compile: if True, `fn` is run through amd_tuned_torch.torch_compile.compile_fn
        the first time this cache actually calls it (a miss -- skipped
        calls never trigger compilation, there's nothing to compile until
        the real function has run at least once), and that compiled
        callable is reused for every later miss instead of recompiling.
        See "Pairing with amd_tuned_torch.torch_compile" in the module docstring.
        False (the default) leaves `fn` exactly as passed in, no import of
        amd_tuned_torch.torch_compile even attempted.
    compile_kwargs: extra keyword arguments forwarded to
        amd_tuned_torch.torch_compile.compile_fn (e.g. dict(fullgraph=True) or
        dict(dynamic=True)) when compile=True. Ignored otherwise.
    debug_name: a label used only in AMD_TUNED_TORCH_SIMILARITY_CACHE_DEBUG log
        lines (see the module docstring) -- has no effect on behavior.
        None (the default) falls back to id(self) in log output.

    Every instance also tracks plain counters -- hits, misses,
    size_bypassed, breaker_bypassed -- for introspection regardless of
    AMD_TUNED_TORCH_SIMILARITY_CACHE_DEBUG (see debug_summary() for the
    module-level auto-cache's aggregate view across every instance).
    """

    def __init__(self, thresh: float = 0.01, enabled: Optional[bool] = None,
                 max_bytes: Optional[int] = None, min_bytes: Optional[int] = None,
                 max_consecutive_misses: Optional[int] = None,
                 min_hit_rate: Optional[float] = None,
                 hit_rate_window_seconds: float = 2.0,
                 hit_rate_min_samples: int = 20,
                 compile: bool = False, compile_kwargs: Optional[dict] = None,
                 debug_name: Optional[str] = None):
        self.thresh = thresh
        self.enabled = _env_flag("AMD_TUNED_TORCH_ENABLE_SIMILARITY_CACHE") if enabled is None else enabled
        self.max_bytes = max_bytes
        self.min_bytes = min_bytes
        self.max_consecutive_misses = max_consecutive_misses
        self.min_hit_rate = min_hit_rate
        self.hit_rate_window_seconds = hit_rate_window_seconds
        self.hit_rate_min_samples = hit_rate_min_samples
        self.compile = compile
        self.compile_kwargs = compile_kwargs or {}
        self.debug_name = debug_name
        self.previous_input: Optional[torch.Tensor] = None
        self.previous_output: Any = None
        self._consecutive_misses = 0
        self._auto_disabled = False
        self.trip_reason: Optional[str] = None
        self._compiled_fn: Optional[Callable] = None
        self.hits = 0
        self.misses = 0
        self.size_bypassed = 0
        self.breaker_bypassed = 0
        # (timestamp, was_hit) for every call that reached _is_close_enough,
        # trimmed to the trailing hit_rate_window_seconds -- only populated
        # when min_hit_rate is set (see _record_outcome_and_maybe_trip).
        self._outcomes: "collections.deque[tuple[float, bool]]" = collections.deque()
        self._outcome_hits = 0  # running count of hits currently in _outcomes
        if _DEBUG:
            log.debug(
                "created %s (thresh=%s max_bytes=%s min_bytes=%s "
                "max_consecutive_misses=%s min_hit_rate=%s compile=%s)",
                self._label(), thresh, max_bytes, min_bytes, max_consecutive_misses,
                min_hit_rate, compile,
            )

    def _label(self) -> str:
        return self.debug_name if self.debug_name is not None else f"SimilarityCache@{id(self):#x}"

    def _trip(self, reason: str) -> None:
        self._auto_disabled = True
        self.trip_reason = reason
        if _DEBUG:
            log.info("%s auto-disabled (%s) -- skipping distance check until reset()",
                      self._label(), reason)

    def _record_outcome_and_maybe_trip(self, is_hit: bool) -> None:
        """Feed the rolling hit-rate window and trip the breaker if the
        recent hit rate has fallen below min_hit_rate. No-op (and never
        allocates the deque's contents) when min_hit_rate is None -- see
        that param's docstring for why this exists alongside
        max_consecutive_misses rather than instead of it."""
        if self.min_hit_rate is None:
            return
        now = time.monotonic()
        self._outcomes.append((now, is_hit))
        if is_hit:
            self._outcome_hits += 1
        cutoff = now - self.hit_rate_window_seconds
        while self._outcomes and self._outcomes[0][0] < cutoff:
            _, was_hit = self._outcomes.popleft()
            if was_hit:
                self._outcome_hits -= 1
        sample_count = len(self._outcomes)
        if sample_count < self.hit_rate_min_samples:
            return
        hit_rate = self._outcome_hits / sample_count
        if hit_rate < self.min_hit_rate:
            self._trip(
                f"hit rate {hit_rate:.0%} over last {sample_count} calls "
                f"in {self.hit_rate_window_seconds:.1f}s below min_hit_rate {self.min_hit_rate:.0%}"
            )

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def reset(self) -> None:
        """Forget the cached input/output, and re-arm both circuit breakers
        (max_consecutive_misses and min_hit_rate) if either had tripped --
        call this between unrelated runs (e.g. a new generation, a new
        video) so the first call of the next run is never mistakenly
        treated as similar to the last call of the previous one, and so a
        run that previously proved unhelpful (auto-disabled) gets a fresh
        chance to prove otherwise this time. min_hit_rate's tracked
        hit/miss history is unconditionally cleared here too, same as
        previous_input -- a genuinely new run deserves a genuinely fresh
        sample, not leftover history from whatever the previous run was
        doing (outside of reset(), that history is windowed by wall-clock
        time on its own -- see min_hit_rate's docstring -- but reset() is a
        harder, immediate clear, not just letting the window age out). Does
        NOT clear the compiled callable (see compile's docstring entry) --
        that isn't stale, learned-per-run state the way input/output/
        breaker state are."""
        if _DEBUG and self._auto_disabled:
            log.info("%s reset (was auto-disabled: %s)", self._label(), self.trip_reason)
        self.previous_input = None
        self.previous_output = None
        self._consecutive_misses = 0
        self._auto_disabled = False
        self.trip_reason = None
        self._outcomes.clear()
        self._outcome_hits = 0

    def _is_close_enough(self, x: torch.Tensor) -> bool:
        if self.previous_input is None or self.previous_input.shape != x.shape:
            return False
        try:
            denom = self.previous_input.abs().mean().clamp(min=1e-8)
            distance = ((x - self.previous_input).abs().mean() / denom).item()
        except RuntimeError:
            # dtype/device mismatch or similar -- treat it as a cache miss,
            # never let the fast path crash a call that would otherwise
            # have succeeded.
            return False
        return distance < self.thresh

    def _exceeds_size_limit(self, x: torch.Tensor) -> bool:
        return self.max_bytes is not None and x.numel() * x.element_size() > self.max_bytes

    def _below_size_limit(self, x: torch.Tensor) -> bool:
        return self.min_bytes is not None and x.numel() * x.element_size() < self.min_bytes

    def _real_fn(self, fn: Callable) -> Callable:
        """The callable to actually invoke for a real (non-skipped) call:
        `fn` itself, unless compile=True, in which case `fn` is compiled at
        most once -- lazily, on the first call that reaches here -- and
        every subsequent real call reuses that same compiled callable
        regardless of which `fn` object is passed on later calls (safe
        because every call into a given SimilarityCache is assumed to
        invoke the same underlying computation -- see the module
        docstring's note on the module-level auto-cache's per-call
        `_forward` closures for why that assumption holds there too)."""
        if not self.compile:
            return fn
        if self._compiled_fn is None:
            from . import torch_compile  # lazy: avoid importing torch_compile at all when compile=False
            if not torch_compile.available():
                return fn
            self._compiled_fn = torch_compile.compile_fn(fn, **self.compile_kwargs)
        return self._compiled_fn

    def call(self, fn: Callable, x: Any, *args: Any, **kwargs: Any) -> Any:
        if not self.enabled or not isinstance(x, torch.Tensor):
            return fn(x, *args, **kwargs)
        if self._auto_disabled:
            # A circuit breaker already tripped -- either max_consecutive_
            # misses (see self.trip_reason for which) or min_hit_rate --
            # skip the distance check entirely, including its mandatory
            # .item() sync, for every call from here until reset(). Still
            # routed through _real_fn so a compile=True callable keeps its
            # compiled speedup even after the skip mechanism itself has
            # given up.
            self.breaker_bypassed += 1
            return self._real_fn(fn)(x, *args, **kwargs)
        if self._exceeds_size_limit(x) or self._below_size_limit(x):
            # Outside the [min_bytes, max_bytes] window -- passthrough, and
            # deliberately leave previous_input/previous_output alone
            # rather than clobbering a properly-sized cached pair with this
            # call's (bypassed, so never cached) input/output.
            self.size_bypassed += 1
            return fn(x, *args, **kwargs)
        if self._is_close_enough(x):
            self.hits += 1
            self._consecutive_misses = 0  # a hit is evidence the check is worth its cost
            self._record_outcome_and_maybe_trip(True)
            return self.previous_output
        self.misses += 1
        if self.max_consecutive_misses is not None:
            self._consecutive_misses += 1
            if self._consecutive_misses >= self.max_consecutive_misses:
                self._trip(f"{self._consecutive_misses} consecutive misses")
        if not self._auto_disabled:
            self._record_outcome_and_maybe_trip(False)
        output = self._real_fn(fn)(x, *args, **kwargs)
        # Detach+clone the *input* reference kept for future comparisons --
        # cheap insurance against silently wrong skip decisions if the
        # caller later mutates the tensor object it passed in. The output
        # is intentionally not cloned; see the module docstring's aliasing
        # caveat.
        self.previous_input = x.detach().clone()
        self.previous_output = output
        return output

    def __call__(self, fn: Callable) -> Callable:
        """Use a SimilarityCache instance directly as a decorator."""

        @functools.wraps(fn)
        def wrapper(x, *args, **kwargs):
            return self.call(fn, x, *args, **kwargs)

        wrapper.cache = self
        return wrapper


def similarity_cached(thresh: float = 0.1, enabled: Optional[bool] = None,
                       max_bytes: Optional[int] = None, min_bytes: Optional[int] = None,
                       max_consecutive_misses: Optional[int] = None,
                       min_hit_rate: Optional[float] = None,
                       hit_rate_window_seconds: float = 2.0,
                       hit_rate_min_samples: int = 20,
                       compile: bool = False, compile_kwargs: Optional[dict] = None) -> Callable:
    """Decorator sugar for SimilarityCache -- see its docstring and the
    module docstring for the exact semantics and the
    AMD_TUNED_TORCH_ENABLE_SIMILARITY_CACHE environment flag. Each decorated
    function gets its own independent SimilarityCache instance, reachable
    as `decorated_fn.cache` for reset()/enable()/disable()."""
    return SimilarityCache(thresh=thresh, enabled=enabled, max_bytes=max_bytes,
                            min_bytes=min_bytes, max_consecutive_misses=max_consecutive_misses,
                            min_hit_rate=min_hit_rate, hit_rate_window_seconds=hit_rate_window_seconds,
                            hit_rate_min_samples=hit_rate_min_samples,
                            compile=compile, compile_kwargs=compile_kwargs)


class KeyedCache:
    """Exact, identity-keyed memoization -- SimilarityCache's sibling for a
    fundamentally different reuse pattern. SimilarityCache answers "is this
    call's input close enough to the *previous* call's to approximate its
    output" -- a statistical judgment call, correct only when consecutive
    calls are *expected* to have similar-but-not-identical inputs
    (diffusion timesteps, video frames). KeyedCache answers a different,
    exact question: "has this *exact logical query* already been computed,
    ever" -- correct whenever the caller can name that query's identity,
    with no approximation and no dependence on how similar two queries'
    tensor contents happen to look.

    THIS DISTINCTION IS NOT COSMETIC. SimilarityCache's _is_close_enough is
    a single coarse statistic (mean absolute difference over an entire
    tensor, relative to its mean magnitude) -- for a large batch of
    quasi-uniformly-distributed values (e.g. adjacent chunks of a dense 3D
    query grid), that aggregate can coincidentally read as "close enough"
    even though every individual element differs, silently returning a
    completely different query's answer. No thresh value fixes this --
    tightening it only shrinks, never eliminates, the chance of a
    coincidental match, because the failure is in what's being measured
    (aggregate similarity), not how tight the tolerance on it is. Whenever
    a workload's correct answer genuinely depends on exact input identity
    rather than tolerating drift -- most non-diffusion, non-video,
    non-iterative-refinement workloads -- KeyedCache is the correct tool,
    not a stricter SimilarityCache.

    Usage (explicit only -- like SimilarityCache's decorator/direct-
    construction usage, and unlike SimilarityCache there's no module-level
    auto-cache equivalent: the whole point of that auto-cache is not
    needing domain knowledge, but a key IS domain knowledge, something
    only the caller can supply):

        cache = amd_tuned_torch.cache.KeyedCache()
        key = (mesh_id, resolution, chunk_index)
        occupancy = cache.call(geo_decoder, key, queries=chunk_queries, latents=latents)

    key must be hashable and must NOT be derived from tensor contents --
    computing a key from a GPU tensor's actual values (even a GPU-side
    hash) would reintroduce exactly the sync-stall problem SimilarityCache
    already has for this kind of workload, for no benefit over just
    comparing values directly. Derive it from whatever logically identifies
    the query instead -- an index, a resolution, a bounding box, a request
    ID -- values you already have on hand before the tensor is ever built.

    Whether this actually helps depends entirely on whether your workload
    ever presents the SAME key twice. A loop that visits each key exactly
    once (e.g. a single sweep over generate_dense_grid_points()'s chunks,
    where every chunk is queried once and the correct output differs by
    `latents` per generation -- so even the exact same spatial chunk index
    decodes to something different for a different mesh) gets zero hits
    and pure overhead (key construction, a dict lookup, unbounded retained
    outputs) from wrapping it in a KeyedCache. Only wrap a call site where
    you've confirmed (or can reasonably expect) repeated identical queries
    -- e.g. the same chunk re-requested across retries of the same
    generation, not a single forward sweep.

    enabled: explicit on/off. None (the default) reads
        AMD_TUNED_TORCH_ENABLE_SIMILARITY_CACHE, same environment flag
        SimilarityCache reads -- both are opt-in caching mechanisms this
        project gates behind the same "genuinely risky by default" flag
        (see the module docstring), even though KeyedCache carries none of
        SimilarityCache's approximation risk; what they share is the more
        basic risk any cache carries: silently serving a stale value if
        the underlying computation's other inputs (e.g. `latents` here)
        change without changing the key.
    max_entries: caps how many distinct keys are retained at once. None
        (the default) means unbounded -- fine for a workload with a known,
        bounded key space (a fixed octree's chunks), a standing risk for
        one with an open-ended key space (e.g. keying by request ID across
        a long-running server). Once the cap is reached, new keys are
        computed and returned normally but NOT stored (the simplest
        possible bound -- no LRU/eviction bookkeeping -- appropriate here
        since exceeding max_entries is meant as a signal this call site's
        key space turned out bigger than expected, not a steady-state
        cache-replacement policy to tune).
    debug_name: a label used only in AMD_TUNED_TORCH_SIMILARITY_CACHE_DEBUG log
        lines -- has no effect on behavior. None (the default) falls back
        to id(self).
    """

    def __init__(self, enabled: Optional[bool] = None, max_entries: Optional[int] = None,
                 debug_name: Optional[str] = None):
        self.enabled = _env_flag("AMD_TUNED_TORCH_ENABLE_SIMILARITY_CACHE") if enabled is None else enabled
        self.max_entries = max_entries
        self.debug_name = debug_name
        self._cache: dict = {}
        self.hits = 0
        self.misses = 0
        self.not_stored = 0  # misses computed but dropped because max_entries was already reached
        if _DEBUG:
            log.debug("created %s (max_entries=%s)", self._label(), max_entries)

    def _label(self) -> str:
        return self.debug_name if self.debug_name is not None else f"KeyedCache@{id(self):#x}"

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def reset(self) -> None:
        """Forget every cached key/output -- call this when the underlying
        computation's *other* inputs change in a way the key doesn't
        capture (e.g. a new mesh's `latents`, if you keyed only by chunk
        index/resolution and not by mesh identity too -- reset() between
        meshes is the cheap fix; keying by mesh identity as well is the
        robust one, see the module docstring's key-design guidance)."""
        if _DEBUG and self._cache:
            log.info("%s reset (%d entries, %d hits, %d misses discarded)",
                      self._label(), len(self._cache), self.hits, self.misses)
        self._cache.clear()
        self.hits = 0
        self.misses = 0
        self.not_stored = 0

    def call(self, fn: Callable, key: Any, *args: Any, **kwargs: Any) -> Any:
        """Look up `key`; on a hit, return the stored output without
        calling `fn` at all. On a miss, call `fn(*args, **kwargs)` -- note
        `key` is NOT passed to `fn`, it's purely the cache's lookup handle
        -- store the result (unless max_entries was already reached), and
        return it."""
        if not self.enabled:
            return fn(*args, **kwargs)
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        output = fn(*args, **kwargs)
        if self.max_entries is None or len(self._cache) < self.max_entries:
            self._cache[key] = output
        else:
            self.not_stored += 1
            if _DEBUG:
                log.debug("%s at max_entries=%d -- not storing key %r",
                           self._label(), self.max_entries, key)
        return output


def keyed_cached(enabled: Optional[bool] = None, max_entries: Optional[int] = None) -> "KeyedCache":
    """Sugar for KeyedCache -- mirrors similarity_cached()'s naming, but
    returns the KeyedCache instance itself rather than a decorated
    function: unlike SimilarityCache.__call__, there's no sensible way to
    decorate a plain function with KeyedCache, since the wrapped callable
    needs a `key` supplied per call (see KeyedCache.call), not just an `x`
    inferred from the first positional argument. Call `.call(fn, key, ...)`
    on the returned instance directly."""
    return KeyedCache(enabled=enabled, max_entries=max_entries)


# ---------------------------------------------------------------------------
# Global torch.nn.Module.__call__ patching -- see the module docstring's
# "Global module-level caching" section for the full risk/safety writeup
# before enabling this. Off by default; only installed automatically when
# AMD_TUNED_TORCH_ENABLE_SIMILARITY_CACHE is set (see the bottom of this file).
# ---------------------------------------------------------------------------

_DEFAULT_THRESH_ENV = "AMD_TUNED_TORCH_SIMILARITY_CACHE_THRESH"
_DEFAULT_MAX_BYTES_ENV = "AMD_TUNED_TORCH_SIMILARITY_CACHE_MAX_BYTES"
_DEFAULT_MAX_BYTES = 64 * 1024 * 1024  # 64 MiB
_DEFAULT_MIN_BYTES_ENV = "AMD_TUNED_TORCH_SIMILARITY_CACHE_MIN_BYTES"
_DEFAULT_MIN_BYTES = 4 * 1024  # 4 KiB
_DEFAULT_MAX_MISSES_ENV = "AMD_TUNED_TORCH_SIMILARITY_CACHE_MAX_MISSES"
_DEFAULT_MAX_MISSES = 5
_DEFAULT_MIN_HIT_RATE_ENV = "AMD_TUNED_TORCH_SIMILARITY_CACHE_MIN_HIT_RATE"
_DEFAULT_MIN_HIT_RATE = 0.2
_DEFAULT_AUTO_EXCLUDE_ENV = "AMD_TUNED_TORCH_SIMILARITY_CACHE_AUTO_EXCLUDE"
_DEFAULT_CLASS_COOLDOWN_ENV = "AMD_TUNED_TORCH_SIMILARITY_CACHE_CLASS_COOLDOWN"
_DEFAULT_SKIP_NESTED_ENV = "AMD_TUNED_TORCH_SIMILARITY_CACHE_SKIP_NESTED"
_DEFAULT_COMPILE_ENV = "AMD_TUNED_TORCH_SIMILARITY_CACHE_COMPILE"

_module_caches: "weakref.WeakKeyDictionary[torch.nn.Module, SimilarityCache]" = (
    weakref.WeakKeyDictionary()
)
# class -> time.monotonic() deadline until which every instance of that
# class skips the wrapper, set by _cached_module_call when any one
# instance's breaker trips -- see AMD_TUNED_TORCH_SIMILARITY_CACHE_CLASS_COOLDOWN.
# A plain dict, not weakref-keyed: classes are long-lived (module-level
# objects), unlike the module instances _module_caches tracks.
_class_cooldown_until: "dict[type, float]" = {}
_module_cache_thresh = 0.1
_module_cache_max_bytes: Optional[int] = _DEFAULT_MAX_BYTES
_module_cache_min_bytes: Optional[int] = _DEFAULT_MIN_BYTES
_module_cache_max_misses: Optional[int] = _DEFAULT_MAX_MISSES
_module_cache_min_hit_rate: Optional[float] = _DEFAULT_MIN_HIT_RATE
_module_cache_hit_rate_window_seconds = 2.0
_module_cache_hit_rate_min_samples = 20
_module_cache_auto_exclude = False
_module_cache_class_cooldown_seconds: Optional[float] = None
_module_cache_skip_nested = True
_module_cache_compile = False
_original_module_call: Optional[Callable] = None
_MODULE_CACHE_ENABLED = False

# Thread-local call-depth counter backing AMD_TUNED_TORCH_SIMILARITY_CACHE_SKIP_
# NESTED -- incremented for the duration of a wrapped module's real
# forward (see _cached_module_call's _forward closure), so any
# nn.Module.__call__ reached from inside it can tell it's nested and
# bypass the wrapper. Thread-local, not a plain module global, because a
# server process may run concurrent forward passes on different threads;
# depth must never leak between them.
_nested_call_state = threading.local()


def _in_cached_module_call() -> bool:
    return getattr(_nested_call_state, "depth", 0) > 0


def _grad_safe(*tensors: Any) -> bool:
    """Same check amd_tuned_torch.__init__ uses to gate its own non-autograd
    kernel swaps -- duplicated here (rather than imported) to keep
    amd_tuned_torch.cache import-order-independent of amd_tuned_torch/__init__.py. Reusing
    a cached output tensor from a *different* forward pass while gradients
    are actually needed for this one would silently corrupt autograd, not
    just approximate the forward value -- so caching is skipped whenever
    this returns False, same as it is for amd_tuned_torch's kernel-swap ops."""
    if hasattr(torch, "is_inference_mode_enabled") and torch.is_inference_mode_enabled():
        return True
    if not torch.is_grad_enabled():
        return True
    for t in tensors:
        if isinstance(t, torch.Tensor) and t.requires_grad:
            return False
    return True


def _read_default_thresh() -> float:
    try:
        return float(os.environ.get(_DEFAULT_THRESH_ENV, "0.1"))
    except ValueError:
        return 0.1


def _read_default_max_bytes() -> Optional[int]:
    try:
        value = int(os.environ.get(_DEFAULT_MAX_BYTES_ENV, str(_DEFAULT_MAX_BYTES)))
    except ValueError:
        return _DEFAULT_MAX_BYTES
    return value if value > 0 else None


def _read_default_min_bytes() -> Optional[int]:
    try:
        value = int(os.environ.get(_DEFAULT_MIN_BYTES_ENV, str(_DEFAULT_MIN_BYTES)))
    except ValueError:
        return _DEFAULT_MIN_BYTES
    return value if value > 0 else None


def _read_default_max_misses() -> Optional[int]:
    try:
        value = int(os.environ.get(_DEFAULT_MAX_MISSES_ENV, str(_DEFAULT_MAX_MISSES)))
    except ValueError:
        return _DEFAULT_MAX_MISSES
    return value if value > 0 else None


def _read_default_min_hit_rate() -> Optional[float]:
    try:
        value = float(os.environ.get(_DEFAULT_MIN_HIT_RATE_ENV, str(_DEFAULT_MIN_HIT_RATE)))
    except ValueError:
        return _DEFAULT_MIN_HIT_RATE
    return value if value > 0 else None


def _read_default_auto_exclude() -> bool:
    return _env_flag(_DEFAULT_AUTO_EXCLUDE_ENV)


def _read_default_class_cooldown() -> Optional[float]:
    try:
        value = float(os.environ.get(_DEFAULT_CLASS_COOLDOWN_ENV, "0"))
    except ValueError:
        return None
    return value if value > 0 else None


def _read_default_skip_nested() -> bool:
    return _env_flag(_DEFAULT_SKIP_NESTED_ENV, "1")


def _read_default_compile() -> bool:
    return _env_flag(_DEFAULT_COMPILE_ENV)


def _cached_module_call(self: torch.nn.Module, *args: Any, **kwargs: Any) -> Any:
    orig_call = _original_module_call
    if (
        not args
        or not isinstance(args[0], torch.Tensor)
        or getattr(self, "_amd_tuned_torch_no_cache", False)
        or not _grad_safe(*args, *kwargs.values())
        or (_module_cache_skip_nested and _in_cached_module_call())
    ):
        return orig_call(self, *args, **kwargs)

    if _module_cache_class_cooldown_seconds is not None:
        cls = type(self)
        deadline = _class_cooldown_until.get(cls)
        if deadline is not None:
            if deadline > time.monotonic():
                # Some OTHER instance of this class tripped a breaker
                # recently -- skip straight through without ever touching
                # this instance's own SimilarityCache (it may not even
                # exist yet), same reasoning as auto_exclude but class-wide
                # and time-bounded instead of instance-wide and permanent.
                return orig_call(self, *args, **kwargs)
            del _class_cooldown_until[cls]  # cooldown expired -- resume normal per-instance tracking

    sc = _module_caches.get(self)
    if sc is None:
        sc = SimilarityCache(thresh=_module_cache_thresh, enabled=True,
                              max_bytes=_module_cache_max_bytes,
                              min_bytes=_module_cache_min_bytes,
                              max_consecutive_misses=_module_cache_max_misses,
                              min_hit_rate=_module_cache_min_hit_rate,
                              hit_rate_window_seconds=_module_cache_hit_rate_window_seconds,
                              hit_rate_min_samples=_module_cache_hit_rate_min_samples,
                              compile=_module_cache_compile,
                              debug_name=f"{type(self).__name__}@{id(self):#x}")
        _module_caches[self] = sc

    def _forward(x: torch.Tensor, *rest: Any, **kw: Any) -> Any:
        # Mark every nn.Module.__call__ reached while this real forward is
        # running as nested -- see AMD_TUNED_TORCH_SIMILARITY_CACHE_SKIP_NESTED.
        # Covers every path that reaches _forward at all (a real miss, a
        # size bypass, a breaker bypass), since they all call through here.
        _nested_call_state.depth = getattr(_nested_call_state, "depth", 0) + 1
        try:
            return orig_call(self, x, *rest, **kw)
        finally:
            _nested_call_state.depth -= 1

    was_disabled_before = sc._auto_disabled
    result = sc.call(_forward, args[0], *args[1:], **kwargs)
    just_tripped = sc._auto_disabled and not was_disabled_before

    if just_tripped and _module_cache_class_cooldown_seconds is not None:
        # This instance proving itself uncacheable is decent evidence
        # other instances of the same class (e.g. every RMSNorm in a deep
        # stack, each seeing its own slice of the same never-similar
        # per-chunk activations) will too -- give them all a temporary
        # free pass instead of each independently re-paying its own
        # max_consecutive_misses/min_hit_rate warm-up. Deliberately
        # time-bounded (not auto_exclude's permanent opt-out): if this
        # trip was actually workload-specific rather than class-wide, the
        # cooldown expiring lets per-instance tracking correct itself
        # instead of wrongly blacklisting the whole class forever. See
        # AMD_TUNED_TORCH_SIMILARITY_CACHE_CLASS_COOLDOWN in the module docstring.
        _class_cooldown_until[type(self)] = time.monotonic() + _module_cache_class_cooldown_seconds
        if _DEBUG:
            log.info("%s tripped -- %s class on cooldown for %.1fs",
                      sc._label(), type(self).__name__, _module_cache_class_cooldown_seconds)

    if _module_cache_auto_exclude and sc._auto_disabled and not sc.compile:
        # A circuit breaker just tripped (or had already tripped) and
        # there's no compiled speedup on this cache worth preserving --
        # skip SimilarityCache.call()'s own (already-minimal) dispatch
        # entirely from here on by opting this module instance out via the
        # same attribute the manual opt-out uses. Cheaper than routing
        # through sc.call() forever just to hit its _auto_disabled
        # fast-path every time -- see AMD_TUNED_TORCH_SIMILARITY_CACHE_AUTO_EXCLUDE
        # in the module docstring for the reset()-semantics trade-off this
        # makes (permanent until you clear this attribute yourself or
        # disable_module_cache(), NOT re-armed by sc.reset()).
        self._amd_tuned_torch_no_cache = True
    return result


def enable_module_cache(thresh: Optional[float] = None,
                         max_bytes: Optional[int] = -1,
                         min_bytes: Optional[int] = -1,
                         max_consecutive_misses: Optional[int] = -1,
                         min_hit_rate: Optional[float] = -1,
                         hit_rate_window_seconds: float = 2.0,
                         hit_rate_min_samples: int = 20,
                         auto_exclude: Optional[bool] = None,
                         class_cooldown_seconds: Optional[float] = -1,
                         skip_nested: Optional[bool] = None,
                         compile: Optional[bool] = None) -> None:
    """Patch torch.nn.Module.__call__ process-wide so every module
    instance gets its own similarity cache automatically -- see the module
    docstring's "Global module-level caching" section for the full
    correctness caveat before calling this yourself; it's normally invoked
    for you at import time when AMD_TUNED_TORCH_ENABLE_SIMILARITY_CACHE is set.

    thresh: relative-L1 threshold for every auto-created per-module cache.
        None (the default) reads AMD_TUNED_TORCH_SIMILARITY_CACHE_THRESH.
    max_bytes: size cap (see SimilarityCache.max_bytes) for every
        auto-created per-module cache. The sentinel -1 (the default) reads
        AMD_TUNED_TORCH_SIMILARITY_CACHE_MAX_BYTES; pass None explicitly for no
        limit, or a positive int for an explicit cap in bytes.
    min_bytes: size floor (see SimilarityCache.min_bytes) for every
        auto-created per-module cache. The sentinel -1 (the default) reads
        AMD_TUNED_TORCH_SIMILARITY_CACHE_MIN_BYTES (default "4096", 4 KiB); pass
        None explicitly for no floor, or a positive int for an explicit
        minimum in bytes.
    max_consecutive_misses: circuit-breaker threshold (see
        SimilarityCache.max_consecutive_misses) for every auto-created
        per-module cache. The sentinel -1 (the default) reads
        AMD_TUNED_TORCH_SIMILARITY_CACHE_MAX_MISSES (default "5"); pass None
        explicitly to never auto-disable, or a positive int for an
        explicit miss-streak limit. Defaults to a real limit here, unlike
        compile below -- measured to roughly halve throughput on a real
        workload (a geometry decoder called once per spatial chunk during
        volume decoding, where chunks are never similar to each other) when
        left unbounded, since every one of that decoder's ~16 internal
        submodule calls otherwise pays _is_close_enough's .item() sync on
        every single chunk for a skip that essentially never happens.
    min_hit_rate/hit_rate_window_seconds/hit_rate_min_samples: the
        rate-based circuit breaker (see SimilarityCache.min_hit_rate) for
        every auto-created per-module cache -- catches what
        max_consecutive_misses can't: occasional *coincidental* hits (not
        genuinely similar inputs, just aggregate statistics that happen to
        land within thresh) that reset the streak counter and indefinitely
        postpone it. min_hit_rate's sentinel -1 (the default) reads
        AMD_TUNED_TORCH_SIMILARITY_CACHE_MIN_HIT_RATE (default "0.2", i.e. 20%);
        pass None explicitly to disable it, or a float in (0, 1] for an
        explicit floor. hit_rate_window_seconds/hit_rate_min_samples aren't
        read from their own environment variables -- pass them here
        directly if the 2.0s/20-sample defaults don't suit your call
        frequency. See AMD_TUNED_TORCH_SIMILARITY_CACHE_MIN_HIT_RATE in the module
        docstring for the full writeup, including why this exists
        alongside max_consecutive_misses rather than instead of it, and why
        it bounds the cost of -- but doesn't fix -- the correctness problem
        those coincidental hits are (each one serves a *different* chunk's
        decoded output, not merely a stale approximation).
    auto_exclude: once a per-module cache's circuit breaker trips (either
        one -- and it has no compile=True speedup worth preserving, see
        SimilarityCache.compile), permanently exclude that module instance
        from the wrapper entirely by setting its `_amd_tuned_torch_no_cache = True`
        (the same attribute the manual opt-out uses -- see "Global
        module-level caching" above), instead of continuing to route every
        future call through SimilarityCache.call()'s own (already-minimal)
        _auto_disabled fast-path. None (the default) reads
        AMD_TUNED_TORCH_SIMILARITY_CACHE_AUTO_EXCLUDE (default off, "0").

        Worth the real per-call savings (a weakref dict lookup, the
        grad-safety check, building a fresh closure -- all for every call,
        forever, even once a breaker has tripped and there's nothing left
        to check) for a module that's proven itself never-caching-worthy --
        but changes what reset() means for that module: reset() only
        clears a SimilarityCache's own internal state (previous_input,
        streak, rate history), it can't un-set an attribute on the module
        it doesn't hold a reference to, so once auto-excluded a module
        stays excluded from the wrapper permanently for the rest of the
        process (until you clear `module._amd_tuned_torch_no_cache` yourself, or
        disable_module_cache() and re-enable -- note even that doesn't
        clear it, since the attribute lives on the module object, not in
        this module's own bookkeeping). Off by default because that's a
        real behavior change or reset()-relying code, not merely a speed
        optimization; on for a workload you've already confirmed (e.g. via
        debug_summary()/trip_reason) never benefits from caching regardless
        of run boundaries, it's a straightforward win.
    class_cooldown_seconds: when any one instance's circuit breaker trips,
        put every OTHER instance of that same class on a temporary cooldown
        too -- for that many seconds, ANY instance of the class (whether or
        not it has its own SimilarityCache yet, whether or not it's
        individually seen a single miss) skips straight through, no
        distance check, no per-instance streak/rate bookkeeping. The
        sentinel -1 (the default) reads AMD_TUNED_TORCH_SIMILARITY_CACHE_
        CLASS_COOLDOWN (an unset/non-positive value disables this
        entirely, same as every other "0 means off" knob here); pass None
        explicitly to disable, or a positive number of seconds.

        Exists for the same class of problem auto_exclude addresses --
        redundant per-instance warm-up cost -- but from a different angle:
        auto_exclude waits for THIS instance to prove itself, permanently,
        once it has; class_cooldown_seconds generalizes one instance's
        trip to its whole class immediately, temporarily. The two compose
        (both can be on at once) and suit different situations -- e.g. a
        deep decoder with many RMSNorm/LayerNorm instances that all see
        their own slice of the same never-similar per-chunk activations:
        the first one to trip (fastest, since whichever happens to see the
        least-fortunate coincidental-hit timing) puts the rest on cooldown
        immediately, instead of each independently re-discovering the same
        conclusion over the next several chunks.

        Unlike auto_exclude, this is time-bounded and self-correcting by
        design: if the trip that triggered a cooldown turns out to have
        been specific to that one instance rather than representative of
        the whole class, the cooldown simply expires and per-instance
        tracking (which was never touched for the OTHER instances during
        the cooldown -- their own streaks/rate windows just didn't advance,
        they weren't reset) resumes normally. No correctness cost beyond
        what any circuit breaker already carries (see max_consecutive_
        misses/min_hit_rate above) -- this only ever widens which calls
        skip the check, never widens which calls get a stale *cached
        output* instead of a fresh computation, since cooldown'd calls run
        `fn` for real every time, they just don't check first.
    skip_nested: while a wrapped module's real forward is executing, skip
        wrapping/checking any nn.Module.__call__ reached from inside it --
        only the outermost module actually invoked from outside amd_tuned_torch's
        own machinery ever gets a SimilarityCache. None (the default)
        reads AMD_TUNED_TORCH_SIMILARITY_CACHE_SKIP_NESTED (default ON, "1") --
        see that flag's docstring entry for the full writeup, including
        the Hunyuan3D-2 DiT measurement (777 independently-wrapped inner
        Linear/RMSNorm/GELU/QKNorm/LayerNorm instances tripping their own
        breakers 158 times combined over one 30-step sampling loop) this
        default is based on. Pass False to cache every nested call
        independently instead, the pre-existing behavior.
    compile: torch.compile every auto-created per-module cache's module on
        its first real call (see SimilarityCache.compile). None (the
        default) reads AMD_TUNED_TORCH_SIMILARITY_CACHE_COMPILE (default off) --
        see that flag's docstring entry for why this defaults to off even
        though max_bytes above doesn't. With skip_nested at its default
        (True) this now compiles only the top-level modules the auto-cache
        actually wraps, not every leaf module process-wide; prefer calling
        amd_tuned_torch.torch_compile.compile_module() yourself instead if you'd
        rather pick which modules get compiled explicitly.
    """
    global _original_module_call, _MODULE_CACHE_ENABLED
    global _module_cache_thresh, _module_cache_max_bytes, _module_cache_min_bytes
    global _module_cache_max_misses, _module_cache_min_hit_rate
    global _module_cache_hit_rate_window_seconds, _module_cache_hit_rate_min_samples
    global _module_cache_auto_exclude, _module_cache_class_cooldown_seconds, _module_cache_compile
    global _module_cache_skip_nested
    if _MODULE_CACHE_ENABLED:
        return
    _module_cache_thresh = _read_default_thresh() if thresh is None else thresh
    _module_cache_max_bytes = _read_default_max_bytes() if max_bytes == -1 else max_bytes
    _module_cache_min_bytes = _read_default_min_bytes() if min_bytes == -1 else min_bytes
    _module_cache_max_misses = (
        _read_default_max_misses() if max_consecutive_misses == -1 else max_consecutive_misses
    )
    _module_cache_min_hit_rate = (
        _read_default_min_hit_rate() if min_hit_rate == -1 else min_hit_rate
    )
    _module_cache_hit_rate_window_seconds = hit_rate_window_seconds
    _module_cache_hit_rate_min_samples = hit_rate_min_samples
    _module_cache_auto_exclude = _read_default_auto_exclude() if auto_exclude is None else auto_exclude
    _module_cache_skip_nested = _read_default_skip_nested() if skip_nested is None else skip_nested
    _module_cache_class_cooldown_seconds = (
        _read_default_class_cooldown() if class_cooldown_seconds == -1 else class_cooldown_seconds
    )
    _module_cache_compile = _read_default_compile() if compile is None else compile
    _original_module_call = torch.nn.Module.__call__
    torch.nn.Module.__call__ = _cached_module_call
    _MODULE_CACHE_ENABLED = True


def disable_module_cache() -> None:
    """Restore stock torch.nn.Module.__call__ and forget every per-module
    cache created by enable_module_cache(), including any active
    class-cooldown state (see class_cooldown_seconds)."""
    global _original_module_call, _MODULE_CACHE_ENABLED
    if not _MODULE_CACHE_ENABLED:
        return
    torch.nn.Module.__call__ = _original_module_call
    _original_module_call = None
    _module_caches.clear()
    _class_cooldown_until.clear()
    _MODULE_CACHE_ENABLED = False


def is_module_cache_enabled() -> bool:
    return _MODULE_CACHE_ENABLED


def debug_summary(top_n: Optional[int] = 20, print_output: bool = True) -> list:
    """Snapshot the module-level auto-cache's per-module counters -- answers
    "is this actually helping, and did the circuit breaker engage?" directly
    from a running process, independent of whether AMD_TUNED_TORCH_SIMILARITY_CACHE_
    DEBUG logging was on. Call it right after the pass you care about, e.g.:

        import amd_tuned_torch
        # ... run a Hunyuan3D-2 volume-decode pass ...
        amd_tuned_torch.cache.debug_summary()

    Returns a list of (debug_name, hits, misses, size_bypassed,
    breaker_bypassed, auto_disabled, trip_reason) tuples, sorted by total
    calls descending (top_n=None for every module instead of just the
    busiest ones). Also prints a table unless print_output=False.
    trip_reason (see SimilarityCache._trip) says WHICH breaker fired --
    e.g. "5 consecutive misses" vs "hit rate 12% over last 43 calls in
    2.0s below min_hit_rate 20%" -- the latter is the signature of the
    coincidental-hit problem AMD_TUNED_TORCH_SIMILARITY_CACHE_MIN_HIT_RATE exists
    for (see the module docstring): a module whose hits column is nonzero
    but whose trip_reason still mentions hit rate got occasional real
    (if likely coincidental) hits, just not enough of them to be worth the
    check.

    An empty result is itself diagnostic: if you expected entries and see
    none, either the module-level auto-cache was never enabled for this
    process (is_module_cache_enabled() is False), or -- worth checking
    first if a fix here doesn't seem to be taking effect -- this process
    isn't actually running the amd_tuned_torch build you think it is. Confirm with
    `import amd_tuned_torch.cache; print(amd_tuned_torch.cache.__file__)`.
    """
    rows = []
    for sc in _module_caches.values():
        total = sc.hits + sc.misses + sc.size_bypassed + sc.breaker_bypassed
        rows.append((sc._label(), sc.hits, sc.misses, sc.size_bypassed, sc.breaker_bypassed,
                      sc._auto_disabled, sc.trip_reason, total))
    rows.sort(key=lambda r: r[-1], reverse=True)
    if top_n is not None:
        rows = rows[:top_n]
    if print_output:
        header = (f"{'module':42s} {'hits':>7s} {'misses':>7s} {'size_byp':>9s} "
                  f"{'breaker_byp':>12s} {'auto_disabled':>14s} {'trip_reason':s}")
        print(header)
        print("-" * len(header))
        for name, hits, misses, size_bp, breaker_bp, auto_dis, reason, _total in rows:
            print(f"{name:42s} {hits:7d} {misses:7d} {size_bp:9d} {breaker_bp:12d} "
                  f"{str(auto_dis):>14s} {reason or ''}")
        if not rows:
            print("(no module caches recorded -- module-level auto-cache never "
                  "enabled for this process, or no wrapped module called yet)")
    return [(name, hits, misses, size_bp, breaker_bp, auto_dis, reason)
            for name, hits, misses, size_bp, breaker_bp, auto_dis, reason, _total in rows]


if _env_flag("AMD_TUNED_TORCH_ENABLE_SIMILARITY_CACHE"):
    enable_module_cache()
