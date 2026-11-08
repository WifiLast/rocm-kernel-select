"""
Pure-Python / NumPy re-implementation of CuMesh's `simplify()` (see
../CuMesh/src/simplify.cu and ../CuMesh/cumesh/cumesh.py:320).

CuMesh's decimator is not a classic sequential quadric-edge-collapse (no
priority heap, no one-edge-at-a-time processing). It's a *batched parallel*
scheme designed for a GPU:

  1. Compute a per-vertex quadric error (QEM) from the planes of its
     incident faces.
  2. Score every remaining edge with cost = QEM(midpoint) + lambda_edge_length
     * edge_length^2 + lambda_skinny * (avg triangle-shape penalty) *
     edge_length^2, and mark it invalid (cost = inf) if collapsing it would
     flip a triangle normal.
  3. Each face keeps the *minimum* cost among all edges touching it
     ("propagation").
  4. An edge is only collapsed if it is at or below the current threshold
     AND it is the argmin edge for every face touching either endpoint --
     this is what lets many independent edges collapse in one shot without
     an explicit conflict/lock scheme.
  5. Compact vertices/faces, repeat until `target_num_faces` is reached,
     multiplying the threshold by 10 whenever a pass removes fewer than 1%
     of the remaining faces (mirrors cumesh.py's `simplify()` loop).

This module reproduces that same batched-collapse algorithm on the CPU with
vectorized NumPy instead of CUDA kernels, so it has no dependency on
CUDA/ROCm, PyTorch, or the CuMesh extension. It trades throughput for
portability.

Performance notes: the naive way to write the per-vertex/per-face
scatter-reductions this algorithm needs is `np.add.at` / `np.minimum.at` /
`np.logical_and.at`, but those are notoriously slow in NumPy (they fall back
to an unbuffered, effectively per-element Python-level loop). Every such
reduction here is instead done as either `np.bincount` (for sums -- QEM
accumulation, per-edge skinny-cost/flip-vote tallies) or a sort +
`np.minimum.reduceat` "groupby-min" (for the per-face cost propagation),
both of which are vectorized in C. The per-vertex QEM is also kept as 10
flat (V,) arrays (the upper triangle of the symmetric 4x4 quadric) instead
of a dense (V, 3, 3) matrix, which avoids a large (3F, 3, 3) intermediate.
Together this is roughly two orders of magnitude faster than the first cut
of this module and is what makes multi-million-face meshes tractable on
CPU (still much slower than the CUDA original, just no longer impractical).

Only `simplify()` is reimplemented here — CuMesh's other operations
(fill_holes, remeshing, uv_unwrap, ...) are out of scope.
"""

from typing import Dict, Optional, Tuple

import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional; verbose progress just becomes plain prints
    tqdm = None


def _face_normals_and_d(vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    e1 = v1 - v0
    e2 = v2 - v0
    n = np.cross(e1, e2)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    norm = np.where(norm < 1e-20, 1.0, norm)
    n = n / norm
    d = -np.einsum("ij,ij->i", n, v0)
    return n, d


def _build_vert2face_csr(faces: np.ndarray, num_vertices: int) -> Tuple[np.ndarray, np.ndarray]:
    num_faces = faces.shape[0]
    face_idx_rep = np.repeat(np.arange(num_faces), 3)
    vid_rep = faces.reshape(-1)
    order = np.argsort(vid_rep)  # order among faces of the same vertex is irrelevant -> no need for 'stable'
    sorted_faceidx = face_idx_rep[order]
    offsets = np.searchsorted(vid_rep[order], np.arange(num_vertices + 1))
    return sorted_faceidx, offsets


# Flat upper-triangle layout of the per-vertex QEM: symmetric 3x3 "A" (6
# channels) + 3x1 "b" + scalar "c", accumulated with bincount instead of a
# dense (V, 3, 3) np.add.at.
_QEM_CHANNELS = ("xx", "xy", "xz", "yy", "yz", "zz", "bx", "by", "bz", "c")


def _compute_qem(vertices: np.ndarray, faces: np.ndarray, num_vertices: int) -> Dict[str, np.ndarray]:
    n, d = _face_normals_and_d(vertices, faces)
    vid_rep = faces.reshape(-1)
    n_rep = np.repeat(n, 3, axis=0)
    d_rep = np.repeat(d, 3)
    nx, ny, nz = n_rep[:, 0], n_rep[:, 1], n_rep[:, 2]

    weights = {
        "xx": nx * nx, "xy": nx * ny, "xz": nx * nz,
        "yy": ny * ny, "yz": ny * nz, "zz": nz * nz,
        "bx": nx * d_rep, "by": ny * d_rep, "bz": nz * d_rep,
        "c": d_rep * d_rep,
    }
    return {k: np.bincount(vid_rep, weights=w, minlength=num_vertices) for k, w in weights.items()}


def _eval_qem_cost(qem: Dict[str, np.ndarray], e0: np.ndarray, e1: np.ndarray, v: np.ndarray) -> np.ndarray:
    """v^T A v + 2 b.v + c for the summed quadric of edge endpoints e0, e1."""
    xx = qem["xx"][e0] + qem["xx"][e1]
    xy = qem["xy"][e0] + qem["xy"][e1]
    xz = qem["xz"][e0] + qem["xz"][e1]
    yy = qem["yy"][e0] + qem["yy"][e1]
    yz = qem["yz"][e0] + qem["yz"][e1]
    zz = qem["zz"][e0] + qem["zz"][e1]
    bx = qem["bx"][e0] + qem["bx"][e1]
    by = qem["by"][e0] + qem["by"][e1]
    bz = qem["bz"][e0] + qem["bz"][e1]
    c = qem["c"][e0] + qem["c"][e1]

    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    quad = xx * vx * vx + yy * vy * vy + zz * vz * vz + 2.0 * (xy * vx * vy + xz * vx * vz + yz * vy * vz)
    lin = bx * vx + by * vy + bz * vz
    return quad + 2.0 * lin + c


def _pack_unique_edges(faces: np.ndarray, num_vertices: int) -> Tuple[np.ndarray, np.ndarray]:
    """Faster equivalent of `np.unique(sorted_edges, axis=0, return_counts=True)`:
    pack each (v0, v1) pair into a single int64 key and run 1-D `np.unique`,
    which uses a much faster code path than the generic 2-D/axis=0 case."""
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    lo = np.minimum(e[:, 0], e[:, 1]).astype(np.int64)
    hi = np.maximum(e[:, 0], e[:, 1]).astype(np.int64)
    key = lo * num_vertices + hi
    uniq_key, counts = np.unique(key, return_counts=True)
    uniq_edges = np.stack([uniq_key // num_vertices, uniq_key % num_vertices], axis=1)
    return uniq_edges, counts


def _boundary_vertex_mask(uniq_edges: np.ndarray, counts: np.ndarray, num_vertices: int) -> np.ndarray:
    is_boundary = np.zeros(num_vertices, dtype=bool)
    boundary_edges = uniq_edges[counts == 1]
    is_boundary[boundary_edges[:, 0]] = True
    is_boundary[boundary_edges[:, 1]] = True
    return is_boundary


def _expand_incident(vert_ids: np.ndarray, v2f_list: np.ndarray, v2f_offsets: np.ndarray):
    """For each id in `vert_ids` (indexed 0..len(vert_ids)-1), expand to
    (owner_index, face_id) pairs for every face incident to that vertex."""
    counts = v2f_offsets[vert_ids + 1] - v2f_offsets[vert_ids]
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    owner_rep = np.repeat(np.arange(vert_ids.shape[0]), counts)
    starts = v2f_offsets[vert_ids]
    cum = np.cumsum(counts) - counts
    pos_within = np.arange(total) - np.repeat(cum, counts)
    flat_idx = np.repeat(starts, counts) + pos_within
    fids = v2f_list[flat_idx]
    return owner_rep, fids


def _groupby_min(group_ids: np.ndarray, values: np.ndarray, num_groups: int) -> np.ndarray:
    """result[g] = min(values[i] for i where group_ids[i] == g), else +inf.
    A sort + `np.minimum.reduceat` "groupby", vectorized in C -- unlike
    `np.minimum.at`, which is a slow unbuffered per-element loop."""
    result = np.full(num_groups, np.inf)
    if group_ids.size == 0:
        return result
    order = np.argsort(group_ids)
    sorted_groups = group_ids[order]
    sorted_vals = values[order]
    uniq_groups, start_idx = np.unique(sorted_groups, return_index=True)
    result[uniq_groups] = np.minimum.reduceat(sorted_vals, start_idx)
    return result


def _incident_triangle_metrics(
    vertices: np.ndarray,
    faces: np.ndarray,
    owner_idx: np.ndarray,
    fids: np.ndarray,
    keep: np.ndarray,
    other: np.ndarray,
    v_new: np.ndarray,
):
    """For each (owner_idx, fid) pair -- the triangles incident to `keep`
    that survive the collapse -- returns (owner_idx, skinny_term, flipped),
    restricted to the triangles not sharing the other endpoint (which are
    the ones being removed by the collapse, not reshaped by it)."""
    if fids.shape[0] == 0:
        empty_f = np.empty(0, dtype=np.float64)
        empty_i = np.empty(0, dtype=np.int64)
        empty_b = np.empty(0, dtype=bool)
        return empty_i, empty_f, empty_b

    f_vids = faces[fids]
    keep_v = keep[owner_idx]
    other_v = other[owner_idx]
    v_new_e = v_new[owner_idx]

    contains_other = (f_vids[:, 0] == other_v) | (f_vids[:, 1] == other_v) | (f_vids[:, 2] == other_v)
    keep_mask = ~contains_other
    if not keep_mask.any():
        empty_f = np.empty(0, dtype=np.float64)
        empty_i = np.empty(0, dtype=np.int64)
        empty_b = np.empty(0, dtype=bool)
        return empty_i, empty_f, empty_b

    owner_idx = owner_idx[keep_mask]
    f_vids = f_vids[keep_mask]
    keep_v = keep_v[keep_mask]
    v_new_e = v_new_e[keep_mask]

    a = vertices[f_vids[:, 0]]
    bpt = vertices[f_vids[:, 1]]
    cpt = vertices[f_vids[:, 2]]

    na = np.where((f_vids[:, 0] == keep_v)[:, None], v_new_e, a)
    nb = np.where((f_vids[:, 1] == keep_v)[:, None], v_new_e, bpt)
    nc = np.where((f_vids[:, 2] == keep_v)[:, None], v_new_e, cpt)

    old_e1v = bpt - a
    old_e2v = cpt - a
    old_normal = np.cross(old_e1v, old_e2v)

    new_e1v = nb - na
    new_e2v = nc - na
    new_normal = np.cross(new_e1v, new_e2v)
    new_area = 0.5 * np.linalg.norm(new_normal, axis=1)

    dotp = np.einsum("ij,ij->i", old_normal, new_normal)
    flipped = dotp < 0.0

    new_e0v = nc - nb
    denom = np.sum(new_e0v ** 2, axis=1) + np.sum(new_e1v ** 2, axis=1) + np.sum(new_e2v ** 2, axis=1)
    denom = np.maximum(denom, 1e-12)
    shape_metric = 4.0 * np.sqrt(3.0) * new_area / denom
    term = 1.0 - np.clip(shape_metric, 0.0, 1.0)

    return owner_idx, term, flipped


def simplify_step(
    vertices: np.ndarray,
    faces: np.ndarray,
    lambda_edge_length: float,
    lambda_skinny: float,
    threshold: float,
    face_frozen: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """One batched parallel edge-collapse pass. Returns (new_vertices, new_faces, new_face_frozen).

    `face_frozen` is an optional [F] bool mask: True marks a face as
    untouchable (e.g. because it carries a texture). Any vertex referenced
    by a frozen face is locked -- it is never moved and never collapsed
    away -- which guarantees frozen faces come out byte-identical (same
    vertex positions, same winding), only their vertex indices get
    renumbered by the final compaction pass. `new_face_frozen` mirrors
    `new_faces` and can be threaded into the next call.
    """
    num_vertices = vertices.shape[0]
    num_faces = faces.shape[0]

    uniq_edges, edge_counts = _pack_unique_edges(faces, num_vertices)
    num_edges = uniq_edges.shape[0]
    if num_edges == 0:
        return vertices, faces, face_frozen

    is_boundary = _boundary_vertex_mask(uniq_edges, edge_counts, num_vertices)
    v2f_list, v2f_offsets = _build_vert2face_csr(faces, num_vertices)
    qem = _compute_qem(vertices, faces, num_vertices)

    e0 = uniq_edges[:, 0]
    e1 = uniq_edges[:, 1]
    v0 = vertices[e0]
    v1 = vertices[e1]

    b0 = is_boundary[e0]
    b1 = is_boundary[e1]
    w0 = np.where(b0 & ~b1, 1.0, np.where(~b0 & b1, 0.0, 0.5)).astype(vertices.dtype)
    v_new = v0 * w0[:, None] + v1 * (1.0 - w0)[:, None]

    qem_cost = _eval_qem_cost(qem, e0, e1, v_new)
    edge_len2 = np.sum((v1 - v0) ** 2, axis=1)
    cost = qem_cost + lambda_edge_length * edge_len2

    # An edge with BOTH endpoints on the boundary lies along the boundary
    # curve itself -- collapsing it blends the two boundary vertices to
    # their midpoint (w0 = 0.5 above), moving the boundary. That's fine for
    # a genuinely open mesh, but when the "boundary" only exists because the
    # caller passed in an isolated sub-mesh (e.g. just the untextured
    # primitives of a GLB, split off from the textured ones), that midpoint
    # move detaches the seam from the untouched geometry on the other side
    # -- a visible crack/shift at the material boundary. Locking
    # boundary-to-boundary edges (only ever collapsing an interior vertex
    # INTO a boundary vertex, which keeps it exactly where it was) avoids
    # that.
    cost = np.where(b0 & b1, np.inf, cost)

    # incident-triangle metrics (skinny-shape penalty + normal-flip veto),
    # computed once per endpoint side and shared with propagation/selection below
    owner0, fids0 = _expand_incident(e0, v2f_list, v2f_offsets)
    owner1, fids1 = _expand_incident(e1, v2f_list, v2f_offsets)

    o0, term0, flip0 = _incident_triangle_metrics(vertices, faces, owner0, fids0, e0, e1, v_new)
    o1, term1, flip1 = _incident_triangle_metrics(vertices, faces, owner1, fids1, e1, e0, v_new)

    skinny_sum = np.bincount(o0, weights=term0, minlength=num_edges) + np.bincount(
        o1, weights=term1, minlength=num_edges
    )
    tri_count = np.bincount(o0, minlength=num_edges) + np.bincount(o1, minlength=num_edges)
    invalid = (np.bincount(o0, weights=flip0.astype(np.float64), minlength=num_edges) > 0) | (
        np.bincount(o1, weights=flip1.astype(np.float64), minlength=num_edges) > 0
    )

    tri_count_safe = np.maximum(tri_count, 1)
    skinny_cost = skinny_sum / tri_count_safe
    cost = cost + lambda_skinny * skinny_cost * edge_len2
    cost = np.where(invalid, np.inf, cost)

    if face_frozen is not None and face_frozen.any():
        locked_vertex = np.zeros(num_vertices, dtype=bool)
        locked_vertex[faces[face_frozen].reshape(-1)] = True
        locked_edge = locked_vertex[e0] | locked_vertex[e1]
        cost = np.where(locked_edge, np.inf, cost)

    # propagate: each face remembers the min cost among all edges touching it
    all_owner = np.concatenate([owner0, owner1])
    all_fids = np.concatenate([fids0, fids1])
    all_cost = cost[all_owner]
    propagated = _groupby_min(all_fids, all_cost, num_faces)

    # select: an edge collapses only if it's the local-min edge for every
    # face touching either endpoint (bincount-based "AND of a vote per edge")
    ok = propagated[all_fids] == all_cost
    violations = np.bincount(all_owner, weights=(~ok).astype(np.float64), minlength=num_edges)
    selected = (violations == 0) & np.isfinite(cost) & (cost <= threshold)

    if not selected.any():
        return vertices, faces, face_frozen

    e0_sel = e0[selected]
    e1_sel = e1[selected]
    v_new_sel = v_new[selected]

    new_vertices = vertices.copy()
    new_vertices[e0_sel] = v_new_sel

    remap = np.arange(num_vertices)
    remap[e1_sel] = e0_sel

    remapped_faces = remap[faces]
    degenerate = (
        (remapped_faces[:, 0] == remapped_faces[:, 1])
        | (remapped_faces[:, 1] == remapped_faces[:, 2])
        | (remapped_faces[:, 0] == remapped_faces[:, 2])
    )
    kept_faces = remapped_faces[~degenerate]
    new_face_frozen = face_frozen[~degenerate] if face_frozen is not None else None

    referenced = np.zeros(num_vertices, dtype=bool)
    if kept_faces.size:
        referenced[kept_faces.reshape(-1)] = True
    new_ids = -np.ones(num_vertices, dtype=np.int64)
    new_ids[referenced] = np.arange(int(referenced.sum()))

    compact_faces = new_ids[kept_faces]
    compact_vertices = new_vertices[referenced]
    return compact_vertices, compact_faces, new_face_frozen


def simplify(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_num_faces: int,
    verbose: bool = False,
    options: Optional[Dict[str, float]] = None,
    face_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Pure-Python/NumPy drop-in for `cumesh.CuMesh().simplify(target_num_faces, ...)`.

    Args:
        vertices: [V, 3] float array of vertex positions.
        faces: [F, 3] int array of triangle vertex indices.
        target_num_faces: stop once the face count is at or below this.
        verbose: print/track progress.
        options: same keys as CuMesh — 'thresh' (initial collapse-cost
            threshold, default 1e-8), 'lambda_edge_length' (default 1e-2),
            'lambda_skinny' (default 1e-3).
        face_mask: optional [F] bool array. True = face may be simplified,
            False = face is frozen (e.g. it carries a texture and must keep
            its exact vertex positions/UV mapping). Any vertex touched by a
            frozen face is locked and never moved or removed, so frozen
            faces always come out geometrically unchanged. If the target
            face count is lower than the number of frozen faces, the loop
            stops early once no more editable faces can be collapsed
            (the safety valve below), rather than hang trying to reach an
            unreachable target.

    Returns:
        (new_vertices, new_faces, new_face_mask) as float64/int64/bool
        NumPy arrays. `new_face_mask` is None if `face_mask` was not given.
    """
    assert isinstance(target_num_faces, int) and target_num_faces > 0, "target_num_faces must be a positive integer"

    options = options or {}
    thresh = options.get("thresh", 1e-8)
    lambda_edge_length = options.get("lambda_edge_length", 1e-2)
    lambda_skinny = options.get("lambda_skinny", 1e-3)

    # float32 roughly halves the memory bandwidth of the per-triangle vector
    # math in `_incident_triangle_metrics`, which dominates wall-clock time
    # on large meshes; glTF stores positions as float32 anyway.
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    face_frozen = None if face_mask is None else ~np.asarray(face_mask, dtype=bool)

    num_face = faces.shape[0]
    if num_face <= target_num_faces:
        return vertices, faces, (None if face_frozen is None else ~face_frozen)

    pbar = None
    if verbose and tqdm is not None:
        pbar = tqdm(total=num_face - target_num_faces, desc="Simplifying")

    # Safety valve: CuMesh's loop assumes the threshold escalation will
    # eventually free up progress; cap iterations/threshold growth so a
    # mesh that gets stuck (e.g. boundary-locked geometry, or an
    # unreachable target because too many faces are frozen) can't hang.
    max_thresh = max(thresh * 1e12, 1.0)
    for _ in range(10000):
        if verbose:
            if pbar is not None:
                pbar.set_description(f"Simplifying [thresh={thresh:.2e}]")
            else:
                print(f"Simplifying [thresh={thresh:.2e}] faces={num_face}")

        vertices, faces, face_frozen = simplify_step(
            vertices, faces, lambda_edge_length, lambda_skinny, thresh, face_frozen
        )
        new_num_face = faces.shape[0]

        if pbar is not None:
            pbar.update(min(num_face - new_num_face, pbar.total - pbar.n))

        if new_num_face <= target_num_faces:
            break

        del_num_face = num_face - new_num_face
        frac = (del_num_face / num_face) if num_face > 0 else 0.0
        if frac < 1e-2:
            # Escalating by a flat x10 needs dozens of passes on meshes
            # where cost is fairly uniform (each pass has fixed O(F) recompute
            # overhead, so pass *count* -- not per-pass cost -- dominates
            # wall-clock time on large meshes). Jump further when progress is
            # smaller: aim for the next pass to clear roughly a 5% target
            # rather than always creeping up by the same factor.
            factor = 100.0 if frac <= 0.0 else min(1e6, max(10.0, 0.05 / frac))
            thresh *= factor
        if thresh > max_thresh:
            break
        num_face = new_num_face

    if pbar is not None:
        pbar.close()

    new_face_mask = None if face_frozen is None else ~face_frozen
    return vertices, faces, new_face_mask
