"""Run the pure-Python CuMesh.simplify() port on a real GLB mesh.

By default every primitive in the file is merged and simplified. Pass
--untextured-only to instead simplify only the primitives whose material
carries no texture (baseColorTexture / metallicRoughnessTexture /
normalTexture / occlusionTexture / emissiveTexture) -- primitives that do
have a texture are copied through byte-identical (same vertex/UV/normal
data, same material), so decimation cannot warp their texture mapping.

Usage:
    python run_on_glb.py INPUT.glb [--ratio 0.5] [--target N]
    python run_on_glb.py INPUT.glb --untextured-only [--ratio 0.5]
"""

import argparse
import time

import numpy as np
import pygltflib

from simplify import simplify

_COMPONENT_TYPE_DTYPE = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}

_TYPE_NUM_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT4": 16,
}


def read_accessor(gltf: pygltflib.GLTF2, blob: bytes, accessor_index: int) -> np.ndarray:
    accessor = gltf.accessors[accessor_index]
    buffer_view = gltf.bufferViews[accessor.bufferView]
    dtype = _COMPONENT_TYPE_DTYPE[accessor.componentType]
    num_components = _TYPE_NUM_COMPONENTS[accessor.type]

    start = buffer_view.byteOffset or 0
    start += accessor.byteOffset or 0
    itemsize = np.dtype(dtype).itemsize * num_components
    stride = buffer_view.byteStride or itemsize
    count = accessor.count

    if stride == itemsize:
        # tightly packed: one vectorized read instead of a per-element loop
        flat = np.frombuffer(blob, dtype=dtype, count=count * num_components, offset=start)
        out = flat.reshape(count, num_components)
    else:
        # interleaved attributes: gather each element's bytes via as_strided-free
        # (still vectorized) fancy indexing instead of a Python-level loop
        byte_view = np.frombuffer(blob, dtype=np.uint8, offset=start, count=stride * (count - 1) + itemsize)
        row_starts = np.arange(count) * stride
        col_offsets = np.arange(itemsize)
        gathered = byte_view[row_starts[:, None] + col_offsets[None, :]]
        out = gathered.copy().view(dtype).reshape(count, num_components)
    return out.reshape(count) if num_components == 1 else out


def _mat4_from_trs(t, r, s) -> np.ndarray:
    """4x4 matrix M such that world = M @ [x, y, z, 1] (standard row/col
    matrix-vector convention), built from glTF's translation/rotation
    (quaternion xyzw)/scale node fields."""
    x, y, z, w = r
    x2, y2, z2 = x + x, y + y, z + z
    xx, xy, xz = x * x2, x * y2, x * z2
    yy, yz, zz = y * y2, y * z2, z * z2
    wx, wy, wz = w * x2, w * y2, w * z2
    sx, sy, sz = s
    m = np.eye(4, dtype=np.float64)
    m[0, 0] = (1 - (yy + zz)) * sx
    m[1, 0] = (xy + wz) * sx
    m[2, 0] = (xz - wy) * sx
    m[0, 1] = (xy - wz) * sy
    m[1, 1] = (1 - (xx + zz)) * sy
    m[2, 1] = (yz + wx) * sy
    m[0, 2] = (xz + wy) * sz
    m[1, 2] = (yz - wx) * sz
    m[2, 2] = (1 - (xx + yy)) * sz
    m[0, 3], m[1, 3], m[2, 3] = t
    return m


def _node_local_matrix(node) -> np.ndarray:
    if node.matrix is not None:
        # glTF stores node.matrix as 16 floats in COLUMN-MAJOR order.
        return np.array(node.matrix, dtype=np.float64).reshape(4, 4, order="F")
    t = node.translation if node.translation is not None else [0.0, 0.0, 0.0]
    r = node.rotation if node.rotation is not None else [0.0, 0.0, 0.0, 1.0]
    s = node.scale if node.scale is not None else [1.0, 1.0, 1.0]
    return _mat4_from_trs(t, r, s)


def compute_world_transforms(gltf: pygltflib.GLTF2) -> list:
    """Returns one world-space 4x4 matrix per node index, composed through
    the scene graph from whichever scene roots reach it. Nodes not reachable
    from any scene root (orphaned) get the identity matrix."""
    nodes = gltf.nodes or []
    world = [None] * len(nodes)
    visited = set()

    def visit(idx, parent):
        if idx in visited:
            return
        visited.add(idx)
        node = nodes[idx]
        world[idx] = parent @ _node_local_matrix(node)
        for child in node.children or []:
            visit(child, world[idx])

    scenes = gltf.scenes or []
    scene_idx = gltf.scene if gltf.scene is not None else 0
    roots = scenes[scene_idx].nodes if scenes and scenes[scene_idx].nodes else list(range(len(nodes)))
    for r in roots:
        visit(r, np.eye(4))
    for i in range(len(nodes)):
        if world[i] is None:
            world[i] = np.eye(4)
    return world


def bake_world_positions(positions: np.ndarray, world_matrix: np.ndarray) -> np.ndarray:
    """Applies `world_matrix` to local-space positions [N,3] -> world-space [N,3].
    Use before merging primitives from different nodes so the merged buffer
    is in one consistent space (see load_primitives' docstring)."""
    homogeneous = np.concatenate([positions, np.ones((positions.shape[0], 1), dtype=positions.dtype)], axis=1)
    return (world_matrix @ homogeneous.T).T[:, :3]


def material_has_texture(gltf: pygltflib.GLTF2, material_index) -> bool:
    """True if the material references any texture slot at all."""
    if material_index is None:
        return False
    mat = gltf.materials[material_index]
    pbr = mat.pbrMetallicRoughness
    if pbr is not None and (pbr.baseColorTexture is not None or pbr.metallicRoughnessTexture is not None):
        return True
    if mat.normalTexture is not None or mat.occlusionTexture is not None or mat.emissiveTexture is not None:
        return True
    return False


def load_primitives(gltf: pygltflib.GLTF2, blob: bytes):
    """Returns a list of dicts, one per (mesh, primitive), each holding the
    primitive object itself (for later identity-based removal), its
    LOCAL-space positions/indices, whether its material has a texture, and
    the world-space transform of whichever node instances its mesh.

    IMPORTANT: `positions` are NOT pre-transformed. Merging primitives from
    different nodes (e.g. to simplify several untextured meshes together)
    must bake `world_matrix` into each primitive's vertices first (see
    `bake_world_positions`), or the merged result will be positioned
    according to whichever single node it's later attached to, silently
    mispositioning every other primitive that had a different transform --
    which is exactly what a real multi-part CAD assembly (many separate
    meshes, each on its own node/transform) has.

    `instance_count` (number of nodes referencing this mesh) tells the
    caller whether it's safe to bake this primitive to world space and
    merge it with others: if `instance_count > 1` (instancing -- the same
    geometry referenced by several nodes at different transforms, common
    for repeated parts like bolts/brackets in a CAD assembly), merging its
    world-space-baked geometry into a blob attached to ONE new node would
    collapse every instance down to a single occurrence, silently deleting
    the rest. Such primitives must instead be simplified in their own local
    space and have their geometry replaced in place (see
    `replace_primitive_geometry`), leaving every referencing node untouched
    and still instancing the (now-simplified) mesh correctly.
    """
    world_transforms = compute_world_transforms(gltf)
    nodes = gltf.nodes or []

    out = []
    for mesh_index, mesh in enumerate(gltf.meshes):
        referencing = [i for i, n in enumerate(nodes) if n.mesh == mesh_index]
        instance_count = len(referencing)
        world_matrix = world_transforms[referencing[0]] if referencing else np.eye(4)

        for prim in mesh.primitives:
            positions = read_accessor(gltf, blob, prim.attributes.POSITION).astype(np.float64)
            indices = read_accessor(gltf, blob, prim.indices).astype(np.int64).reshape(-1, 3)
            out.append(
                {
                    "mesh": mesh,
                    "mesh_index": mesh_index,
                    "primitive": prim,
                    "positions": positions,
                    "indices": indices,
                    "textured": material_has_texture(gltf, prim.material),
                    "world_matrix": world_matrix,
                    "instance_count": instance_count,
                }
            )
    return out


def total_area(v: np.ndarray, f: np.ndarray) -> float:
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())


def _append_geometry_accessors(gltf: pygltflib.GLTF2, blob: bytes, vertices: np.ndarray, faces: np.ndarray):
    """Appends `vertices`/`faces` as new accessors/bufferViews, tacking the
    binary data onto the end of `blob` (so every EXISTING accessor's
    byteOffset stays valid). Returns (new_blob, pos_accessor_idx, idx_accessor_idx)."""
    vertices = vertices.astype(np.float32)
    faces = faces.astype(np.uint32)

    positions_blob = vertices.tobytes()
    indices_blob = faces.tobytes()
    base_offset = len(blob)
    combined_blob = blob + positions_blob + indices_blob

    pos_accessor_idx = len(gltf.accessors)
    idx_accessor_idx = pos_accessor_idx + 1
    pos_bv_idx = len(gltf.bufferViews)
    idx_bv_idx = pos_bv_idx + 1

    gltf.bufferViews.append(
        pygltflib.BufferView(
            buffer=0, byteOffset=base_offset, byteLength=len(positions_blob), target=pygltflib.ARRAY_BUFFER
        )
    )
    gltf.bufferViews.append(
        pygltflib.BufferView(
            buffer=0,
            byteOffset=base_offset + len(positions_blob),
            byteLength=len(indices_blob),
            target=pygltflib.ELEMENT_ARRAY_BUFFER,
        )
    )
    gltf.accessors.append(
        pygltflib.Accessor(
            bufferView=pos_bv_idx,
            componentType=pygltflib.FLOAT,
            count=vertices.shape[0],
            type=pygltflib.VEC3,
            min=vertices.min(axis=0).tolist(),
            max=vertices.max(axis=0).tolist(),
        )
    )
    gltf.accessors.append(
        pygltflib.Accessor(
            bufferView=idx_bv_idx,
            componentType=pygltflib.UNSIGNED_INT,
            count=int(faces.size),
            type=pygltflib.SCALAR,
        )
    )
    return combined_blob, pos_accessor_idx, idx_accessor_idx


def append_simplified_primitive(
    gltf: pygltflib.GLTF2,
    blob: bytes,
    vertices: np.ndarray,
    faces: np.ndarray,
    host_mesh,
    material_index=None,
):
    """Appends `vertices`/`faces` as a new position-only primitive on
    `host_mesh`. Assigns `material_index` (the ORIGINAL material this
    geometry group came from, e.g. a plain base-color/metallic-roughness
    factor with no texture) so the simplified result isn't silently
    rendered with the glTF default material -- pass None only for geometry
    that genuinely had no material to begin with. Returns the new full blob."""
    combined_blob, pos_accessor_idx, idx_accessor_idx = _append_geometry_accessors(gltf, blob, vertices, faces)
    host_mesh.primitives.append(
        pygltflib.Primitive(
            attributes=pygltflib.Attributes(POSITION=pos_accessor_idx),
            indices=idx_accessor_idx,
            material=material_index,
            mode=pygltflib.TRIANGLES,
        )
    )
    return combined_blob


def replace_primitive_geometry(gltf: pygltflib.GLTF2, blob: bytes, primitive, vertices: np.ndarray, faces: np.ndarray):
    """For an INSTANCED primitive (see load_primitives' `instance_count`):
    swaps `primitive`'s POSITION/indices accessors for new ones holding the
    (locally, not world-space, simplified) geometry, IN PLACE on the same
    primitive object -- same mesh, same material, every node instancing it
    keeps instancing it. Any other attribute (NORMAL, TEXCOORD_0, ...)
    referenced the OLD vertex count/order and no longer matches the new
    geometry, so it's dropped rather than left stale. Returns the new blob."""
    combined_blob, pos_accessor_idx, idx_accessor_idx = _append_geometry_accessors(gltf, blob, vertices, faces)
    primitive.attributes = pygltflib.Attributes(POSITION=pos_accessor_idx)
    primitive.indices = idx_accessor_idx
    return combined_blob


def run(args):
    gltf = pygltflib.GLTF2().load(args.input)
    blob = gltf.binary_blob()
    prims = load_primitives(gltf, blob)

    if args.untextured_only:
        target_prims = [p for p in prims if not p["textured"]]
        frozen_prims = [p for p in prims if p["textured"]]
        print(
            f"primitives: {len(prims)} total, "
            f"{len(frozen_prims)} textured (kept as-is), "
            f"{len(target_prims)} untextured (simplify target)"
        )
    else:
        target_prims = prims
        frozen_prims = []

    if not target_prims:
        print("Nothing to simplify (no untextured primitives found); model left unchanged.")
        if args.out:
            gltf.save(args.out)
            print(f"wrote {args.out}")
        return

    # Instanced primitives (the same mesh referenced by several nodes at
    # different transforms -- common for repeated parts like bolts in a
    # CAD assembly) must NOT go through the merge-by-material path below:
    # baking to world space and merging into geometry attached to ONE new
    # node collapses every instance down to a single occurrence, silently
    # deleting the rest. They're simplified individually further down, in
    # their own local space, with geometry replaced in place instead.
    instanced_prims = [p for p in target_prims if p["instance_count"] > 1]
    mergeable_prims = [p for p in target_prims if p["instance_count"] <= 1]
    if instanced_prims:
        print(f"{len(instanced_prims)} primitive(s) are instanced (simplified individually, in place)")

    # Group the remaining (non-instanced) target primitives by material: a
    # glTF primitive can only reference one material, so primitives with
    # DIFFERENT materials can't be merged into a single output primitive
    # without losing one of them. Even an "untextured" primitive can have a
    # real material (base color/metallic/roughness factors, doubleSided,
    # alphaMode -- a shader in every practical sense), so each group keeps
    # and reuses its original material index rather than the merged
    # geometry coming out material-less.
    groups: dict = {}
    for p in mergeable_prims:
        key = p["primitive"].material  # None is a valid key (no material)
        groups.setdefault(key, []).append(p)
    print(f"grouped into {len(groups)} material group(s) for simplification")

    # A material with a texture requires TEXCOORD_0 on any primitive that
    # uses it (glTF spec); we don't carry UVs through the geometry pipeline,
    # so simplified/merged geometry can never validly keep a textured
    # material. This only matters without --untextured-only (which never
    # includes textured primitives among target_prims to begin with) --
    # here we just fall back to no material for those groups, same as
    # before this fix, rather than emit invalid glTF.
    def output_material_index(material_index):
        if material_index is not None and material_has_texture(gltf, material_index):
            return None
        return material_index

    out_path = args.out or args.input.replace(".glb", "_simplified.glb")

    group_results = []  # list of (vertices, faces, material_index)
    instance_results = []  # list of (primitive, vertices, faces)
    total_before = total_after = 0
    area_before_sum = area_after_sum = 0.0
    degenerate_sum = 0
    t0 = time.time()

    def simplify_one(label, vertices, faces):
        nonlocal total_before, total_after, area_before_sum, area_after_sum, degenerate_sum
        target = args.target if args.target is not None else max(1, int(faces.shape[0] * args.ratio))
        print(f"{label}: V={vertices.shape[0]} F={faces.shape[0]} -> target {target}")

        area_before = total_area(vertices, faces)
        new_vertices, new_faces, _ = simplify(vertices, faces, target, verbose=True)
        area_after = total_area(new_vertices, new_faces)
        degenerate = (
            (new_faces[:, 0] == new_faces[:, 1])
            | (new_faces[:, 1] == new_faces[:, 2])
            | (new_faces[:, 0] == new_faces[:, 2])
        )
        print(f"  -> V={new_vertices.shape[0]} F={new_faces.shape[0]} degenerate={int(degenerate.sum())}")

        total_before += faces.shape[0]
        total_after += new_faces.shape[0]
        area_before_sum += area_before
        area_after_sum += area_after
        degenerate_sum += int(degenerate.sum())
        return new_vertices, new_faces

    # Instanced primitives: simplify each one on its OWN local-space
    # geometry (not merged with anything, not baked to world space), then
    # swap its accessors in place -- every node instancing it keeps
    # instancing it, now at the simplified geometry.
    for p in instanced_prims:
        label = f"instanced [mesh {p['mesh_index']}, x{p['instance_count']}]"
        new_vertices, new_faces = simplify_one(label, p["positions"], p["indices"])
        instance_results.append((p["primitive"], new_vertices, new_faces))

    for material_index, group_prims in groups.items():
        all_faces = []
        all_verts = []
        offset = 0
        for p in group_prims:
            all_faces.append(p["indices"] + offset)
            all_verts.append(bake_world_positions(p["positions"], p["world_matrix"]))
            offset += p["positions"].shape[0]
        vertices = np.concatenate(all_verts, axis=0)
        faces = np.concatenate(all_faces, axis=0)

        label = "group [no material]" if material_index is None else f"group [material {material_index}]"
        new_vertices, new_faces = simplify_one(label, vertices, faces)
        group_results.append((new_vertices, new_faces, output_material_index(material_index)))

    dt = time.time() - t0
    print(f"simplified {len(groups)} group(s) in {dt:.1f}s: F {total_before} -> {total_after}")
    print(f"area before={area_before_sum:.4f} after={area_after_sum:.4f}")
    print(f"degenerate faces: {degenerate_sum}")

    # Mutate the ORIGINAL file in place (both with and without
    # --untextured-only): drop only the mergeable (non-instanced)
    # primitives that got merged into new_mesh below; instanced primitives
    # keep their original primitive slot on their original mesh, with only
    # their geometry swapped (replace_primitive_geometry, called below), so
    # every node that instances them keeps doing so. Reusing some arbitrary
    # existing node's transform for the merged geometry (instead of a fresh
    # identity-transform node) would silently re-apply that node's own
    # transform on top of already-world-space vertices.
    mergeable_prim_objs = {id(p["primitive"]) for p in mergeable_prims}
    for mesh in gltf.meshes:
        mesh.primitives = [pr for pr in mesh.primitives if id(pr) not in mergeable_prim_objs]

    new_blob = blob
    for primitive, new_vertices, new_faces in instance_results:
        new_blob = replace_primitive_geometry(gltf, new_blob, primitive, new_vertices, new_faces)

    if group_results:
        new_mesh = pygltflib.Mesh(primitives=[])
        new_mesh_index = len(gltf.meshes)
        gltf.meshes.append(new_mesh)

        if gltf.nodes is None:
            gltf.nodes = []
        new_node_index = len(gltf.nodes)
        gltf.nodes.append(pygltflib.Node(mesh=new_mesh_index))

        if not gltf.scenes:
            gltf.scenes = [pygltflib.Scene(nodes=[])]
        scene_idx = gltf.scene if gltf.scene is not None else 0
        if gltf.scenes[scene_idx].nodes is None:
            gltf.scenes[scene_idx].nodes = []
        gltf.scenes[scene_idx].nodes.append(new_node_index)
        gltf.scene = scene_idx

        for new_vertices, new_faces, material_index in group_results:
            new_blob = append_simplified_primitive(
                gltf, new_blob, new_vertices, new_faces, new_mesh, material_index=material_index
            )

    gltf.buffers[0].byteLength = len(new_blob)
    gltf.set_binary_blob(new_blob)
    gltf.save(out_path)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--ratio", type=float, default=0.5, help="target face count as a fraction of the editable input")
    parser.add_argument("--target", type=int, default=None, help="explicit target face count (overrides --ratio)")
    parser.add_argument(
        "--untextured-only",
        action="store_true",
        help="only simplify primitives whose material has no texture; copy textured primitives through unchanged",
    )
    parser.add_argument("--out", default=None)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
