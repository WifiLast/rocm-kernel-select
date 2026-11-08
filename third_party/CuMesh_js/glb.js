// Minimal binary glTF (.glb) reader/writer for the browser. No dependencies.
// Mirrors the (Python) logic in ../CuMesh_py/run_on_glb.py: read every
// primitive's POSITION + indices, detect whether its material has any
// texture, and (after simplification) splice in a new position/index-only
// primitive while leaving textured primitives byte-identical.

const GLB_MAGIC = 0x46546c67; // "glTF"
const CHUNK_JSON = 0x4e4f534a; // "JSON"
const CHUNK_BIN = 0x004e4942; // "BIN\0"

const COMPONENT_TYPE = {
  BYTE: 5120,
  UNSIGNED_BYTE: 5121,
  SHORT: 5122,
  UNSIGNED_SHORT: 5123,
  UNSIGNED_INT: 5125,
  FLOAT: 5126,
};

const COMPONENT_CTOR = {
  [COMPONENT_TYPE.BYTE]: Int8Array,
  [COMPONENT_TYPE.UNSIGNED_BYTE]: Uint8Array,
  [COMPONENT_TYPE.SHORT]: Int16Array,
  [COMPONENT_TYPE.UNSIGNED_SHORT]: Uint16Array,
  [COMPONENT_TYPE.UNSIGNED_INT]: Uint32Array,
  [COMPONENT_TYPE.FLOAT]: Float32Array,
};

const TYPE_NUM_COMPONENTS = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 };

// Column-major 4x4 matrices throughout (glTF convention: m[col*4+row]).
const IDENTITY_MAT4 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

function mat4FromTRS(t, r, s) {
  const [x, y, z, w] = r;
  const x2 = x * 2, y2 = y * 2, z2 = z * 2;
  const xx = x * x2, xy = x * y2, xz = x * z2;
  const yy = y * y2, yz = y * z2, zz = z * z2;
  const wx = w * x2, wy = w * y2, wz = w * z2;
  const [sx, sy, sz] = s;
  return [
    (1 - (yy + zz)) * sx, (xy + wz) * sx, (xz - wy) * sx, 0,
    (xy - wz) * sy, (1 - (xx + zz)) * sy, (yz + wx) * sy, 0,
    (xz + wy) * sz, (yz - wx) * sz, (1 - (xx + yy)) * sz, 0,
    t[0], t[1], t[2], 1,
  ];
}

function nodeLocalMatrix(node) {
  if (node.matrix) return node.matrix;
  return mat4FromTRS(node.translation ?? [0, 0, 0], node.rotation ?? [0, 0, 0, 1], node.scale ?? [1, 1, 1]);
}

function multiplyMat4(a, b) {
  const out = new Array(16);
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      let sum = 0;
      for (let k = 0; k < 4; k++) sum += a[k * 4 + row] * b[col * 4 + k];
      out[col * 4 + row] = sum;
    }
  }
  return out;
}

export function transformPoint(m, x, y, z) {
  return [
    m[0] * x + m[4] * y + m[8] * z + m[12],
    m[1] * x + m[5] * y + m[9] * z + m[13],
    m[2] * x + m[6] * y + m[10] * z + m[14],
  ];
}

/** Returns one world-space (column-major 4x4) matrix per node index, composed
 * through the scene graph from whichever scene roots reach it. Nodes not
 * reachable from any scene root (orphaned) get the identity matrix. */
export function computeWorldTransforms(json) {
  const nodes = json.nodes ?? [];
  const world = new Array(nodes.length).fill(null);
  const visited = new Set();

  function visit(nodeIndex, parentMatrix) {
    if (visited.has(nodeIndex)) return; // avoid infinite loops on malformed cycles
    visited.add(nodeIndex);
    const node = nodes[nodeIndex];
    if (!node) return;
    const worldMatrix = multiplyMat4(parentMatrix, nodeLocalMatrix(node));
    world[nodeIndex] = worldMatrix;
    for (const child of node.children ?? []) visit(child, worldMatrix);
  }

  const scenes = json.scenes ?? [];
  const sceneIndex = json.scene ?? 0;
  const roots = scenes[sceneIndex]?.nodes ?? nodes.map((_, i) => i);
  for (const r of roots) visit(r, IDENTITY_MAT4);

  for (let i = 0; i < nodes.length; i++) if (world[i] == null) world[i] = IDENTITY_MAT4;
  return world;
}

/** Parses a .glb ArrayBuffer into { json, bin (ArrayBuffer) }. */
export function parseGLB(arrayBuffer) {
  const dv = new DataView(arrayBuffer);
  if (dv.getUint32(0, true) !== GLB_MAGIC) throw new Error("Not a .glb file (bad magic)");
  const totalLength = dv.getUint32(8, true);

  let offset = 12;
  let json = null;
  let bin = null;
  while (offset < totalLength) {
    const chunkLength = dv.getUint32(offset, true);
    const chunkType = dv.getUint32(offset + 4, true);
    const chunkStart = offset + 8;
    if (chunkType === CHUNK_JSON) {
      const text = new TextDecoder("utf-8").decode(new Uint8Array(arrayBuffer, chunkStart, chunkLength));
      json = JSON.parse(text);
    } else if (chunkType === CHUNK_BIN) {
      bin = arrayBuffer.slice(chunkStart, chunkStart + chunkLength);
    }
    offset = chunkStart + chunkLength;
  }
  if (!json) throw new Error("No JSON chunk found in .glb");
  return { json, bin: bin ?? new ArrayBuffer(0) };
}

/** Reads accessor `accessorIndex` from `json`/`bin` as a flat typed array
 * (count * numComponents elements), handling interleaved bufferViews. */
export function readAccessor(json, bin, accessorIndex) {
  const accessor = json.accessors[accessorIndex];
  const bufferView = json.bufferViews[accessor.bufferView];
  const Ctor = COMPONENT_CTOR[accessor.componentType];
  const numComponents = TYPE_NUM_COMPONENTS[accessor.type];
  const itemSize = Ctor.BYTES_PER_ELEMENT * numComponents;

  const start = (bufferView.byteOffset ?? 0) + (accessor.byteOffset ?? 0);
  const stride = bufferView.byteStride ?? itemSize;
  const count = accessor.count;

  const out = new Ctor(count * numComponents);
  if (stride === itemSize) {
    const src = new Ctor(bin, start, count * numComponents);
    out.set(src);
  } else {
    for (let i = 0; i < count; i++) {
      const rowStart = start + i * stride;
      const row = new Ctor(bin.slice(rowStart, rowStart + itemSize));
      out.set(row, i * numComponents);
    }
  }
  return out;
}

function materialHasTexture(json, materialIndex) {
  if (materialIndex == null || !json.materials) return false;
  const mat = json.materials[materialIndex];
  if (!mat) return false;
  const pbr = mat.pbrMetallicRoughness;
  if (pbr && (pbr.baseColorTexture || pbr.metallicRoughnessTexture)) return true;
  if (mat.normalTexture || mat.occlusionTexture || mat.emissiveTexture) return true;
  return false;
}

/**
 * Enumerates every (mesh, primitive) pair with its geometry, texture status,
 * the world-space transform of whichever node instances it, and how many
 * nodes instance it (`instanceCount`).
 *
 * IMPORTANT: `positions` are in the primitive's own LOCAL space (as stored
 * in the file) -- they are NOT pre-transformed. Merging primitives from
 * different nodes (e.g. to simplify several untextured meshes together)
 * must bake `worldMatrix` into each primitive's vertices first (see
 * `transformPoint`), or the merged result will be positioned according to
 * whichever single node it's later attached to, silently mispositioning
 * every other primitive that had a different transform.
 *
 * `instanceCount > 1` means this mesh is INSTANCED (the same geometry
 * referenced by several nodes at different transforms -- common for
 * repeated parts like bolts/brackets in a CAD assembly). Callers MUST NOT
 * merge an instanced primitive's geometry into a world-space blob attached
 * to one new node the way a single-instance primitive can be: doing so
 * collapses every instance down to the one node's position, silently
 * deleting the rest. Instanced primitives should instead be simplified in
 * their own local space and have their geometry replaced in place (see
 * `buildGLB`'s `inPlaceReplacements`), leaving every node that references
 * the mesh untouched and still instancing it correctly.
 */
export function loadPrimitives(json, bin) {
  const worldTransforms = computeWorldTransforms(json);
  const nodes = json.nodes ?? [];

  const out = [];
  (json.meshes ?? []).forEach((mesh, meshIndex) => {
    const referencingNodes = nodes.map((n, i) => (n.mesh === meshIndex ? i : -1)).filter((i) => i >= 0);
    const instanceCount = referencingNodes.length;
    const worldMatrix = instanceCount > 0 ? worldTransforms[referencingNodes[0]] : IDENTITY_MAT4;

    mesh.primitives.forEach((prim, primIndex) => {
      const positions = readAccessor(json, bin, prim.attributes.POSITION); // Float32Array, flat xyz, LOCAL space
      const rawIndices = readAccessor(json, bin, prim.indices);
      const indices = rawIndices instanceof Uint32Array ? rawIndices : Uint32Array.from(rawIndices);
      out.push({
        meshIndex,
        primIndex,
        primitive: prim,
        positions,
        indices,
        textured: materialHasTexture(json, prim.material),
        worldMatrix,
        instanceCount,
      });
    });
  });
  return out;
}

/** Applies `worldMatrix` to a flat local-space position array, returning a
 * new Float32Array in world space. Use before merging primitives from
 * different nodes so the merged buffer is in one consistent space. */
export function bakeWorldPositions(positions, worldMatrix) {
  const out = new Float32Array(positions.length);
  for (let i = 0; i < positions.length; i += 3) {
    const [x, y, z] = transformPoint(worldMatrix, positions[i], positions[i + 1], positions[i + 2]);
    out[i] = x; out[i + 1] = y; out[i + 2] = z;
  }
  return out;
}

function align4(n) {
  return (n + 3) & ~3;
}

/**
 * Builds a new .glb ArrayBuffer from `json`/`bin`, with the primitives at
 * `removePrimitiveRefs` (each `{meshIndex, primIndex}`) dropped and one new
 * mesh added -- holding one primitive per entry in `newPrimitives`, each
 * `{vertices, faces, materialIndex}` -- on a brand-new node (identity
 * transform, added directly to the active scene's root node list).
 *
 * `materialIndex` (nullable) is carried over from the ORIGINAL primitive(s)
 * that contributed to that group, so a material with no texture (a plain
 * base-color/metallic-roughness factor, doubleSided flag, alpha mode, etc.)
 * survives simplification instead of being silently replaced by the glTF
 * default material. Primitives with genuinely different materials must be
 * kept in separate `newPrimitives` entries -- one glTF primitive can only
 * reference one material -- which is also why they should generally be
 * *simplified* separately too (see group-by-material in index.html), not
 * merged into one blob and then arbitrarily assigned a single material.
 *
 * For backwards compatibility, a single `{vertices, faces, materialIndex}`
 * may be passed directly instead of `newPrimitives`.
 *
 * `vertices` MUST already be in world space (see `bakeWorldPositions`) if
 * they were merged from more than one source primitive/node -- attaching
 * to a fresh identity-transform node (rather than reusing some arbitrary
 * existing node, which has its own, generally different, transform) is
 * what makes that world-space baking actually land in the right place.
 *
 * `inPlaceReplacements` (each `{meshIndex, primIndex, vertices, faces}`)
 * handles INSTANCED primitives (see `loadPrimitives`' `instanceCount`):
 * instead of removing the primitive and adding merged/world-space geometry
 * elsewhere, its own accessors are swapped for new ones holding the
 * (locally, not world-space, simplified) geometry, in place on the SAME
 * primitive object -- every node that instances its mesh keeps instancing
 * it, just now with simplified geometry, instead of all-but-one occurrence
 * silently disappearing.
 *
 * Every other accessor/bufferView/material/image/node byte in the file is
 * left untouched.
 */
export function buildGLB(
  json,
  bin,
  { removePrimitiveRefs = [], newPrimitives, vertices, faces, materialIndex, inPlaceReplacements = [] }
) {
  const primitiveGroups = newPrimitives ?? (vertices ? [{ vertices, faces, materialIndex }] : []);

  const newJson = structuredClone(json);
  const removeSet = new Set(removePrimitiveRefs.map((r) => `${r.meshIndex}:${r.primIndex}`));

  newJson.meshes.forEach((mesh, meshIndex) => {
    mesh.primitives = mesh.primitives.filter((_, primIndex) => !removeSet.has(`${meshIndex}:${primIndex}`));
  });

  // Collect pieces and concatenate ONCE at the end rather than growing (and
  // re-copying the ENTIRE blob so far) on every one of potentially hundreds
  // of groups -- that pattern is O(groups^2) in total bytes copied, which
  // for a multi-material CAD assembly (hundreds of material groups) adds
  // up to real, needless time.
  const chunks = [new Uint8Array(bin)];
  let binLength = bin.byteLength;

  function appendGeometry(groupVertices, groupFaces) {
    const posBytes = new Float32Array(groupVertices).buffer;
    const idxBytes = new Uint32Array(groupFaces).buffer;

    const posOffset = align4(binLength);
    if (posOffset > binLength) chunks.push(new Uint8Array(posOffset - binLength));
    chunks.push(new Uint8Array(posBytes));
    binLength = posOffset + posBytes.byteLength;

    const idxOffset = align4(binLength);
    if (idxOffset > binLength) chunks.push(new Uint8Array(idxOffset - binLength));
    chunks.push(new Uint8Array(idxBytes));
    binLength = idxOffset + idxBytes.byteLength;

    const numVerts = groupVertices.length / 3;
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (let i = 0; i < numVerts; i++) {
      const x = groupVertices[i * 3], y = groupVertices[i * 3 + 1], z = groupVertices[i * 3 + 2];
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
      if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
    }

    const posBvIdx = newJson.bufferViews.length;
    newJson.bufferViews.push({ buffer: 0, byteOffset: posOffset, byteLength: posBytes.byteLength, target: 34962 });
    const idxBvIdx = newJson.bufferViews.length;
    newJson.bufferViews.push({ buffer: 0, byteOffset: idxOffset, byteLength: idxBytes.byteLength, target: 34963 });

    const posAccIdx = newJson.accessors.length;
    newJson.accessors.push({
      bufferView: posBvIdx,
      componentType: COMPONENT_TYPE.FLOAT,
      count: numVerts,
      type: "VEC3",
      min: [minX, minY, minZ],
      max: [maxX, maxY, maxZ],
    });
    const idxAccIdx = newJson.accessors.length;
    newJson.accessors.push({
      bufferView: idxBvIdx,
      componentType: COMPONENT_TYPE.UNSIGNED_INT,
      count: groupFaces.length,
      type: "SCALAR",
    });

    return { posAccIdx, idxAccIdx };
  }

  // Instanced primitives: swap accessors in place, same primitive object,
  // same mesh, same material -- every referencing node is left alone.
  for (const { meshIndex, primIndex, vertices: v, faces: f } of inPlaceReplacements) {
    const { posAccIdx, idxAccIdx } = appendGeometry(v, f);
    const prim = newJson.meshes[meshIndex].primitives[primIndex];
    prim.attributes.POSITION = posAccIdx;
    prim.indices = idxAccIdx;
    // Any other attributes (NORMAL, TEXCOORD_0, ...) referenced the OLD
    // vertex count/order and no longer correspond to the new geometry --
    // drop them rather than leave stale/mismatched data.
    for (const key of Object.keys(prim.attributes)) {
      if (key !== "POSITION") delete prim.attributes[key];
    }
  }

  const primitiveDefs = [];

  for (const group of primitiveGroups) {
    const { posAccIdx, idxAccIdx } = appendGeometry(group.vertices, group.faces);
    const primDef = { attributes: { POSITION: posAccIdx }, indices: idxAccIdx, mode: 4 };
    if (group.materialIndex != null) primDef.material = group.materialIndex;
    primitiveDefs.push(primDef);
  }

  // A mesh with zero primitives is invalid glTF -- only add one if there's
  // actually merged (non-instanced) geometry to hold. A file where every
  // editable primitive turned out to be instanced (handled entirely via
  // inPlaceReplacements above) legitimately produces none.
  if (primitiveDefs.length > 0) {
    const newMeshIndex = newJson.meshes.length;
    newJson.meshes.push({ primitives: primitiveDefs });

    const newNodeIndex = (newJson.nodes ?? (newJson.nodes = [])).length;
    newJson.nodes.push({ mesh: newMeshIndex }); // no transform fields = identity

    const sceneIndex = newJson.scene ?? 0;
    if (!newJson.scenes) newJson.scenes = [{ nodes: [] }];
    if (!newJson.scenes[sceneIndex]) newJson.scenes[sceneIndex] = { nodes: [] };
    if (!newJson.scenes[sceneIndex].nodes) newJson.scenes[sceneIndex].nodes = [];
    newJson.scenes[sceneIndex].nodes.push(newNodeIndex);
    newJson.scene = sceneIndex;
  }

  // Single concatenation of every chunk collected above, instead of the
  // repeated whole-buffer copies the old grow-in-place loop did.
  let newBin = new Uint8Array(binLength);
  let writeOffset = 0;
  for (const chunk of chunks) {
    newBin.set(chunk, writeOffset);
    writeOffset += chunk.length;
  }

  newJson.buffers = [{ byteLength: newBin.byteLength }];

  const jsonText = JSON.stringify(newJson);
  const jsonBytesRaw = new TextEncoder().encode(jsonText);
  const jsonPadded = new Uint8Array(align4(jsonBytesRaw.length));
  jsonPadded.set(jsonBytesRaw);
  jsonPadded.fill(0x20, jsonBytesRaw.length); // pad JSON with spaces per glTF spec

  const totalLength = 12 + 8 + jsonPadded.length + 8 + newBin.length;
  const out = new ArrayBuffer(totalLength);
  const dv = new DataView(out);
  let o = 0;
  dv.setUint32(o, GLB_MAGIC, true); o += 4;
  dv.setUint32(o, 2, true); o += 4; // version
  dv.setUint32(o, totalLength, true); o += 4;

  dv.setUint32(o, jsonPadded.length, true); o += 4;
  dv.setUint32(o, CHUNK_JSON, true); o += 4;
  new Uint8Array(out, o, jsonPadded.length).set(jsonPadded); o += jsonPadded.length;

  dv.setUint32(o, newBin.length, true); o += 4;
  dv.setUint32(o, CHUNK_BIN, true); o += 4;
  new Uint8Array(out, o, newBin.length).set(newBin); o += newBin.length;

  return out;
}
