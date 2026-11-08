// Pure-JS / typed-array re-implementation of CuMesh's `simplify()`
// (see ../CuMesh/src/simplify.cu and ../CuMesh/cumesh/cumesh.py:320, and
// the pure-Python port in ../CuMesh_py/simplify.py, which this mirrors).
//
// CuMesh's decimator is a *batched parallel* edge-collapse scheme designed
// for a GPU, not a sequential priority-heap decimator:
//
//   1. Compute a per-vertex quadric error (QEM) from the planes of its
//      incident faces.
//   2. Score every remaining edge with cost = QEM(midpoint) +
//      lambdaEdgeLength * edgeLength^2 + lambdaSkinny * (avg triangle-shape
//      penalty) * edgeLength^2, and mark it invalid (cost = Infinity) if
//      collapsing it would flip a triangle normal.
//   3. Each face keeps the *minimum* cost among all edges touching it
//      ("propagation").
//   4. An edge only collapses if it is at or below the current threshold
//      AND it is the argmin edge for every face touching either endpoint --
//      this is what lets many independent edges collapse in one shot
//      without an explicit conflict/lock scheme.
//   5. Compact vertices/faces, repeat until targetNumFaces is reached,
//      escalating the threshold whenever a pass frees up too little
//      progress.
//
// Unlike the NumPy port, this is written as plain loops over typed arrays
// rather than vectorized array ops: browser JS engines JIT tight numeric
// loops over Float32Array/Uint32Array well, so there's no need for NumPy's
// bincount/sort-reduceat workarounds -- a direct translation of the CUDA
// kernels' per-thread loops is both the simplest and the fastest option
// here. No dependencies, no WASM, no build step.
//
// Only `simplify()` is implemented -- CuMesh's other ops (fill_holes,
// remeshing, uv_unwrap, ...) are out of scope, same as the Python port.

/**
 * @param {Float32Array|number[]} verticesIn flat [x0,y0,z0, x1,y1,z1, ...]
 * @param {Uint32Array|number[]} facesIn flat [a0,b0,c0, a1,b1,c1, ...] vertex indices
 * @param {number} targetNumFaces stop once face count is at or below this
 * @param {object} [opts]
 * @param {number} [opts.thresh=1e-8] initial collapse-cost threshold
 * @param {number} [opts.lambdaEdgeLength=1e-2]
 * @param {number} [opts.lambdaSkinny=1e-3]
 * @param {Uint8Array|boolean[]} [opts.faceMask] true = face may be simplified,
 *   false = frozen (e.g. textured). Any vertex touched by a frozen face is
 *   locked -- never moved, never collapsed away -- so frozen faces come out
 *   byte-identical (only vertex indices get renumbered by compaction).
 * @param {(info:{pass:number, faces:number, thresh:number})=>void} [opts.onProgress]
 * @returns {{vertices: Float32Array, faces: Uint32Array, faceMask: (Uint8Array|null)}}
 */
export function simplify(verticesIn, facesIn, targetNumFaces, opts = {}) {
  if (!Number.isInteger(targetNumFaces) || targetNumFaces <= 0) {
    throw new Error("targetNumFaces must be a positive integer");
  }
  const thresh0 = opts.thresh ?? 1e-8;
  const lambdaEdgeLength = opts.lambdaEdgeLength ?? 1e-2;
  const lambdaSkinny = opts.lambdaSkinny ?? 1e-3;
  const onProgress = opts.onProgress ?? null;

  let vertices = Float32Array.from(verticesIn);
  let faces = Uint32Array.from(facesIn);
  let faceFrozen = opts.faceMask ? invertMask(opts.faceMask) : null;

  let numFace = faces.length / 3;
  if (numFace <= targetNumFaces) {
    return { vertices, faces, faceMask: faceFrozen ? invertMask(faceFrozen) : null };
  }

  let thresh = thresh0;
  const maxThresh = Math.max(thresh0 * 1e12, 1.0);

  for (let pass = 0; pass < 10000; pass++) {
    const step = simplifyStep(vertices, faces, lambdaEdgeLength, lambdaSkinny, thresh, faceFrozen);
    vertices = step.vertices;
    faces = step.faces;
    faceFrozen = step.faceFrozen;
    const newNumFace = faces.length / 3;

    if (onProgress) onProgress({ pass, faces: newNumFace, thresh });

    if (newNumFace <= targetNumFaces) break;

    const delFace = numFace - newNumFace;
    const frac = numFace > 0 ? delFace / numFace : 0;
    if (frac < 1e-2) {
      thresh *= 10;
    }
    if (thresh > maxThresh) break;
    numFace = newNumFace;
  }

  return { vertices, faces, faceMask: faceFrozen ? invertMask(faceFrozen) : null };
}

/** Async variant that yields to the event loop between passes so a page
 * stays responsive (progress bar, cancel button) during long runs. */
export async function simplifyAsync(verticesIn, facesIn, targetNumFaces, opts = {}) {
  const thresh0 = opts.thresh ?? 1e-8;
  const lambdaEdgeLength = opts.lambdaEdgeLength ?? 1e-2;
  const lambdaSkinny = opts.lambdaSkinny ?? 1e-3;
  const onProgress = opts.onProgress ?? null;
  const signal = opts.signal ?? null;

  let vertices = Float32Array.from(verticesIn);
  let faces = Uint32Array.from(facesIn);
  let faceFrozen = opts.faceMask ? invertMask(opts.faceMask) : null;

  let numFace = faces.length / 3;
  if (numFace <= targetNumFaces) {
    return { vertices, faces, faceMask: faceFrozen ? invertMask(faceFrozen) : null };
  }

  let thresh = thresh0;
  const maxThresh = Math.max(thresh0 * 1e12, 1.0);

  for (let pass = 0; pass < 10000; pass++) {
    if (signal && signal.aborted) break;

    const step = simplifyStep(vertices, faces, lambdaEdgeLength, lambdaSkinny, thresh, faceFrozen);
    vertices = step.vertices;
    faces = step.faces;
    faceFrozen = step.faceFrozen;
    const newNumFace = faces.length / 3;

    if (onProgress) onProgress({ pass, faces: newNumFace, thresh });
    await new Promise((r) => setTimeout(r, 0));

    if (newNumFace <= targetNumFaces) break;

    const delFace = numFace - newNumFace;
    const frac = numFace > 0 ? delFace / numFace : 0;
    if (frac < 1e-2) {
      thresh *= 10;
    }
    if (thresh > maxThresh) break;
    numFace = newNumFace;
  }

  return { vertices, faces, faceMask: faceFrozen ? invertMask(faceFrozen) : null };
}

function invertMask(mask) {
  const out = new Uint8Array(mask.length);
  for (let i = 0; i < mask.length; i++) out[i] = mask[i] ? 0 : 1;
  return out;
}

/** One batched parallel edge-collapse pass. */
function simplifyStep(vertices, faces, lambdaEdgeLength, lambdaSkinny, threshold, faceFrozen) {
  const V = vertices.length / 3;
  const F = faces.length / 3;

  const { normals, planeD } = computeFaceNormalsAndD(vertices, faces, F);
  const qem = computeQEM(faces, F, V, normals, planeD);
  const { edgeV0, edgeV1, edgeCount, E } = buildUniqueEdges(faces, F, V);
  if (E === 0) return { vertices, faces, faceFrozen };

  const isBoundary = boundaryVertexMask(edgeV0, edgeV1, edgeCount, E, V);
  const csr = buildVert2FaceCSR(faces, F, V);

  const vNew = new Float32Array(E * 3);
  const cost = new Float64Array(E);

  for (let e = 0; e < E; e++) {
    const va = edgeV0[e];
    const vb = edgeV1[e];
    const ba = isBoundary[va];
    const bb = isBoundary[vb];
    let w0 = 0.5;
    if (ba && !bb) w0 = 1.0;
    else if (!ba && bb) w0 = 0.0;
    const w1 = 1.0 - w0;

    const ax = vertices[va * 3], ay = vertices[va * 3 + 1], az = vertices[va * 3 + 2];
    const bx = vertices[vb * 3], by = vertices[vb * 3 + 1], bz = vertices[vb * 3 + 2];
    const vx = ax * w0 + bx * w1;
    const vy = ay * w0 + by * w1;
    const vz = az * w0 + bz * w1;
    vNew[e * 3] = vx;
    vNew[e * 3 + 1] = vy;
    vNew[e * 3 + 2] = vz;

    const qa = va * 10, qb = vb * 10;
    const xx = qem[qa] + qem[qb];
    const xy = qem[qa + 1] + qem[qb + 1];
    const xz = qem[qa + 2] + qem[qb + 2];
    const yy = qem[qa + 3] + qem[qb + 3];
    const yz = qem[qa + 4] + qem[qb + 4];
    const zz = qem[qa + 5] + qem[qb + 5];
    const qbx = qem[qa + 6] + qem[qb + 6];
    const qby = qem[qa + 7] + qem[qb + 7];
    const qbz = qem[qa + 8] + qem[qb + 8];
    const qc = qem[qa + 9] + qem[qb + 9];

    const quad = xx * vx * vx + yy * vy * vy + zz * vz * vz + 2 * (xy * vx * vy + xz * vx * vz + yz * vy * vz);
    const lin = qbx * vx + qby * vy + qbz * vz;
    const qemCost = quad + 2 * lin + qc;

    const dx = bx - ax, dy = by - ay, dz = bz - az;
    const edgeLen2 = dx * dx + dy * dy + dz * dz;

    cost[e] = qemCost + lambdaEdgeLength * edgeLen2;

    // An edge with BOTH endpoints on the boundary lies along the boundary
    // curve itself -- collapsing it would blend the two boundary vertices
    // to their midpoint (w0 = 0.5 above), moving the boundary. That's fine
    // for a genuinely open mesh, but when the "boundary" only exists
    // because the caller passed in an isolated sub-mesh (e.g. just the
    // untextured primitives of a GLB, split off from the textured ones),
    // that midpoint move detaches the seam from the untouched geometry on
    // the other side -- a visible crack/shift at the material boundary.
    // Locking boundary-to-boundary edges (never collapsing the boundary
    // curve itself, only ever collapsing an interior vertex INTO a
    // boundary vertex, which keeps it exactly where it was) avoids that.
    if (ba && bb) cost[e] = Infinity;
  }

  const skinnySum = new Float64Array(E);
  const triCount = new Uint32Array(E);
  const invalid = new Uint8Array(E);

  for (let e = 0; e < E; e++) {
    const va = edgeV0[e];
    const vb = edgeV1[e];
    accumulateIncidentTriangles(vertices, faces, csr, e, va, vb, vNew, skinnySum, triCount, invalid);
    accumulateIncidentTriangles(vertices, faces, csr, e, vb, va, vNew, skinnySum, triCount, invalid);
  }

  for (let e = 0; e < E; e++) {
    const va = edgeV0[e], vb = edgeV1[e];
    const dx = vertices[vb * 3] - vertices[va * 3];
    const dy = vertices[vb * 3 + 1] - vertices[va * 3 + 1];
    const dz = vertices[vb * 3 + 2] - vertices[va * 3 + 2];
    const edgeLen2 = dx * dx + dy * dy + dz * dz;
    const tc = triCount[e] || 1;
    const skinnyCost = skinnySum[e] / tc;
    cost[e] += lambdaSkinny * skinnyCost * edgeLen2;
    if (invalid[e]) cost[e] = Infinity;
  }

  if (faceFrozen && faceFrozen.some(Boolean)) {
    const lockedVertex = new Uint8Array(V);
    for (let f = 0; f < F; f++) {
      if (faceFrozen[f]) {
        lockedVertex[faces[f * 3]] = 1;
        lockedVertex[faces[f * 3 + 1]] = 1;
        lockedVertex[faces[f * 3 + 2]] = 1;
      }
    }
    for (let e = 0; e < E; e++) {
      if (lockedVertex[edgeV0[e]] || lockedVertex[edgeV1[e]]) cost[e] = Infinity;
    }
  }

  // propagate: each face remembers the min cost among all edges touching it
  const propagated = new Float64Array(F).fill(Infinity);
  for (let e = 0; e < E; e++) {
    const c = cost[e];
    if (!Number.isFinite(c)) continue;
    scanIncidentFaces(csr, edgeV0[e], (fid) => {
      if (c < propagated[fid]) propagated[fid] = c;
    });
    scanIncidentFaces(csr, edgeV1[e], (fid) => {
      if (c < propagated[fid]) propagated[fid] = c;
    });
  }

  // select: an edge collapses only if it's the local-min edge for every
  // face touching either endpoint
  const selected = new Uint8Array(E);
  let anySelected = false;
  for (let e = 0; e < E; e++) {
    const c = cost[e];
    if (!Number.isFinite(c) || c > threshold) continue;
    let ok = true;
    scanIncidentFaces(csr, edgeV0[e], (fid) => {
      if (propagated[fid] !== c) ok = false;
    });
    if (ok) {
      scanIncidentFaces(csr, edgeV1[e], (fid) => {
        if (propagated[fid] !== c) ok = false;
      });
    }
    if (ok) {
      selected[e] = 1;
      anySelected = true;
    }
  }

  if (!anySelected) return { vertices, faces, faceFrozen };

  return applyCollapse(vertices, faces, faceFrozen, edgeV0, edgeV1, selected, vNew, V, F, E);
}

// Shared tail for both the CPU and WebGPU paths: apply the selected edge
// collapses (vertex remap), drop degenerate faces, and compact vertices/
// faces down to a dense id range. `selected`/`vNew` may be plain typed
// arrays read back from a GPU buffer just as well as ones built on the CPU.
function applyCollapse(vertices, faces, faceFrozen, edgeV0, edgeV1, selected, vNew, V, F, E) {
  const remap = new Uint32Array(V);
  for (let v = 0; v < V; v++) remap[v] = v;
  const newVertices = Float32Array.from(vertices);
  for (let e = 0; e < E; e++) {
    if (!selected[e]) continue;
    const va = edgeV0[e], vb = edgeV1[e];
    newVertices[va * 3] = vNew[e * 3];
    newVertices[va * 3 + 1] = vNew[e * 3 + 1];
    newVertices[va * 3 + 2] = vNew[e * 3 + 2];
    remap[vb] = va;
  }

  // degenerate-face filtering + compaction
  const keptFaceIdx = [];
  for (let f = 0; f < F; f++) {
    const a = remap[faces[f * 3]];
    const b = remap[faces[f * 3 + 1]];
    const c = remap[faces[f * 3 + 2]];
    if (a !== b && b !== c && a !== c) keptFaceIdx.push(f);
  }

  const referenced = new Uint8Array(V);
  for (const f of keptFaceIdx) {
    referenced[remap[faces[f * 3]]] = 1;
    referenced[remap[faces[f * 3 + 1]]] = 1;
    referenced[remap[faces[f * 3 + 2]]] = 1;
  }
  const newId = new Int32Array(V).fill(-1);
  let nextId = 0;
  for (let v = 0; v < V; v++) if (referenced[v]) newId[v] = nextId++;

  const compactVertices = new Float32Array(nextId * 3);
  for (let v = 0; v < V; v++) {
    if (referenced[v]) {
      const nid = newId[v];
      compactVertices[nid * 3] = newVertices[v * 3];
      compactVertices[nid * 3 + 1] = newVertices[v * 3 + 1];
      compactVertices[nid * 3 + 2] = newVertices[v * 3 + 2];
    }
  }

  const compactFaces = new Uint32Array(keptFaceIdx.length * 3);
  const newFaceFrozen = faceFrozen ? new Uint8Array(keptFaceIdx.length) : null;
  for (let i = 0; i < keptFaceIdx.length; i++) {
    const f = keptFaceIdx[i];
    compactFaces[i * 3] = newId[remap[faces[f * 3]]];
    compactFaces[i * 3 + 1] = newId[remap[faces[f * 3 + 1]]];
    compactFaces[i * 3 + 2] = newId[remap[faces[f * 3 + 2]]];
    if (newFaceFrozen) newFaceFrozen[i] = faceFrozen[f];
  }

  return { vertices: compactVertices, faces: compactFaces, faceFrozen: newFaceFrozen };
}

function computeFaceNormalsAndD(vertices, faces, F) {
  const normals = new Float32Array(F * 3);
  const planeD = new Float32Array(F);
  for (let f = 0; f < F; f++) {
    const i0 = faces[f * 3], i1 = faces[f * 3 + 1], i2 = faces[f * 3 + 2];
    const ax = vertices[i0 * 3], ay = vertices[i0 * 3 + 1], az = vertices[i0 * 3 + 2];
    const e1x = vertices[i1 * 3] - ax, e1y = vertices[i1 * 3 + 1] - ay, e1z = vertices[i1 * 3 + 2] - az;
    const e2x = vertices[i2 * 3] - ax, e2y = vertices[i2 * 3 + 1] - ay, e2z = vertices[i2 * 3 + 2] - az;
    let nx = e1y * e2z - e1z * e2y;
    let ny = e1z * e2x - e1x * e2z;
    let nz = e1x * e2y - e1y * e2x;
    const len = Math.hypot(nx, ny, nz) || 1;
    nx /= len; ny /= len; nz /= len;
    normals[f * 3] = nx;
    normals[f * 3 + 1] = ny;
    normals[f * 3 + 2] = nz;
    planeD[f] = -(nx * ax + ny * ay + nz * az);
  }
  return { normals, planeD };
}

// Flat per-vertex QEM layout: 10 channels [xx,xy,xz,yy,yz,zz,bx,by,bz,c]
// (upper triangle of the symmetric plane quadric), interleaved per vertex.
function computeQEM(faces, F, V, normals, planeD) {
  const qem = new Float64Array(V * 10);
  for (let f = 0; f < F; f++) {
    const nx = normals[f * 3], ny = normals[f * 3 + 1], nz = normals[f * 3 + 2], d = planeD[f];
    const xx = nx * nx, xy = nx * ny, xz = nx * nz;
    const yy = ny * ny, yz = ny * nz, zz = nz * nz;
    const bx = nx * d, by = ny * d, bz = nz * d, c = d * d;
    for (let k = 0; k < 3; k++) {
      const base = faces[f * 3 + k] * 10;
      qem[base] += xx; qem[base + 1] += xy; qem[base + 2] += xz;
      qem[base + 3] += yy; qem[base + 4] += yz; qem[base + 5] += zz;
      qem[base + 6] += bx; qem[base + 7] += by; qem[base + 8] += bz;
      qem[base + 9] += c;
    }
  }
  return qem;
}

function edgeKey(a, b, V) {
  const lo = a < b ? a : b;
  const hi = a < b ? b : a;
  return lo * V + hi; // exact in a double as long as V^2 < 2^53 (V well under ~94M)
}

function buildUniqueEdges(faces, F, V) {
  const n = F * 3;
  const keys = new Float64Array(n);
  for (let f = 0; f < F; f++) {
    keys[f * 3] = edgeKey(faces[f * 3], faces[f * 3 + 1], V);
    keys[f * 3 + 1] = edgeKey(faces[f * 3 + 1], faces[f * 3 + 2], V);
    keys[f * 3 + 2] = edgeKey(faces[f * 3 + 2], faces[f * 3], V);
  }
  keys.sort();

  // count unique keys first, to size the output arrays
  let uniqueCount = n > 0 ? 1 : 0;
  for (let i = 1; i < n; i++) if (keys[i] !== keys[i - 1]) uniqueCount++;

  const edgeV0 = new Uint32Array(uniqueCount);
  const edgeV1 = new Uint32Array(uniqueCount);
  const edgeCount = new Uint32Array(uniqueCount);

  let w = -1;
  for (let i = 0; i < n; i++) {
    if (i === 0 || keys[i] !== keys[i - 1]) {
      w++;
      const hi = keys[i] % V;
      const lo = (keys[i] - hi) / V;
      edgeV0[w] = lo;
      edgeV1[w] = hi;
      edgeCount[w] = 1;
    } else {
      edgeCount[w]++;
    }
  }

  return { edgeV0, edgeV1, edgeCount, E: uniqueCount };
}

function boundaryVertexMask(edgeV0, edgeV1, edgeCount, E, V) {
  const isBoundary = new Uint8Array(V);
  for (let e = 0; e < E; e++) {
    if (edgeCount[e] === 1) {
      isBoundary[edgeV0[e]] = 1;
      isBoundary[edgeV1[e]] = 1;
    }
  }
  return isBoundary;
}

// CSR vertex->incident-face adjacency, built with a counting sort (no
// generic sort needed since vertex ids are bounded integers in [0, V)).
function buildVert2FaceCSR(faces, F, V) {
  const degree = new Uint32Array(V);
  for (let i = 0; i < faces.length; i++) degree[faces[i]]++;

  const offsets = new Uint32Array(V + 1);
  for (let v = 0; v < V; v++) offsets[v + 1] = offsets[v] + degree[v];

  const cursor = Uint32Array.from(offsets.subarray(0, V));
  const list = new Uint32Array(F * 3);
  for (let f = 0; f < F; f++) {
    for (let k = 0; k < 3; k++) {
      const v = faces[f * 3 + k];
      list[cursor[v]++] = f;
    }
  }
  return { list, offsets };
}

function scanIncidentFaces(csr, vertex, callback) {
  const start = csr.offsets[vertex];
  const end = csr.offsets[vertex + 1];
  for (let i = start; i < end; i++) callback(csr.list[i]);
}

// For the triangles incident to `keep` that survive collapsing edge
// (keep, other) into vNew[e]: accumulate the skinny-shape penalty and flag
// a normal flip. Mirrors process_incident_tri() in simplify.cu.
function accumulateIncidentTriangles(vertices, faces, csr, e, keep, other, vNew, skinnySumOut, triCountOut, invalidOut) {
  const vnx = vNew[e * 3], vny = vNew[e * 3 + 1], vnz = vNew[e * 3 + 2];
  const start = csr.offsets[keep];
  const end = csr.offsets[keep + 1];

  for (let i = start; i < end; i++) {
    const fid = csr.list[i];
    const i0 = faces[fid * 3], i1 = faces[fid * 3 + 1], i2 = faces[fid * 3 + 2];
    if (i0 === other || i1 === other || i2 === other) continue; // removed by the collapse, not reshaped

    const ax = vertices[i0 * 3], ay = vertices[i0 * 3 + 1], az = vertices[i0 * 3 + 2];
    const bx = vertices[i1 * 3], by = vertices[i1 * 3 + 1], bz = vertices[i1 * 3 + 2];
    const cx = vertices[i2 * 3], cy = vertices[i2 * 3 + 1], cz = vertices[i2 * 3 + 2];

    const nax = i0 === keep ? vnx : ax, nay = i0 === keep ? vny : ay, naz = i0 === keep ? vnz : az;
    const nbx = i1 === keep ? vnx : bx, nby = i1 === keep ? vny : by, nbz = i1 === keep ? vnz : bz;
    const ncx = i2 === keep ? vnx : cx, ncy = i2 === keep ? vny : cy, ncz = i2 === keep ? vnz : cz;

    const oe1x = bx - ax, oe1y = by - ay, oe1z = bz - az;
    const oe2x = cx - ax, oe2y = cy - ay, oe2z = cz - az;
    const onx = oe1y * oe2z - oe1z * oe2y;
    const ony = oe1z * oe2x - oe1x * oe2z;
    const onz = oe1x * oe2y - oe1y * oe2x;

    const ne1x = nbx - nax, ne1y = nby - nay, ne1z = nbz - naz;
    const ne2x = ncx - nax, ne2y = ncy - nay, ne2z = ncz - naz;
    const nnx = ne1y * ne2z - ne1z * ne2y;
    const nny = ne1z * ne2x - ne1x * ne2z;
    const nnz = ne1x * ne2y - ne1y * ne2x;
    const newArea = 0.5 * Math.hypot(nnx, nny, nnz);

    const dotp = onx * nnx + ony * nny + onz * nnz;
    if (dotp < 0.0) {
      invalidOut[e] = 1;
      continue;
    }

    const ne0x = ncx - nbx, ne0y = ncy - nby, ne0z = ncz - nbz;
    let denom =
      ne0x * ne0x + ne0y * ne0y + ne0z * ne0z +
      ne1x * ne1x + ne1y * ne1y + ne1z * ne1z +
      ne2x * ne2x + ne2y * ne2y + ne2z * ne2z;
    if (denom < 1e-12) denom = 1e-12;
    const shapeMetric = (4.0 * Math.sqrt(3.0) * newArea) / denom;
    const term = 1.0 - Math.min(Math.max(shapeMetric, 0.0), 1.0);

    skinnySumOut[e] += term;
    triCountOut[e]++;
  }
}

// ---------------------------------------------------------------------------
// WebGPU-accelerated path.
//
// Same algorithm as simplifyStep() above, but the per-face/per-edge numeric
// kernels (the ones that were the actual hot loops -- QEM, edge cost, the
// skinny/flip triangle scan, propagation, selection) run as WGSL compute
// shaders instead of JS loops. Edge extraction (sort-based dedup) and the
// final collapse/compaction are still done on the CPU: they're comparatively
// cheap and are sequential/prefix-sum-shaped, which GPUs aren't a natural
// fit for without a lot of extra machinery.
//
// Faithful translations of simplify.cu's own trick: the CUDA kernel
// propagates the per-face minimum cost via `atomicMin` on a *bit-packed*
// value (see pack_key_value_positive in simplify.cu). WGSL's atomics only
// operate on u32/i32 (no atomic float ops), but IEEE-754 guarantees that for
// any two non-negative finite floats, `bitcast<u32>(a) < bitcast<u32>(b)`
// iff `a < b` -- so atomicMin on the bit-cast cost works directly, no
// separate key/value packing needed (we don't need to recover *which* edge
// produced the minimum, just its cost, to do the local-argmin check).
// ---------------------------------------------------------------------------

const WORKGROUP_SIZE = 64;
const INF_BITS = 0x7f800000; // bit pattern of +Infinity as f32

export function isWebGPUSupported() {
  return typeof navigator !== "undefined" && !!navigator.gpu;
}

// The widest kernel (skinnyFlip) binds 10 storage buffers; the default
// WebGPU limit is only 8 per shader stage, so a higher limit must be
// requested explicitly (bounded by whatever the adapter actually reports).
const NEEDED_STORAGE_BUFFERS_PER_STAGE = 10;

/** Requests a WebGPU device. Throws a descriptive error if unavailable. */
export async function requestWebGPUDevice() {
  if (!isWebGPUSupported()) {
    throw new Error("WebGPU is not available in this browser (navigator.gpu is undefined).");
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("navigator.gpu.requestAdapter() returned null -- no compatible GPU adapter.");

  const maxStorageBuffers = Math.min(
    NEEDED_STORAGE_BUFFERS_PER_STAGE,
    adapter.limits.maxStorageBuffersPerShaderStage
  );
  if (maxStorageBuffers < NEEDED_STORAGE_BUFFERS_PER_STAGE) {
    throw new Error(
      `This GPU adapter only supports ${adapter.limits.maxStorageBuffersPerShaderStage} storage buffers per shader stage; ` +
        `${NEEDED_STORAGE_BUFFERS_PER_STAGE} are needed.`
    );
  }
  const device = await adapter.requestDevice({
    requiredLimits: { maxStorageBuffersPerShaderStage: maxStorageBuffers },
  });
  return device;
}

function makeBuffer(device, data, usage) {
  const buffer = device.createBuffer({
    size: Math.max(data.byteLength, 16),
    usage,
    mappedAtCreation: true,
  });
  new (data.constructor)(buffer.getMappedRange()).set(data);
  buffer.unmap();
  return buffer;
}

function makeUniform(device, u32s = [], f32s = []) {
  // Always pad to a 16-byte-aligned struct: {u32 fields..., f32 fields...}
  const numFields = Math.max(4, u32s.length + f32s.length);
  const buf = new ArrayBuffer(numFields * 4);
  const dv = new DataView(buf);
  let o = 0;
  for (const v of u32s) { dv.setUint32(o, v, true); o += 4; }
  for (const v of f32s) { dv.setFloat32(o, v, true); o += 4; }
  return makeBuffer(device, new Uint8Array(buf), GPUBufferUsage.UNIFORM);
}

// Records a copy into a fresh MAP_READ staging buffer on the SHARED
// encoder (no submit yet -- see the "one submit per pass" note below).
function recordCopyToStaging(device, encoder, srcBuffer, byteLength) {
  const staging = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  encoder.copyBufferToBuffer(srcBuffer, 0, staging, 0, byteLength);
  return staging;
}

// Must only be called after the encoder holding the matching
// recordCopyToStaging() has actually been submitted.
async function mapStagingBuffer(staging, TypedArrayCtor) {
  await staging.mapAsync(GPUMapMode.READ);
  const copy = TypedArrayCtor.from(new TypedArrayCtor(staging.getMappedRange()));
  staging.unmap();
  staging.destroy();
  return copy;
}

function dispatchCount(n) {
  return Math.max(1, Math.ceil(n / WORKGROUP_SIZE));
}

const WGSL = {
  faceNormals: `
struct Params { count: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> vertices: array<f32>;
@group(0) @binding(2) var<storage, read> faces: array<u32>;
@group(0) @binding(3) var<storage, read_write> normals: array<f32>;
@group(0) @binding(4) var<storage, read_write> planeD: array<f32>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let f = gid.x;
  if (f >= params.count) { return; }
  let i0 = faces[f*3u]; let i1 = faces[f*3u+1u]; let i2 = faces[f*3u+2u];
  let ax = vertices[i0*3u]; let ay = vertices[i0*3u+1u]; let az = vertices[i0*3u+2u];
  let e1x = vertices[i1*3u]-ax; let e1y = vertices[i1*3u+1u]-ay; let e1z = vertices[i1*3u+2u]-az;
  let e2x = vertices[i2*3u]-ax; let e2y = vertices[i2*3u+1u]-ay; let e2z = vertices[i2*3u+2u]-az;
  var nx = e1y*e2z - e1z*e2y;
  var ny = e1z*e2x - e1x*e2z;
  var nz = e1x*e2y - e1y*e2x;
  var len = sqrt(nx*nx + ny*ny + nz*nz);
  if (len < 1e-20) { len = 1.0; }
  nx = nx / len; ny = ny / len; nz = nz / len;
  normals[f*3u] = nx; normals[f*3u+1u] = ny; normals[f*3u+2u] = nz;
  planeD[f] = -(nx*ax + ny*ay + nz*az);
}
`,

  qemAccumulate: `
struct Params { count: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> normals: array<f32>;
@group(0) @binding(2) var<storage, read> planeD: array<f32>;
@group(0) @binding(3) var<storage, read> v2fList: array<u32>;
@group(0) @binding(4) var<storage, read> v2fOffsets: array<u32>;
@group(0) @binding(5) var<storage, read_write> qem: array<f32>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let v = gid.x;
  if (v >= params.count) { return; }
  var xx = 0.0; var xy = 0.0; var xz = 0.0; var yy = 0.0; var yz = 0.0; var zz = 0.0;
  var bx = 0.0; var by = 0.0; var bz = 0.0; var c = 0.0;
  let start = v2fOffsets[v]; let end = v2fOffsets[v+1u];
  for (var i = start; i < end; i = i + 1u) {
    let f = v2fList[i];
    let nx = normals[f*3u]; let ny = normals[f*3u+1u]; let nz = normals[f*3u+2u]; let d = planeD[f];
    xx += nx*nx; xy += nx*ny; xz += nx*nz;
    yy += ny*ny; yz += ny*nz; zz += nz*nz;
    bx += nx*d; by += ny*d; bz += nz*d; c += d*d;
  }
  let base = v*10u;
  qem[base]=xx; qem[base+1u]=xy; qem[base+2u]=xz;
  qem[base+3u]=yy; qem[base+4u]=yz; qem[base+5u]=zz;
  qem[base+6u]=bx; qem[base+7u]=by; qem[base+8u]=bz;
  qem[base+9u]=c;
}
`,

  edgeCostBase: `
struct Params { count: u32, pad0: u32, pad1: u32, lambdaEdgeLength: f32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> vertices: array<f32>;
@group(0) @binding(2) var<storage, read> edgeV0: array<u32>;
@group(0) @binding(3) var<storage, read> edgeV1: array<u32>;
@group(0) @binding(4) var<storage, read> isBoundary: array<u32>;
@group(0) @binding(5) var<storage, read> qem: array<f32>;
@group(0) @binding(6) var<storage, read_write> vNew: array<f32>;
@group(0) @binding(7) var<storage, read_write> cost: array<f32>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let e = gid.x;
  if (e >= params.count) { return; }
  let va = edgeV0[e]; let vb = edgeV1[e];
  let ba = isBoundary[va]; let bb = isBoundary[vb];
  var w0 = 0.5;
  if (ba != 0u && bb == 0u) { w0 = 1.0; } else if (ba == 0u && bb != 0u) { w0 = 0.0; }
  let w1 = 1.0 - w0;

  let ax = vertices[va*3u]; let ay = vertices[va*3u+1u]; let az = vertices[va*3u+2u];
  let bx = vertices[vb*3u]; let by = vertices[vb*3u+1u]; let bz = vertices[vb*3u+2u];
  let vx = ax*w0 + bx*w1;
  let vy = ay*w0 + by*w1;
  let vz = az*w0 + bz*w1;
  vNew[e*3u] = vx; vNew[e*3u+1u] = vy; vNew[e*3u+2u] = vz;

  let qa = va*10u; let qb = vb*10u;
  let xx = qem[qa]+qem[qb]; let xy = qem[qa+1u]+qem[qb+1u]; let xz = qem[qa+2u]+qem[qb+2u];
  let yy = qem[qa+3u]+qem[qb+3u]; let yz = qem[qa+4u]+qem[qb+4u]; let zz = qem[qa+5u]+qem[qb+5u];
  let qbx = qem[qa+6u]+qem[qb+6u]; let qby = qem[qa+7u]+qem[qb+7u]; let qbz = qem[qa+8u]+qem[qb+8u];
  let qc = qem[qa+9u]+qem[qb+9u];

  let quad = xx*vx*vx + yy*vy*vy + zz*vz*vz + 2.0*(xy*vx*vy + xz*vx*vz + yz*vy*vz);
  let lin = qbx*vx + qby*vy + qbz*vz;
  let qemCost = quad + 2.0*lin + qc;

  let dx = bx-ax; let dy = by-ay; let dz = bz-az;
  let edgeLen2 = dx*dx + dy*dy + dz*dz;

  var c = qemCost + params.lambdaEdgeLength * edgeLen2;
  // Both endpoints on the boundary -> this edge lies along the boundary
  // curve itself; never collapse it (see the matching comment in the CPU
  // path's simplifyStep for why -- avoids detaching a material seam).
  if (ba != 0u && bb != 0u) { c = 3.0e38; }
  cost[e] = c;
}
`,

  // For each edge, scans the incident faces of both endpoints (mirrors
  // accumulateIncidentTriangles() twice, once per endpoint as "keep").
  skinnyFlip: `
struct Params { count: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> vertices: array<f32>;
@group(0) @binding(2) var<storage, read> faces: array<u32>;
@group(0) @binding(3) var<storage, read> edgeV0: array<u32>;
@group(0) @binding(4) var<storage, read> edgeV1: array<u32>;
@group(0) @binding(5) var<storage, read> vNew: array<f32>;
@group(0) @binding(6) var<storage, read> v2fList: array<u32>;
@group(0) @binding(7) var<storage, read> v2fOffsets: array<u32>;
@group(0) @binding(8) var<storage, read_write> skinnySum: array<f32>;
@group(0) @binding(9) var<storage, read_write> triCount: array<u32>;
@group(0) @binding(10) var<storage, read_write> invalidFlags: array<u32>;

fn scanSide(e: u32, keep: u32, other: u32, vnx: f32, vny: f32, vnz: f32) -> vec2<f32> {
  var sum = 0.0;
  var cnt = 0.0;
  let start = v2fOffsets[keep]; let end = v2fOffsets[keep+1u];
  for (var i = start; i < end; i = i + 1u) {
    let fid = v2fList[i];
    let i0 = faces[fid*3u]; let i1 = faces[fid*3u+1u]; let i2 = faces[fid*3u+2u];
    if (i0 == other || i1 == other || i2 == other) { continue; }

    let ax = vertices[i0*3u]; let ay = vertices[i0*3u+1u]; let az = vertices[i0*3u+2u];
    let bx = vertices[i1*3u]; let by = vertices[i1*3u+1u]; let bz = vertices[i1*3u+2u];
    let cx = vertices[i2*3u]; let cy = vertices[i2*3u+1u]; let cz = vertices[i2*3u+2u];

    var nax = ax; var nay = ay; var naz = az;
    if (i0 == keep) { nax = vnx; nay = vny; naz = vnz; }
    var nbx = bx; var nby = by; var nbz = bz;
    if (i1 == keep) { nbx = vnx; nby = vny; nbz = vnz; }
    var ncx = cx; var ncy = cy; var ncz = cz;
    if (i2 == keep) { ncx = vnx; ncy = vny; ncz = vnz; }

    let oe1x = bx-ax; let oe1y = by-ay; let oe1z = bz-az;
    let oe2x = cx-ax; let oe2y = cy-ay; let oe2z = cz-az;
    let onx = oe1y*oe2z - oe1z*oe2y;
    let ony = oe1z*oe2x - oe1x*oe2z;
    let onz = oe1x*oe2y - oe1y*oe2x;

    let ne1x = nbx-nax; let ne1y = nby-nay; let ne1z = nbz-naz;
    let ne2x = ncx-nax; let ne2y = ncy-nay; let ne2z = ncz-naz;
    let nnx = ne1y*ne2z - ne1z*ne2y;
    let nny = ne1z*ne2x - ne1x*ne2z;
    let nnz = ne1x*ne2y - ne1y*ne2x;
    let newArea = 0.5 * sqrt(nnx*nnx + nny*nny + nnz*nnz);

    let dotp = onx*nnx + ony*nny + onz*nnz;
    if (dotp < 0.0) {
      invalidFlags[e] = 1u;
      continue;
    }

    let ne0x = ncx-nbx; let ne0y = ncy-nby; let ne0z = ncz-nbz;
    var denom = ne0x*ne0x+ne0y*ne0y+ne0z*ne0z + ne1x*ne1x+ne1y*ne1y+ne1z*ne1z + ne2x*ne2x+ne2y*ne2y+ne2z*ne2z;
    if (denom < 1e-12) { denom = 1e-12; }
    let shapeMetric = (4.0 * sqrt(3.0) * newArea) / denom;
    let term = 1.0 - clamp(shapeMetric, 0.0, 1.0);
    sum += term;
    cnt += 1.0;
  }
  return vec2<f32>(sum, cnt);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let e = gid.x;
  if (e >= params.count) { return; }
  let va = edgeV0[e]; let vb = edgeV1[e];
  let vnx = vNew[e*3u]; let vny = vNew[e*3u+1u]; let vnz = vNew[e*3u+2u];
  let sideA = scanSide(e, va, vb, vnx, vny, vnz);
  let sideB = scanSide(e, vb, va, vnx, vny, vnz);
  skinnySum[e] = sideA.x + sideB.x;
  triCount[e] = u32(sideA.y + sideB.y);
}
`,

  finalizeCost: `
struct Params { count: u32, pad0: u32, pad1: u32, lambdaSkinny: f32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> edgeV0: array<u32>;
@group(0) @binding(2) var<storage, read> edgeV1: array<u32>;
@group(0) @binding(3) var<storage, read> vertices: array<f32>;
@group(0) @binding(4) var<storage, read> skinnySum: array<f32>;
@group(0) @binding(5) var<storage, read> triCount: array<u32>;
@group(0) @binding(6) var<storage, read> invalidFlags: array<u32>;
@group(0) @binding(7) var<storage, read> lockedVertex: array<u32>;
@group(0) @binding(8) var<storage, read_write> cost: array<f32>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let e = gid.x;
  if (e >= params.count) { return; }
  let va = edgeV0[e]; let vb = edgeV1[e];
  let dx = vertices[vb*3u]-vertices[va*3u];
  let dy = vertices[vb*3u+1u]-vertices[va*3u+1u];
  let dz = vertices[vb*3u+2u]-vertices[va*3u+2u];
  let edgeLen2 = dx*dx + dy*dy + dz*dz;
  var tc = triCount[e];
  if (tc == 0u) { tc = 1u; }
  let skinnyCost = skinnySum[e] / f32(tc);

  var c = cost[e] + params.lambdaSkinny * skinnyCost * edgeLen2;
  if (invalidFlags[e] != 0u || lockedVertex[va] != 0u || lockedVertex[vb] != 0u) {
    // WGSL forbids folding a const-expression to +Infinity at shader-creation
    // time (bitcast<f32>(0x7f800000u) as a literal errors); a large finite
    // sentinel does the same job here (never wins atomicMin against a real
    // cost, and comparisons against any realistic threshold behave the same).
    c = 3.0e38;
  }
  cost[e] = c;
}
`,

  // Scatters cost[e] via atomicMin into every face incident to either
  // endpoint. `propagated` must be pre-filled with the +Infinity bit pattern.
  propagateCost: `
struct Params { count: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> edgeV0: array<u32>;
@group(0) @binding(2) var<storage, read> edgeV1: array<u32>;
@group(0) @binding(3) var<storage, read> cost: array<f32>;
@group(0) @binding(4) var<storage, read> v2fList: array<u32>;
@group(0) @binding(5) var<storage, read> v2fOffsets: array<u32>;
@group(0) @binding(6) var<storage, read_write> propagated: array<atomic<u32>>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let e = gid.x;
  if (e >= params.count) { return; }
  let bits = bitcast<u32>(cost[e]);
  for (var side = 0u; side < 2u; side = side + 1u) {
    var v = edgeV0[e];
    if (side == 1u) { v = edgeV1[e]; }
    let start = v2fOffsets[v]; let end = v2fOffsets[v+1u];
    for (var i = start; i < end; i = i + 1u) {
      let fid = v2fList[i];
      atomicMin(&propagated[fid], bits);
    }
  }
}
`,

  selectEdges: `
struct Params { count: u32, pad0: u32, pad1: u32, threshold: f32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> edgeV0: array<u32>;
@group(0) @binding(2) var<storage, read> edgeV1: array<u32>;
@group(0) @binding(3) var<storage, read> cost: array<f32>;
@group(0) @binding(4) var<storage, read> propagated: array<u32>;
@group(0) @binding(5) var<storage, read> v2fList: array<u32>;
@group(0) @binding(6) var<storage, read> v2fOffsets: array<u32>;
@group(0) @binding(7) var<storage, read_write> selected: array<u32>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let e = gid.x;
  if (e >= params.count) { return; }
  let c = cost[e];
  if (c > params.threshold) { selected[e] = 0u; return; }
  let bits = bitcast<u32>(c);
  var ok = true;
  for (var side = 0u; side < 2u; side = side + 1u) {
    var v = edgeV0[e];
    if (side == 1u) { v = edgeV1[e]; }
    let start = v2fOffsets[v]; let end = v2fOffsets[v+1u];
    for (var i = start; i < end; i = i + 1u) {
      if (propagated[v2fList[i]] != bits) { ok = false; }
    }
  }
  if (ok) { selected[e] = 1u; } else { selected[e] = 0u; }
}
`,
};

// Compiling the 8 compute pipelines is not cheap (shader module parse +
// pipeline creation for each). A caller that simplifies many separate
// groups against the same device (e.g. one call per material, which for a
// multi-material CAD assembly can be hundreds of groups) must not pay that
// cost again for every group -- cache by device so it happens once.
const gpuPipelineCache = new WeakMap();

async function getGPUPipelines(device) {
  let pipelines = gpuPipelineCache.get(device);
  if (pipelines) return pipelines;

  pipelines = {};
  for (const [name, source] of Object.entries(WGSL)) {
    const module = device.createShaderModule({ code: source });
    pipelines[name] = await device.createComputePipelineAsync({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
  }
  gpuPipelineCache.set(device, pipelines);
  return pipelines;
}

// Records one compute dispatch onto a SHARED encoder (no submit). Each
// device.queue.submit()/mapAsync round-trip has real, fixed GPU-driver IPC
// latency regardless of how little work is being computed -- recording
// every kernel of a pass (and both readback copies) onto one encoder and
// submitting once, instead of once per kernel, turns ~9 round-trips per
// algorithmic pass into 1, which matters enormously when many small
// material groups each only need a handful of passes.
function recordPass(device, encoder, pipeline, bindings, count) {
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: bindings.map((resource, i) => ({ binding: i, resource: { buffer: resource } })),
  });
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(dispatchCount(count));
  pass.end();
}

// Computed lazily (not at module load time): referencing GPUBufferUsage at
// import time would throw in any environment without WebGPU globals (e.g.
// Node, or a browser lacking WebGPU support), breaking the CPU-only path
// for everyone even if they never call the GPU functions.
function storageUsage() {
  return GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
}

/** One batched parallel edge-collapse pass, run on the GPU. */
async function simplifyStepGPU(device, pipelines, vertices, faces, lambdaEdgeLength, lambdaSkinny, threshold, faceFrozen) {
  const V = vertices.length / 3;
  const F = faces.length / 3;

  const { edgeV0, edgeV1, edgeCount, E } = buildUniqueEdges(faces, F, V);
  if (E === 0) return { vertices, faces, faceFrozen };

  const isBoundary = boundaryVertexMask(edgeV0, edgeV1, edgeCount, E, V);
  const csr = buildVert2FaceCSR(faces, F, V);

  const lockedVertex = new Uint32Array(V);
  if (faceFrozen && faceFrozen.some(Boolean)) {
    for (let f = 0; f < F; f++) {
      if (faceFrozen[f]) {
        lockedVertex[faces[f * 3]] = 1;
        lockedVertex[faces[f * 3 + 1]] = 1;
        lockedVertex[faces[f * 3 + 2]] = 1;
      }
    }
  }

  const bVertices = makeBuffer(device, vertices, storageUsage());
  const bFaces = makeBuffer(device, faces, storageUsage());
  const bEdgeV0 = makeBuffer(device, edgeV0, storageUsage());
  const bEdgeV1 = makeBuffer(device, edgeV1, storageUsage());
  const bIsBoundary = makeBuffer(device, Uint32Array.from(isBoundary), storageUsage());
  const bV2fList = makeBuffer(device, csr.list, storageUsage());
  const bV2fOffsets = makeBuffer(device, csr.offsets, storageUsage());
  const bLockedVertex = makeBuffer(device, lockedVertex, storageUsage());

  const bNormals = device.createBuffer({ size: Math.max(F * 3 * 4, 16), usage: storageUsage() });
  const bPlaneD = device.createBuffer({ size: Math.max(F * 4, 16), usage: storageUsage() });
  const bQem = device.createBuffer({ size: Math.max(V * 10 * 4, 16), usage: storageUsage() });
  const bVNew = device.createBuffer({ size: Math.max(E * 3 * 4, 16), usage: storageUsage() });
  const bCost = device.createBuffer({ size: Math.max(E * 4, 16), usage: storageUsage() });
  const bSkinnySum = device.createBuffer({ size: Math.max(E * 4, 16), usage: storageUsage() });
  const bTriCount = device.createBuffer({ size: Math.max(E * 4, 16), usage: storageUsage() });
  const bInvalid = device.createBuffer({ size: Math.max(E * 4, 16), usage: storageUsage() });
  const bPropagated = makeBuffer(device, new Uint32Array(F).fill(INF_BITS), storageUsage());
  const bSelected = device.createBuffer({ size: Math.max(E * 4, 16), usage: storageUsage() });

  // Every kernel of this pass, plus both readback copies, are recorded
  // onto ONE shared encoder and submitted ONCE below -- see recordPass's
  // comment for why that matters. WebGPU's automatic inter-pass
  // synchronization means recording them sequentially into one encoder is
  // just as correctly ordered as separate submits would have been (e.g.
  // selectEdges reading what propagateCost's atomicMin wrote).
  const encoder = device.createCommandEncoder();

  recordPass(device, encoder, pipelines.faceNormals, [makeUniform(device, [F]), bVertices, bFaces, bNormals, bPlaneD], F);
  recordPass(device, encoder, pipelines.qemAccumulate, [makeUniform(device, [V]), bNormals, bPlaneD, bV2fList, bV2fOffsets, bQem], V);
  recordPass(
    device, encoder, pipelines.edgeCostBase,
    [makeUniform(device, [E, 0, 0], [lambdaEdgeLength]), bVertices, bEdgeV0, bEdgeV1, bIsBoundary, bQem, bVNew, bCost],
    E
  );
  recordPass(
    device, encoder, pipelines.skinnyFlip,
    [makeUniform(device, [E]), bVertices, bFaces, bEdgeV0, bEdgeV1, bVNew, bV2fList, bV2fOffsets, bSkinnySum, bTriCount, bInvalid],
    E
  );
  recordPass(
    device, encoder, pipelines.finalizeCost,
    [makeUniform(device, [E, 0, 0], [lambdaSkinny]), bEdgeV0, bEdgeV1, bVertices, bSkinnySum, bTriCount, bInvalid, bLockedVertex, bCost],
    E
  );
  recordPass(device, encoder, pipelines.propagateCost, [makeUniform(device, [E]), bEdgeV0, bEdgeV1, bCost, bV2fList, bV2fOffsets, bPropagated], E);
  recordPass(
    device, encoder, pipelines.selectEdges,
    [makeUniform(device, [E, 0, 0], [threshold]), bEdgeV0, bEdgeV1, bCost, bPropagated, bV2fList, bV2fOffsets, bSelected],
    E
  );

  const selectedStaging = recordCopyToStaging(device, encoder, bSelected, E * 4);
  const vNewStaging = recordCopyToStaging(device, encoder, bVNew, E * 3 * 4);

  device.queue.submit([encoder.finish()]);

  const [selected, vNewFlat] = await Promise.all([
    mapStagingBuffer(selectedStaging, Uint32Array),
    mapStagingBuffer(vNewStaging, Float32Array),
  ]);

  for (const b of [
    bVertices, bFaces, bEdgeV0, bEdgeV1, bIsBoundary, bV2fList, bV2fOffsets, bLockedVertex,
    bNormals, bPlaneD, bQem, bVNew, bCost, bSkinnySum, bTriCount, bInvalid, bPropagated, bSelected,
  ]) b.destroy();

  let anySelected = false;
  for (let e = 0; e < E; e++) if (selected[e]) { anySelected = true; break; }
  if (!anySelected) return { vertices, faces, faceFrozen };

  return applyCollapse(vertices, faces, faceFrozen, edgeV0, edgeV1, selected, vNewFlat, V, F, E);
}

/**
 * WebGPU-accelerated equivalent of `simplifyAsync()`. Requires a browser
 * with WebGPU support (check with `isWebGPUSupported()` first). Accepts the
 * same options, plus:
 * @param {GPUDevice} [opts.device] reuse an existing device (e.g. one the
 *   page already created); otherwise one is requested internally.
 */
export async function simplifyWebGPU(verticesIn, facesIn, targetNumFaces, opts = {}) {
  if (!Number.isInteger(targetNumFaces) || targetNumFaces <= 0) {
    throw new Error("targetNumFaces must be a positive integer");
  }
  const thresh0 = opts.thresh ?? 1e-8;
  const lambdaEdgeLength = opts.lambdaEdgeLength ?? 1e-2;
  const lambdaSkinny = opts.lambdaSkinny ?? 1e-3;
  const onProgress = opts.onProgress ?? null;
  const signal = opts.signal ?? null;

  let vertices = Float32Array.from(verticesIn);
  let faces = Uint32Array.from(facesIn);
  let faceFrozen = opts.faceMask ? invertMask(opts.faceMask) : null;

  let numFace = faces.length / 3;
  if (numFace <= targetNumFaces) {
    return { vertices, faces, faceMask: faceFrozen ? invertMask(faceFrozen) : null };
  }

  // Requesting a device and compiling pipelines only happens once actual
  // work is known to be needed, and (via getGPUPipelines' cache) at most
  // once per device even across many calls -- important when a caller
  // simplifies many separate groups against the same device (e.g. one call
  // per material on a multi-material assembly).
  const device = opts.device ?? (await requestWebGPUDevice());
  const pipelines = await getGPUPipelines(device);

  let thresh = thresh0;
  const maxThresh = Math.max(thresh0 * 1e12, 1.0);

  for (let pass = 0; pass < 10000; pass++) {
    if (signal && signal.aborted) break;

    const step = await simplifyStepGPU(device, pipelines, vertices, faces, lambdaEdgeLength, lambdaSkinny, thresh, faceFrozen);
    vertices = step.vertices;
    faces = step.faces;
    faceFrozen = step.faceFrozen;
    const newNumFace = faces.length / 3;

    if (onProgress) onProgress({ pass, faces: newNumFace, thresh });

    if (newNumFace <= targetNumFaces) break;

    const delFace = numFace - newNumFace;
    const frac = numFace > 0 ? delFace / numFace : 0;
    if (frac < 1e-2) {
      thresh *= 10;
    }
    if (thresh > maxThresh) break;
    numFace = newNumFace;
  }

  return { vertices, faces, faceMask: faceFrozen ? invertMask(faceFrozen) : null };
}
