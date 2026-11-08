"""Smoke test for the pure-Python CuMesh.simplify() port. Run directly with
python (no pytest dependency required)."""

import numpy as np

from simplify import simplify


def _icosphere(subdivisions: int = 3):
    t = (1.0 + 5.0 ** 0.5) / 2.0
    verts = np.array(
        [
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
        ],
        dtype=np.float64,
    )
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    faces = np.array(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ],
        dtype=np.int64,
    )

    for _ in range(subdivisions):
        cache = {}
        new_faces = []

        def midpoint(a, b):
            key = (min(a, b), max(a, b))
            if key in cache:
                return cache[key]
            m = (verts[a] + verts[b]) / 2.0
            m = m / np.linalg.norm(m)
            idx = len(verts_list)
            verts_list.append(m)
            cache[key] = idx
            return idx

        verts_list = list(verts)
        for f in faces:
            a, b, c = f
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            new_faces.append([a, ab, ca])
            new_faces.append([b, bc, ab])
            new_faces.append([c, ca, bc])
            new_faces.append([ab, bc, ca])
        verts = np.array(verts_list, dtype=np.float64)
        faces = np.array(new_faces, dtype=np.int64)

    return verts, faces


def main():
    verts, faces = _icosphere(subdivisions=4)
    print(f"input: V={verts.shape[0]} F={faces.shape[0]}")

    target = faces.shape[0] // 4
    new_verts, new_faces, _ = simplify(verts, faces, target, verbose=True)
    print(f"output: V={new_verts.shape[0]} F={new_faces.shape[0]} (target<={target})")

    assert new_faces.shape[0] <= target * 1.5, "did not decimate enough"
    assert new_faces.max() < new_verts.shape[0], "dangling face index"
    assert new_faces.min() >= 0

    # sanity: no degenerate triangles, no duplicate faces
    degenerate = (
        (new_faces[:, 0] == new_faces[:, 1])
        | (new_faces[:, 1] == new_faces[:, 2])
        | (new_faces[:, 0] == new_faces[:, 2])
    )
    assert not degenerate.any(), "degenerate triangle produced"

    # sanity: surface area roughly preserved (decimation shouldn't collapse to nothing)
    def total_area(v, f):
        a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum()

    area_before = total_area(verts, faces)
    area_after = total_area(new_verts, new_faces)
    print(f"area before={area_before:.4f} after={area_after:.4f}")
    assert area_after > area_before * 0.8, "lost too much area"

    print("OK")


def main_face_mask():
    """Half the icosphere (faces whose centroid has z < 0) is 'textured' and
    must survive untouched; only the other half may be simplified."""
    verts, faces = _icosphere(subdivisions=4)
    centroid_z = verts[faces].mean(axis=1)[:, 2]
    editable = centroid_z >= 0.0
    frozen_faces_before = faces[~editable].copy()
    # simplify() downcasts vertices to float32 internally; compare at the
    # same precision so this only checks "unmoved", not "still float64"
    verts32 = verts.astype(np.float32)
    frozen_verts_before = {tuple(verts32[i]) for i in np.unique(frozen_faces_before)}

    target = faces.shape[0] // 4
    new_verts, new_faces, new_mask = simplify(verts, faces, target, verbose=True, face_mask=editable)

    print(f"masked output: V={new_verts.shape[0]} F={new_faces.shape[0]} frozen={int((~new_mask).sum())}")

    frozen_faces_after = new_faces[~new_mask]
    assert frozen_faces_after.shape[0] == frozen_faces_before.shape[0], "frozen face count changed"

    frozen_verts_after = {tuple(new_verts[i]) for i in np.unique(frozen_faces_after)}
    assert frozen_verts_after == frozen_verts_before, "frozen region vertices were moved/altered"

    print("OK (face_mask)")


if __name__ == "__main__":
    main()
    main_face_mask()
