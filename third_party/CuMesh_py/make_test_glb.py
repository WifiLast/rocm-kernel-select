"""Builds a small synthetic GLB with two primitives -- a textured cube and
an untextured icosphere -- to exercise `run_on_glb.py --untextured-only`
end to end (the real sample asset only has a single, fully textured
material, so it can't demonstrate the split)."""

import numpy as np
import pygltflib

from test_simplify import _icosphere


def _cube():
    v = np.array(
        [
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
        ],
        dtype=np.float64,
    )
    f = np.array(
        [
            [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
            [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0],
        ],
        dtype=np.int64,
    )
    return v, f


def _checkerboard_png_bytes(size=8, cell=8):
    import struct
    import zlib

    n = size * cell
    rows = []
    for y in range(n):
        row = bytearray([0])  # filter byte
        for x in range(n):
            on = ((x // cell) + (y // cell)) % 2 == 0
            val = 255 if on else 40
            row += bytes([val, val, val])
        rows.append(bytes(row))
    raw = b"".join(rows)

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", n, n, 8, 2, 0, 0, 0)
    idat = zlib.compress(raw, 9)
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def build(out_path="synthetic_textured_untextured.glb"):
    cube_v, cube_f = _cube()
    sphere_v, sphere_f = _icosphere(subdivisions=4)
    sphere_v = sphere_v * 0.6 + np.array([3.0, 0.0, 0.0])  # move it beside the cube

    cube_uv = np.zeros((cube_v.shape[0], 2), dtype=np.float32)
    cube_uv[:, 0] = (cube_v[:, 0] + 1) / 2
    cube_uv[:, 1] = (cube_v[:, 1] + 1) / 2

    png_bytes = _checkerboard_png_bytes()

    cube_v32 = cube_v.astype(np.float32)
    cube_f32 = cube_f.astype(np.uint32)
    sphere_v32 = sphere_v.astype(np.float32)
    sphere_f32 = sphere_f.astype(np.uint32)

    blobs = [
        cube_v32.tobytes(),
        cube_uv.tobytes(),
        cube_f32.tobytes(),
        sphere_v32.tobytes(),
        sphere_f32.tobytes(),
        png_bytes,
    ]
    offsets = [0]
    for b in blobs:
        offsets.append(offsets[-1] + len(b))
    blob = b"".join(blobs)

    gltf = pygltflib.GLTF2()
    gltf.scenes = [pygltflib.Scene(nodes=[0])]
    gltf.scene = 0
    gltf.nodes = [pygltflib.Node(mesh=0)]

    gltf.images = [pygltflib.Image(bufferView=5, mimeType="image/png")]
    gltf.samplers = [pygltflib.Sampler()]
    gltf.textures = [pygltflib.Texture(source=0, sampler=0)]
    gltf.materials = [
        pygltflib.Material(
            name="textured_cube_mat",
            pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                baseColorTexture=pygltflib.TextureInfo(index=0)
            ),
        ),
        pygltflib.Material(name="plain_sphere_mat"),
    ]

    gltf.bufferViews = [
        pygltflib.BufferView(buffer=0, byteOffset=offsets[0], byteLength=len(blobs[0]), target=pygltflib.ARRAY_BUFFER),
        pygltflib.BufferView(buffer=0, byteOffset=offsets[1], byteLength=len(blobs[1]), target=pygltflib.ARRAY_BUFFER),
        pygltflib.BufferView(buffer=0, byteOffset=offsets[2], byteLength=len(blobs[2]), target=pygltflib.ELEMENT_ARRAY_BUFFER),
        pygltflib.BufferView(buffer=0, byteOffset=offsets[3], byteLength=len(blobs[3]), target=pygltflib.ARRAY_BUFFER),
        pygltflib.BufferView(buffer=0, byteOffset=offsets[4], byteLength=len(blobs[4]), target=pygltflib.ELEMENT_ARRAY_BUFFER),
        pygltflib.BufferView(buffer=0, byteOffset=offsets[5], byteLength=len(blobs[5])),
    ]

    gltf.accessors = [
        pygltflib.Accessor(bufferView=0, componentType=pygltflib.FLOAT, count=cube_v32.shape[0], type=pygltflib.VEC3, min=cube_v32.min(0).tolist(), max=cube_v32.max(0).tolist()),
        pygltflib.Accessor(bufferView=1, componentType=pygltflib.FLOAT, count=cube_uv.shape[0], type=pygltflib.VEC2),
        pygltflib.Accessor(bufferView=2, componentType=pygltflib.UNSIGNED_INT, count=int(cube_f32.size), type=pygltflib.SCALAR),
        pygltflib.Accessor(bufferView=3, componentType=pygltflib.FLOAT, count=sphere_v32.shape[0], type=pygltflib.VEC3, min=sphere_v32.min(0).tolist(), max=sphere_v32.max(0).tolist()),
        pygltflib.Accessor(bufferView=4, componentType=pygltflib.UNSIGNED_INT, count=int(sphere_f32.size), type=pygltflib.SCALAR),
    ]

    gltf.meshes = [
        pygltflib.Mesh(
            primitives=[
                pygltflib.Primitive(
                    attributes=pygltflib.Attributes(POSITION=0, TEXCOORD_0=1),
                    indices=2,
                    material=0,
                    mode=pygltflib.TRIANGLES,
                ),
                pygltflib.Primitive(
                    attributes=pygltflib.Attributes(POSITION=3),
                    indices=4,
                    material=1,
                    mode=pygltflib.TRIANGLES,
                ),
            ]
        )
    ]
    gltf.buffers = [pygltflib.Buffer(byteLength=len(blob))]
    gltf.set_binary_blob(blob)
    gltf.save(out_path)
    print(f"wrote {out_path}: cube F={cube_f.shape[0]} (textured), sphere F={sphere_f.shape[0]} (untextured)")


if __name__ == "__main__":
    build()
