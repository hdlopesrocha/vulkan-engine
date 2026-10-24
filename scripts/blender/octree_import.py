#!/usr/bin/env python3
"""Blender importer + material builder for the octree Blender export.

Usage:
  blender --background --python octree_import.py -- <config.octree.json | file.glb> [options]
  or open this file in Blender's Scripting workspace and press Run Script
  (it then looks for a config/glb next to this file).

Options:
  --render <png>            render a validation image after setup
  --save-blend <path>       save the configured scene as a .blend file
  --engine <eevee|cycles>   render engine (default: eevee)
  --samples <n>             render samples (default: 64)
  --no-grass                skip grass blade generation
  --no-sdf                  skip SDF proxy objects
  --no-light                keep the scene's existing lights

The material graph is a port of the Vulkan solid renderer:
  shaders/includes/solid/SolidTesc.glsl      (per-patch unique slots/weights)
  shaders/includes/solid/SolidTese.glsl      (barycentric weights, flat indices)
  shaders/includes/solid/SolidSurface.glsl   (albedo/normal/rough/AO blending)
  shaders/includes/solid/Triplanar.glsl      (triplanar UVs and dominant picks)
  shaders/includes/solid/TangentBasis.glsl   (normal map conventions)
  shaders/includes/Hsv.glsl                  (per-vertex HSV tint)

Known differences (also documented in the export README):
  * Lighting is a sun + flat world colour instead of the engine's sky/CSM/RT
    setup; the material blending itself is preserved.
  * Mixed layers (21-25) are baked on export from the same two source layers
    and equations; the Perlin noise is re-implemented on the CPU.
  * Tessellation displacement is not applied (the export is the undisplaced
    tessellated surface).
  * Grass is a static bake (no wind); SDF effects are proxy meshes because the
    engine raymarches them.
"""

import json
import math
import os
import struct
import sys

try:
    import bpy
    import numpy as np
    from mathutils import Matrix, Vector
except ModuleNotFoundError as exc:  # -------------------------------------------------
    # `bpy`/`mathutils` only exist inside Blender's bundled Python. Running this
    # file with a system interpreter (python3 octree_import.py) can never work,
    # so fail with the correct invocation instead of a bare traceback.
    _missing = getattr(exc, "name", "bpy")
    sys.stderr.write(
        "octree_import.py must run inside Blender (module '%s' is Blender-only).\n"
        "\n"
        "This file is a Blender setup script, not a standalone Python program.\n"
        "Run it with Blender 4.x instead:\n"
        "\n"
        "  blender --background --python %s -- %s\n"
        "\n"
        "or open Blender, switch to the Scripting workspace, load this file and\n"
        "press Run Script. The optional arguments after '--' are the sidecar\n"
        "(<name>.octree.json) and options such as --save-blend <file.blend> or\n"
        "--render <out.png>.\n"
        "\n"
        "A ready-to-render .blend is usually already next to this script (the\n"
        "engine writes it when exporting); you can open that directly.\n"
        % (_missing, os.path.basename(__file__) if "__file__" in globals()
           else "octree_import.py", "<name>.octree.json")
    )
    sys.exit(2)



# ---------------------------------------------------------------------------
# CLI / config discovery
# ---------------------------------------------------------------------------

def parse_args():
    argv = sys.argv
    args = argv[argv.index("--") + 1:] if "--" in argv else []
    opts = {
        "target": None,
        "render": None,
        "save_blend": None,
        "engine": "eevee",
        "samples": 64,
        "grass": True,
        "sdf": True,
        "light": True,
    }
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--render" and i + 1 < len(args):
            opts["render"] = args[i + 1]; i += 2
        elif a == "--save-blend" and i + 1 < len(args):
            opts["save_blend"] = args[i + 1]; i += 2
        elif a == "--engine" and i + 1 < len(args):
            opts["engine"] = args[i + 1].lower(); i += 2
        elif a == "--samples" and i + 1 < len(args):
            opts["samples"] = int(args[i + 1]); i += 2
        elif a == "--no-grass":
            opts["grass"] = False; i += 1
        elif a == "--no-sdf":
            opts["sdf"] = False; i += 1
        elif a == "--no-light":
            opts["light"] = False; i += 1
        elif not a.startswith("--"):
            opts["target"] = a; i += 1
        else:
            raise SystemExit("unknown option: " + a)
    return opts


def resolve_target(target):
    candidates = []
    if target:
        candidates.append(target)
    here = os.path.dirname(os.path.abspath(__file__))
    for directory in (here, os.getcwd()):
        try:
            for name in sorted(os.listdir(directory)):
                if name.endswith(".octree.json"):
                    candidates.append(os.path.join(directory, name))
        except OSError:
            pass
    for path in candidates:
        if path.endswith(".glb"):
            sidecar = path[:-4] + ".octree.json"
            if os.path.isfile(sidecar):
                return sidecar
        if os.path.isfile(path) and path.endswith(".json"):
            return path
    raise SystemExit("no .octree.json sidecar found; pass it as the first argument")


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    cfg["_dir"] = os.path.dirname(os.path.abspath(path))
    cfg["_config_path"] = os.path.abspath(path)
    return cfg


# ---------------------------------------------------------------------------
# GLB parsing: materialize embedded texture layers as UDIM tiles
# ---------------------------------------------------------------------------

def read_glb(path):
    with open(path, "rb") as handle:
        data = handle.read()
    magic, _version, _length = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67:
        raise RuntimeError("not a GLB file: " + path)
    offset = 12
    gltf = None
    binary = b""
    while offset + 8 <= len(data):
        chunk_len, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset:offset + chunk_len]
        offset += chunk_len
        if chunk_type == 0x4E4F534A:
            gltf = json.loads(chunk.decode("utf-8"))
        elif chunk_type == 0x004E4942:
            binary = chunk
    if gltf is None:
        raise RuntimeError("GLB has no JSON chunk")
    return gltf, binary


def udim_tile(index):
    return 1001 + (index % 10) + 10 * (index // 10)


def extract_tiles(cfg):
    gltf, binary = read_glb(os.path.join(cfg["_dir"], cfg["glb"]))
    views = gltf.get("bufferViews", [])
    tile_dir = os.path.join(cfg["_dir"], cfg["tilesDir"])
    os.makedirs(tile_dir, exist_ok=True)
    tiles = {}
    for image in gltf.get("images", []):
        name = image.get("name", "")
        parts = name.split("_")
        if len(parts) != 3 or parts[0] != "layer":
            continue
        try:
            layer = int(parts[1])
        except ValueError:
            continue
        kind = parts[2]
        if kind not in ("albedo", "normal", "rough", "ao"):
            continue
        view = views[image["bufferView"]]
        start = view.get("byteOffset", 0)
        blob = binary[start:start + view["byteLength"]]
        mime = image.get("mimeType", "image/jpeg")
        ext = ".png" if "png" in mime else ".jpg"
        path = os.path.join(tile_dir, "octree_%s.%d%s" % (kind, udim_tile(layer), ext))
        with open(path, "wb") as handle:
            handle.write(blob)
        tiles.setdefault(layer, {})[kind] = path
    return tiles


def load_udim(tiles, kind, used_layers):
    available = sorted(layer for layer in used_layers if kind in tiles.get(layer, {}))
    if not available:
        return None
    first = tiles[available[0]][kind]
    image = bpy.data.images.load(first, check_existing=False)
    image.name = "octree_" + kind
    image.source = "TILED"
    image.filepath = first
    image.colorspace_settings.name = "sRGB" if kind == "albedo" else "Non-Color"
    image.reload()
    return image


# ---------------------------------------------------------------------------
# Node helpers
# ---------------------------------------------------------------------------

class Nodes:
    def __init__(self, tree):
        self.tree = tree
        self.nodes = tree.nodes
        self.links = tree.links
        self._serial = 0

    def loc(self):
        self._serial += 1
        return (-1800 + (self._serial % 40) * 40, -300 + (self._serial // 40) * 120)

    def math(self, op, a, b=None, c=None, clamp=False):
        node = self.nodes.new("ShaderNodeMath")
        node.operation = op
        node.location = self.loc()
        inputs = (a, b, c)
        for i, value in enumerate(inputs):
            if value is None:
                break
            if isinstance(value, (int, float)):
                node.inputs[i].default_value = float(value)
            else:
                self.links.new(value, node.inputs[i])
        node.use_clamp = clamp
        return node.outputs[0]

    def vmath(self, op, a, b=None, scale=None):
        node = self.nodes.new("ShaderNodeVectorMath")
        node.operation = op
        node.location = self.loc()
        if isinstance(a, (tuple, list)):
            node.inputs[0].default_value = a
        else:
            self.links.new(a, node.inputs[0])
        if b is not None:
            if isinstance(b, (tuple, list)):
                node.inputs[1].default_value = b
            else:
                self.links.new(b, node.inputs[1])
        if scale is not None:
            if isinstance(scale, (int, float)):
                node.inputs[3].default_value = float(scale)
            else:
                self.links.new(scale, node.inputs[3])
        if op in ("DOT_PRODUCT", "DISTANCE", "LENGTH"):
            return node.outputs["Value"]
        return node.outputs["Vector"]

    def xyz(self, x, y, z):
        node = self.nodes.new("ShaderNodeCombineXYZ")
        node.location = self.loc()
        for i, value in enumerate((x, y, z)):
            if isinstance(value, (int, float)):
                node.inputs[i].default_value = float(value)
            else:
                self.links.new(value, node.inputs[i])
        return node.outputs[0]

    def sep(self, vec):
        node = self.nodes.new("ShaderNodeSeparateXYZ")
        node.location = self.loc()
        self.links.new(vec, node.inputs[0])
        return node.outputs[0], node.outputs[1], node.outputs[2]

    def attr(self, name, vector=False):
        node = self.nodes.new("ShaderNodeAttribute")
        node.location = self.loc()
        node.attribute_type = "GEOMETRY"
        node.attribute_name = name
        return node.outputs["Vector"] if vector else node.outputs["Fac"]

    def sample(self, image, uv):
        node = self.nodes.new("ShaderNodeTexImage")
        node.location = self.loc()
        node.image = image
        node.extension = "REPEAT"
        node.interpolation = "Linear"
        node.projection = "FLAT"
        self.links.new(uv, node.inputs["Vector"])
        return node


def clamp01(nb, value):
    return nb.math("MINIMUM", nb.math("MAXIMUM", value, 0.0), 1.0)


def lerp(nb, a, b, t):
    return nb.math("ADD", a, nb.math("MULTIPLY", nb.math("SUBTRACT", b, a), t))


def vlerp(nb, a, b, t):
    return nb.vmath("ADD", a, nb.vmath("SCALE", nb.vmath("SUBTRACT", b, a), scale=t))


def vscale(nb, v, s):
    return nb.vmath("SCALE", v, scale=s)


def engine_position(nb, blender_pos):
    """Blender/glTF (x, y, z) -> engine (x, z, -y) (Y-up right handed)."""
    x, y, z = nb.sep(blender_pos)
    return nb.xyz(x, z, nb.math("MULTIPLY", y, -1.0))


def blender_vector(nb, engine_vec):
    """Engine (x, y, z) -> Blender (x, -z, y)."""
    x, y, z = nb.sep(engine_vec)
    return nb.xyz(x, nb.math("MULTIPLY", z, -1.0), y)


# ---------------------------------------------------------------------------
# Baked per-face-corner attributes (the TCS/TES port)
# ---------------------------------------------------------------------------

DEFAULT_LAYER = {
    "triplanar": True,
    "normalFlipY": False,
    "normalSwapXZ": False,
    "roughnessFactor": 1.0,
    "aoFactor": 1.0,
    "useAO": True,
    "shininess": 32.0,
    "specularStrength": 0.5,
}


def _new_attr(mesh, name, kind, count, components):
    if name in mesh.attributes:
        mesh.attributes.remove(mesh.attributes[name])
    attribute = mesh.attributes.new(name, kind, "CORNER")
    return attribute


MATERIAL_PARAMS = [
    ("triplanar", "TRIPLANAR", 1.0),
    ("normalFlipY", "FLIPY", 0.0),
    ("normalSwapXZ", "SWAPXZ", 0.0),
    ("roughnessFactor", "ROUGH", 1.0),
    ("aoFactor", "AOFACT", 1.0),
    ("useAO", "USEAO", 1.0),
    ("shininess", "SHINE", 32.0),
    ("specularStrength", "SPEC", 0.5),
]


def build_corner_material_attributes(obj, cfg):
    """Compresses per-vertex texture indices into up to three unique slots per
    face (SolidTesc.glsl), bakes per-corner weights and face normals, and
    returns the material parameters that are constant across the used layers
    (varying ones are baked per face-corner instead)."""
    mesh = obj.data
    n_loops = len(mesh.loops)
    n_faces = len(mesh.polygons)
    n_verts = len(mesh.vertices)
    if n_loops == 0 or n_faces == 0:
        return {}

    loop_vert = np.empty(n_loops, dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", loop_vert)
    mat_attr = mesh.attributes.get("_OCTREE_MAT")
    if mat_attr is None:
        raise RuntimeError("terrain mesh has no _OCTREE_MAT attribute")
    raw = np.empty(n_verts * 4, dtype=np.float32)
    mat_attr.data.foreach_get("color", raw)
    vertex_mat = raw.reshape(-1, 4)
    corner_tex = np.rint(vertex_mat[loop_vert, 0]).astype(np.int32)

    loop_start = np.empty(n_faces, dtype=np.int32)
    loop_total = np.empty(n_faces, dtype=np.int32)
    mesh.polygons.foreach_get("loop_start", loop_start)
    mesh.polygons.foreach_get("loop_total", loop_total)
    if not np.all(loop_total == 3):
        raise RuntimeError("terrain mesh is not triangulated")

    c0 = corner_tex[loop_start]
    c1 = corner_tex[loop_start + 1]
    c2 = corner_tex[loop_start + 2]

    # SolidTesc.glsl: u0 = i0 if i0>=0; u1 if distinct; u2 if distinct.
    u0 = np.where(c0 >= 0, c0, -1)
    u1 = np.where((c1 >= 0) & (c1 != u0), c1, -1)
    u2 = np.where((c2 >= 0) & (c2 != u0) & (c2 != u1), c2, -1)
    slots = np.stack([np.maximum(u0, 0), np.maximum(u1, 0), np.maximum(u2, 0)], axis=1)
    raw_slots = np.stack([u0, u1, u2], axis=1)
    idx = np.stack([c0, c1, c2], axis=1)

    weights = np.zeros((n_faces, 3, 3), dtype=np.float32)
    for s in range(3):
        weights[:, s, :] = (idx == raw_slots[:, s:s + 1]) & (idx >= 0)

    face_of_loop = np.repeat(np.arange(n_faces, dtype=np.int32), 3)
    corner_of_loop = np.arange(n_loops, dtype=np.int32) - loop_start[face_of_loop]

    count = int(cfg.get("materialCount", 28))
    table = [dict(DEFAULT_LAYER) for _ in range(max(count, 1))]
    for entry in cfg.get("materials", []):
        i = int(entry.get("index", -1))
        if 0 <= i < len(table):
            table[i] = entry

    used = sorted({int(v) for v in slots.reshape(-1) if 0 <= v < len(table)})
    if not used:
        used = [0]

    def param_array(key, default):
        return np.array([float(table[i].get(key, default)) for i in range(len(table))],
                        dtype=np.float32)

    constants = {}
    for key, attr_name, default in MATERIAL_PARAMS:
        values = [float(table[i].get(key, default)) for i in used]
        if max(values) - min(values) <= 1e-9:
            constants[attr_name] = values[0]
            continue
        values_by_layer = param_array(key, default)
        for s in range(3):
            slot_index = np.clip(slots[:, s], 0, len(table) - 1)
            attribute = _new_attr(mesh, "_%s_%d" % (attr_name, s), "FLOAT", n_loops, 1)
            attribute.data.foreach_set("value", values_by_layer[slot_index][face_of_loop])

    for s in range(3):
        attribute = _new_attr(mesh, "_TEXINDEX_%d" % s, "FLOAT", n_loops, 1)
        attribute.data.foreach_set("value", slots[:, s].astype(np.float32)[face_of_loop])
        attribute = _new_attr(mesh, "_TEXWEIGHT_%d" % s, "FLOAT", n_loops, 1)
        gathered = weights[:, s, :].reshape(-1)[
            (face_of_loop * 3 + corner_of_loop).astype(np.int64)]
        attribute.data.foreach_set("value", gathered)

    face_normals = np.empty(n_faces * 3, dtype=np.float32)
    mesh.polygons.foreach_get("normal", face_normals)
    attribute = _new_attr(mesh, "_FACENORMAL", "FLOAT_VECTOR", n_loops, 3)
    attribute.data.foreach_set("vector",
                               face_normals.reshape(-1, 3)[face_of_loop].reshape(-1))
    return constants


# ---------------------------------------------------------------------------
# Terrain material
# ---------------------------------------------------------------------------

def build_terrain_material(name, cfg, tiles, used_layers, constants):
    threshold = float(cfg.get("triplanar", {}).get("threshold", 0.12))
    exponent = float(cfg.get("triplanar", {}).get("exponent", 1.0))

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.use_backface_culling = False
    tree = mat.node_tree
    tree.nodes.clear()
    nb = Nodes(tree)
    output = nb.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nb.nodes.new("ShaderNodeBsdfPrincipled")
    nb.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    images = {}
    for kind in ("albedo", "normal", "rough", "ao"):
        image = load_udim(tiles, kind, used_layers)
        if image is not None:
            images[kind] = image

    geo = nb.nodes.new("ShaderNodeNewGeometry")
    pos = engine_position(nb, geo.outputs["Position"])
    normal = engine_position(nb, geo.outputs["Normal"])
    face = engine_position(nb, nb.attr("_FACENORMAL", vector=True))

    # Triplanar weights: abs(geomN), thresholded, exponent, renormalized.
    abs_face = nb.vmath("ABSOLUTE", face)
    wt = nb.vmath("MAXIMUM", nb.vmath("SUBTRACT", abs_face,
                                      (threshold, threshold, threshold)),
                  (0.0, 0.0, 0.0))
    if exponent > 1.0:
        wt = nb.vmath("POWER", wt, (exponent, exponent, exponent))
    wt_sum = nb.vmath("DOT_PRODUCT", wt, (1.0, 1.0, 1.0))
    wt = nb.vmath("SCALE", wt, scale=nb.math("DIVIDE", 1.0,
                                             nb.math("MAXIMUM", wt_sum, 1e-6)))
    wx, wy, wz = nb.sep(wt)

    fnx, fny, fnz = nb.sep(face)
    px, py, pz = nb.sep(pos)

    # Engine triplanar UVs (world position, signed by the face normal).
    neg_pz = nb.math("MULTIPLY", pz, -1.0)
    neg_py = nb.math("MULTIPLY", py, -1.0)
    neg_px = nb.math("MULTIPLY", px, -1.0)
    def axis_sign(component):
        # +1 when the face normal component is >= 0 (the shader tests >= 0).
        return nb.math("SUBTRACT",
                       nb.math("MULTIPLY", nb.math("GREATER_THAN", component, -1e-12), 2.0),
                       1.0)

    uvx = nb.xyz(nb.math("MULTIPLY", neg_pz, axis_sign(fnx)), neg_py, 0.0)
    uvy = nb.xyz(px, nb.math("MULTIPLY", pz, axis_sign(fny)), 0.0)
    uvz = nb.xyz(nb.math("MULTIPLY", px, axis_sign(fnz)), neg_py, 0.0)

    # Surface frame for transformProjectionNormal (surface normal = vertex N).
    nx, ny, nz = nb.sep(normal)
    use_y = nb.math("LESS_THAN", nb.math("ABSOLUTE", ny), 0.999)
    up = nb.xyz(lerp(nb, 1.0, 0.0, use_y), lerp(nb, 0.0, 1.0, use_y), 0.0)
    tangent = nb.vmath("NORMALIZE", nb.vmath("CROSS_PRODUCT", up, normal))
    bitangent = nb.vmath("CROSS_PRODUCT", normal, tangent)

    vertex_uv = nb.nodes.new("ShaderNodeUVMap")
    vertex_uv.uv_map = "UVMap"

    def tiles_uv(uv, index):
        frac = nb.vmath("FRACTION", uv)
        tile_u = nb.math("MODULO", index, 10.0)
        tile_v = nb.math("FLOOR", nb.math("DIVIDE", index, 10.0))
        return nb.vmath("ADD", frac, nb.xyz(tile_u, tile_v, 0.0))

    def param(key, slot):
        if key in constants:
            return constants[key]
        return nb.attr("_%s_%d" % (key, slot))

    def slot_uvs(slot):
        tri = param("TRIPLANAR", slot)
        index = nb.math("ROUND", nb.attr("_TEXINDEX_%d" % slot))
        # TEXCOORD_0 is the per-vertex mapping data painted by the engine's
        # TextureCoordPainter: the (scaleU, scaleV) for the GPU projection
        # techniques, or the authored UV for Custom. There is no per-material
        # triplanar scale to apply.
        uv_u, uv_v, _ = nb.sep(vertex_uv.outputs["UV"])
        scale = nb.xyz(uv_u, uv_v, 1.0)
        tri_uvs = [tiles_uv(nb.vmath("MULTIPLY", uv, scale), index)
                   for uv in (uvx, uvy, uvz)]
        flat_uv = tiles_uv(vertex_uv.outputs["UV"], index)
        return [vlerp(nb, flat_uv, tri_uv, tri) for tri_uv in tri_uvs], index

    albedo_sum = None
    normal_sum = None
    spec_power_sum = None
    spec_strength_sum = None
    ao_sum = None

    for slot in range(3):
        weight = nb.attr("_TEXWEIGHT_%d" % slot)
        uvs, _index = slot_uvs(slot)

        if "albedo" in images:
            rgb = None
            for uv, w in zip(uvs, (wx, wy, wz)):
                sample = nb.sample(images["albedo"], uv).outputs["Color"]
                term = vscale(nb, sample, w)
                rgb = term if rgb is None else nb.vmath("ADD", rgb, term)
            term = vscale(nb, rgb, weight)
            albedo_sum = term if albedo_sum is None else nb.vmath("ADD", albedo_sum, term)

        if "normal" in images:
            flip = param("FLIPY", slot)
            swap = param("SWAPXZ", slot)
            proj = None
            for uv, w in zip(uvs, (wx, wy, wz)):
                color = nb.sample(images["normal"], uv).outputs["Color"]
                n = nb.vmath("SUBTRACT", nb.vmath("SCALE", color, scale=2.0), (1.0, 1.0, 1.0))
                n = vlerp(nb, n, nb.vmath("MULTIPLY", n, (1.0, -1.0, 1.0)), flip)
                sx, sy, sz = nb.sep(n)
                n = vlerp(nb, n, nb.xyz(sz, sy, sx), swap)
                term = vscale(nb, n, w)
                proj = term if proj is None else nb.vmath("ADD", proj, term)
            proj = nb.vmath("NORMALIZE", proj)
            tx, ty, tz = nb.sep(proj)
            world_n = nb.vmath("ADD",
                               nb.vmath("ADD", vscale(nb, tangent, tx),
                                        vscale(nb, bitangent, ty)),
                               vscale(nb, normal, tz))
            world_n = nb.vmath("NORMALIZE", world_n)
            term = vscale(nb, world_n, weight)
            normal_sum = term if normal_sum is None else nb.vmath("ADD", normal_sum, term)

        if "rough" in images or "ao" in images:
            sx = nb.math("GREATER_THAN", wx, wy)
            sz = nb.math("GREATER_THAN", wx, wz)
            dx = nb.math("MULTIPLY", sx, sz)
            dy = nb.math("MULTIPLY", nb.math("SUBTRACT", 1.0, dx),
                         nb.math("GREATER_THAN", wy, wz))
            dz = nb.math("SUBTRACT", nb.math("SUBTRACT", 1.0, dx), dy)
            dom = nb.vmath("ADD",
                           nb.vmath("ADD", vscale(nb, uvs[0], dx), vscale(nb, uvs[1], dy)),
                           vscale(nb, uvs[2], dz))
            if "rough" in images:
                rough = nb.sep(nb.sample(images["rough"], dom).outputs["Color"])[0]
                rough_k = param("ROUGH", slot)
                shine = param("SHINE", slot)
                power = lerp(nb, shine, 1.0, nb.math("MULTIPLY", rough, rough_k))
                term = nb.math("MULTIPLY", power, weight)
                spec_power_sum = term if spec_power_sum is None else nb.math(
                    "ADD", spec_power_sum, term)
                s = nb.math("MULTIPLY", param("SPEC", slot), weight)
                spec_strength_sum = s if spec_strength_sum is None else nb.math(
                    "ADD", spec_strength_sum, s)
            if "ao" in images:
                ao = nb.sep(nb.sample(images["ao"], dom).outputs["Color"])[0]
                ao_k = param("AOFACT", slot)
                use_ao = param("USEAO", slot)
                value = nb.math("MULTIPLY", use_ao, lerp(nb, 1.0, ao, ao_k))
                term = nb.math("MULTIPLY", value, weight)
                ao_sum = term if ao_sum is None else nb.math("ADD", ao_sum, term)

    base = albedo_sum if albedo_sum is not None else (0.6, 0.6, 0.6)
    if ao_sum is not None:
        base = vscale(nb, base, clamp01(nb, ao_sum))

    # Per-vertex HSV tint (SolidSurface.glsl 803-819). Blender's HSV uses a
    # 0..1 hue while the engine uses degrees.
    mat_attr = nb.nodes.new("ShaderNodeAttribute")
    mat_attr.attribute_type = "GEOMETRY"
    mat_attr.attribute_name = "_OCTREE_MAT"
    color_split = nb.nodes.new("ShaderNodeSeparateColor")
    color_split.mode = "RGB"
    nb.links.new(mat_attr.outputs["Color"], color_split.inputs[0])
    hx = color_split.outputs["Green"]
    hy = color_split.outputs["Blue"]
    hz = mat_attr.outputs["Alpha"]
    sep = nb.nodes.new("ShaderNodeSeparateColor")
    sep.mode = "HSV"
    nb.links.new(base, sep.inputs[0])
    comb = nb.nodes.new("ShaderNodeCombineColor")
    comb.mode = "HSV"
    nb.links.new(nb.math("MODULO", nb.math("ADD", sep.outputs[0],
                                           nb.math("DIVIDE", hx, 360.0)), 1.0),
                 comb.inputs[0])
    nb.links.new(clamp01(nb, nb.math("MULTIPLY", sep.outputs[1],
                                     nb.math("MULTIPLY", hy, 2.0))), comb.inputs[1])
    nb.links.new(nb.math("MULTIPLY", sep.outputs[2], nb.math("MULTIPLY", hz, 2.0)),
                 comb.inputs[2])
    nb.links.new(comb.outputs[0], bsdf.inputs["Base Color"])

    if spec_power_sum is not None:
        # Blinn-Phong exponent -> GGX roughness: alpha = sqrt(2 / (p + 2)).
        power = nb.math("MAXIMUM", spec_power_sum, 1.0)
        alpha = nb.math("SQRT", nb.math("DIVIDE", 2.0, nb.math("ADD", power, 2.0)))
        nb.links.new(clamp01(nb, alpha), bsdf.inputs["Roughness"])
    if spec_strength_sum is not None and "Specular IOR Level" in bsdf.inputs:
        nb.links.new(clamp01(nb, spec_strength_sum), bsdf.inputs["Specular IOR Level"])

    if normal_sum is not None:
        nb.links.new(blender_vector(nb, nb.vmath("NORMALIZE", normal_sum)),
                     bsdf.inputs["Normal"])

    return mat


# ---------------------------------------------------------------------------
# Grass blades (port of VegetationRenderer.vert / BladeGrass.glsl)
# ---------------------------------------------------------------------------

BLADE_COUNT = 12


def blade_hash(index):
    p = np.array([index * 17.13, index * 9.71, index * 5.37], dtype=np.float64)
    d = np.array([np.dot(p, [127.1, 311.7, 74.7]),
                  np.dot(p, [269.5, 183.3, 246.1]),
                  np.dot(p, [113.5, 271.9, 124.6])])
    h = np.mod(np.sin(d) * 43758.5453, 1.0)
    return np.array([h[0], h[1], h[2],
                     np.mod(h[0] * 7.13 + h[1] * 3.71 + index * 0.317, 1.0)])


def rotate_y(v, c, s):
    out = np.empty_like(v)
    out[..., 0] = c * v[..., 0] - s * v[..., 2]
    out[..., 1] = v[..., 1]
    out[..., 2] = s * v[..., 0] + c * v[..., 2]
    return out


def tilt_to_normal(v, n):
    """VegetationCommon.glsl tiltToNormal, vectorized over the leading axis."""
    k = np.stack([n[..., 2], np.zeros_like(n[..., 1]), -n[..., 0]], axis=-1)
    s = np.linalg.norm(k, axis=-1, keepdims=True)
    safe = s > 1e-4
    kk = np.where(safe, k / np.maximum(s, 1e-8), np.array([1.0, 0.0, 0.0]))
    c = np.clip(n[..., 1], -1.0, 1.0)[..., None]
    cross = np.cross(kk, v)
    dot = np.sum(kk * v, axis=-1, keepdims=True)
    tilted = v * c + cross * s + kk * dot * (1.0 - c)
    fallback = np.where(n[..., 1:2] >= 0.0, v, v * np.array([1.0, -1.0, 1.0]))
    return np.where(safe, tilted, fallback)


def grass_variation(world_pos):
    cell = np.floor(world_pos[:, [0, 2]] * 3.0)
    h = np.mod(np.sin(cell[:, 0] * 127.1 + cell[:, 1] * 311.7) * 43758.5453, 1.0)
    return (h - 0.5) * 0.22


def build_grass(cfg, scene_collection):
    data = cfg.get("grass", {})
    if int(data.get("count", 0)) == 0:
        return 0

    gltf, binary = read_glb(os.path.join(cfg["_dir"], cfg["glb"]))
    mesh_index = None
    for index, mesh in enumerate(gltf.get("meshes", [])):
        if mesh.get("name") == cfg["meshes"]["grass"]:
            mesh_index = index
            break
    if mesh_index is None:
        return 0

    prim = gltf["meshes"][mesh_index]["primitives"][0]
    accessors = gltf["accessors"]
    views = gltf["bufferViews"]

    def read_accessor(accessor_index, components):
        acc = accessors[accessor_index]
        view = views[acc["bufferView"]]
        offset = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        arr = np.frombuffer(binary, dtype=np.float32,
                            count=acc["count"] * components, offset=offset)
        return arr.reshape(acc["count"], components).astype(np.float64)

    positions = read_accessor(prim["attributes"]["POSITION"], 3)
    normals = read_accessor(prim["attributes"]["NORMAL"], 3)
    veg = read_accessor(prim["attributes"]["_VEG"], 4)
    variants = np.rint(veg[:, 0]).astype(np.int32)
    rotations = veg[:, 1]
    heights = veg[:, 2]

    billboard_scale = float(data.get("billboardScale", 10.0))
    palettes = data.get("variants", [])
    roots = np.array([p["root"] for p in palettes], dtype=np.float64)
    tips = np.array([p["tip"] for p in palettes], dtype=np.float64)

    # Blender -> engine (Y-up) space.
    pos_e = np.stack([positions[:, 0], positions[:, 2], -positions[:, 1]], axis=-1)
    nrm_e = np.stack([normals[:, 0], normals[:, 2], -normals[:, 1]], axis=-1)
    nrm_e /= np.maximum(np.linalg.norm(nrm_e, axis=-1, keepdims=True), 1e-8)

    total_verts = 0
    for variant in range(max(3, int(variants.max()) + 1)):
        mask = variants == variant
        m = int(np.sum(mask))
        if m == 0:
            continue
        p = pos_e[mask]
        nrm = nrm_e[mask]
        theta = rotations[mask] * (2.0 * math.pi)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        scale = billboard_scale * heights[mask]
        var_mix = grass_variation(p)[:, None]
        root = roots[variant] if variant < len(roots) else roots[0]
        tip = tips[variant] if variant < len(tips) else tips[0]

        verts = np.zeros((m, BLADE_COUNT, 3, 3), dtype=np.float64)
        norms = np.zeros((m, BLADE_COUNT, 3, 3), dtype=np.float64)
        cols = np.zeros((m, BLADE_COUNT, 3, 4), dtype=np.float64)
        cols[..., 3] = 1.0

        for b in range(BLADE_COUNT):
            bh = blade_hash(b)
            ang = b / float(BLADE_COUNT) * 2.0 * math.pi
            dir0 = np.tile([math.cos(ang), 0.0, math.sin(ang)], (m, 1))
            base_dir = rotate_y(dir0, cos_t, sin_t)
            jitter = (bh[2] - 0.5) * 0.9
            base_dir = rotate_y(base_dir, math.cos(jitter), math.sin(jitter))
            width_dir = np.stack([-base_dir[:, 2], np.zeros(m), base_dir[:, 0]], axis=-1)
            width_dir += np.array([1e-5, 0.0, 0.0])
            width_dir /= np.maximum(np.linalg.norm(width_dir, axis=-1, keepdims=True), 1e-8)
            lean_ang = bh[2] * 2.0 * math.pi
            lean_dir = rotate_y(np.tile([math.cos(lean_ang), 0.0, math.sin(lean_ang)], (m, 1)),
                                cos_t, sin_t)

            height_var = 0.70 + 0.50 * bh[0]
            lean_amt = 0.10 + 0.35 * bh[1]
            width_var = 0.75 + 0.50 * bh[3]
            blade_h = (0.48 + 0.22 * bh[0]) * height_var
            half_width = 0.05 * width_var
            lean_scale = 1.0 if b < 8 else 0.5

            if b >= 8:
                sat = float(b) - 8.0
                sat_ang = sat * 1.5707963 + 0.7853982 + (bh[1] - 0.5) * 0.6
                sat_r = 0.20 + 0.06 * bh[3]
                disc = rotate_y(np.tile([math.cos(sat_ang), 0.0, math.sin(sat_ang)], (m, 1)),
                                cos_t, sin_t) * sat_r
            else:
                disc = base_dir * (0.07 * (0.35 + 0.65 * bh[1]))

            base_left = disc - width_dir * half_width
            base_right = disc + width_dir * half_width
            tip_pos = (disc + np.stack([np.zeros(m), np.full(m, blade_h), np.zeros(m)], axis=-1)
                       + lean_dir * (lean_amt * lean_scale * blade_h)
                       + base_dir * (0.12 * blade_h * bh[3]))
            corner = np.stack([base_left, base_right, tip_pos], axis=1) * scale[:, None, None]
            corner = tilt_to_normal(corner, nrm[:, None, :])
            verts[:, b, :, :] = corner + p[:, None, :]

            for corner_index, (t, side) in enumerate(((0.0, -1.0), (0.0, 1.0), (1.0, 0.0))):
                albedo = root + (tip - root) * (t ** 1.25)
                albedo = albedo * (1.0 + var_mix[:, 0])[:, None]
                albedo = albedo * (1.0 - 0.18 * abs(side))
                cols[:, b, corner_index, :3] = albedo

        verts = verts.reshape(-1, 3)
        norms = norms.reshape(-1, 3)
        cols = cols.reshape(-1, 4)

        # Engine Y-up -> Blender Z-up.
        blender_verts = np.stack([verts[:, 0], -verts[:, 2], verts[:, 1]], axis=-1)

        mesh = bpy.data.meshes.new("GrassVariant%d" % variant)
        mesh.vertices.add(len(blender_verts))
        mesh.vertices.foreach_set("co", blender_verts.reshape(-1).astype(np.float32))
        loops = np.arange(len(blender_verts), dtype=np.int32)
        mesh.loops.add(len(loops))
        mesh.loops.foreach_set("vertex_index", loops)
        mesh.polygons.add(len(loops) // 3)
        mesh.polygons.foreach_set("loop_start", np.arange(0, len(loops), 3, dtype=np.int32))
        smooth = np.zeros(len(loops) // 3, dtype=np.bool_)
        mesh.polygons.foreach_set("use_smooth", smooth)
        mesh.update(calc_edges=True)

        color_attr = mesh.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
        color_attr.data.foreach_set("color", cols.reshape(-1).astype(np.float32))
        obj = bpy.data.objects.new("GrassVariant%d" % variant, mesh)
        scene_collection.objects.link(obj)

        mat = bpy.data.materials.new("GrassVariant%d" % variant)
        mat.use_nodes = True
        mat.use_backface_culling = False
        tree = mat.node_tree
        tree.nodes.clear()
        out = tree.nodes.new("ShaderNodeOutputMaterial")
        bsdf = tree.nodes.new("ShaderNodeBsdfPrincipled")
        vcol = tree.nodes.new("ShaderNodeVertexColor")
        vcol.layer_name = "Col"
        tree.links.new(vcol.outputs["Color"], bsdf.inputs["Base Color"])
        bsdf.inputs["Roughness"].default_value = 0.72
        tree.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        mesh.materials.append(mat)
        total_verts += len(blender_verts)

    return total_verts


# ---------------------------------------------------------------------------
# SDF proxy objects
# ---------------------------------------------------------------------------

def sdf_material(cfg, sdf_mat, tiles):
    mat = bpy.data.materials.new("SdfMaterial")
    mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes.clear()
    out = tree.nodes.new("ShaderNodeOutputMaterial")
    bsdf = tree.nodes.new("ShaderNodeBsdfPrincipled")
    tree.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    color = sdf_mat.get("baseColor", [1, 1, 1, 1])
    bsdf.inputs["Base Color"].default_value = (color[0], color[1], color[2], 1.0)
    bsdf.inputs["Roughness"].default_value = float(sdf_mat.get("roughness", 0.5))
    bsdf.inputs["Metallic"].default_value = float(sdf_mat.get("metallic", 0.0))
    emission = sdf_mat.get("emission", [0, 0, 0])
    strength = float(sdf_mat.get("emissionIntensity", 0.0))
    if strength > 0.0:
        bsdf.inputs["Emission Color"].default_value = (emission[0], emission[1],
                                                       emission[2], 1.0)
        bsdf.inputs["Emission Strength"].default_value = strength
    opacity = float(sdf_mat.get("opacity", 1.0))
    if opacity < 1.0:
        bsdf.inputs["Alpha"].default_value = opacity
        mat.blend_method = 'BLEND'
    layer = int(round(float(sdf_mat.get("textureLayer", -1.0))))
    if layer >= 0 and layer in tiles and "albedo" in tiles[layer]:
        image = load_udim(tiles, "albedo", [layer])
        if image is not None:
            tex = tree.nodes.new("ShaderNodeTexImage")
            tex.image = image
            tex.projection = 'BOX'
            if hasattr(tex, 'projection_blend'):
                tex.projection_blend = 0.3
            coord = tree.nodes.new("ShaderNodeTexCoord")
            mapping = tree.nodes.new("ShaderNodeMapping")
            tiling = 1.0 / max(1.0, float(sdf_mat.get("textureTiling", 64.0)))
            mapping.inputs["Scale"].default_value = (tiling, tiling, tiling)
            tree.links.new(coord.outputs["Object"], mapping.inputs["Vector"])
            tree.links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
            tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def make_proxy(definition, material):
    prim = int(definition.get("type", 0))
    p0 = definition.get("params0", [0.0, 0.0, 0.0, 0.0])
    p1 = definition.get("params1", [0.0, 0.0, 0.0, 0.0])

    if prim == 1:  # Box
        bpy.ops.mesh.primitive_cube_add(size=2.0)
    elif prim == 6:  # Torus
        bpy.ops.mesh.primitive_torus_add(major_radius=max(p0[0], 0.01),
                                         minor_radius=max(p0[1], 0.01))
    elif prim == 5:  # Cone
        bpy.ops.mesh.primitive_cone_add(radius1=max(p0[0], 0.01),
                                        depth=max(p0[1], 0.01))
    elif prim == 4:  # Cylinder
        bpy.ops.mesh.primitive_cylinder_add(radius=max(p0[0], 0.01),
                                            depth=max(2.0 * p0[1], 0.01))
    elif prim in (8,):  # Flame (tapered cone proxy)
        bpy.ops.mesh.primitive_cone_add(radius1=max(p0[0], 0.01),
                                        radius2=max(p1[0], 0.0),
                                        depth=max(p0[1], 0.01))
    elif prim == 7:  # Plane
        bpy.ops.mesh.primitive_plane_add(size=2.0)
    else:  # Sphere / Smoke / RoundedBox / Capsule / Grass proxies
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=1.0)

    obj = bpy.context.object
    if prim == 1:
        obj.scale = (max(p0[0], 0.01), max(p0[1], 0.01), max(p0[2], 0.01))
    elif prim in (0, 9):
        obj.scale = (max(p0[0], 0.01),) * 3
    elif prim == 2:  # RoundedBox: box proxy (rounding not reproduced)
        obj.scale = (max(p0[0], 0.01), max(p0[1], 0.01), max(p0[2], 0.01))
    elif prim == 3:  # Capsule: elongated sphere proxy
        obj.scale = (max(p0[0], 0.01), max(p0[0] + p0[1], 0.01), max(p0[0], 0.01))
    obj.data.materials.append(material)
    return obj


def build_sdf(cfg, scene_collection, tiles):
    sdf = cfg.get("sdf") or {}
    instances = sdf.get("instances", [])
    definitions = sdf.get("definitions", [])
    materials = sdf.get("materials", [])
    if not instances or not definitions:
        return 0, 0

    created = 0
    material_cache = {}
    for inst in instances:
        def_idx = int(inst.get("defIdx", 0))
        mat_idx = int(inst.get("matIdx", 0))
        if def_idx >= len(definitions):
            continue
        if mat_idx not in material_cache:
            material_cache[mat_idx] = sdf_material(
                cfg, materials[mat_idx] if mat_idx < len(materials) else {}, tiles)
        material = material_cache[mat_idx]
        template = make_proxy(definitions[def_idx], material)
        template.data.materials.append(material)

        pos = inst.get("position", [0.0, 0.0, 0.0])
        rot = inst.get("rotation", [0.0, 0.0, 0.0])
        scale = float(inst.get("scale", 1.0))
        hs = float(inst.get("heightScale", 1.0))
        rs = float(inst.get("radiusScale", 1.0))
        rotation = (Matrix.Rotation(rot[0], 4, 'X') @
                    Matrix.Rotation(rot[1], 4, 'Y') @
                    Matrix.Rotation(rot[2], 4, 'Z'))
        to_blender = Matrix.Rotation(math.radians(90.0), 4, 'X')
        world = (to_blender @ Matrix.Translation(Vector(pos)) @ rotation @
                 Matrix.Diagonal(Vector((scale * rs, scale * rs, scale * hs, 1.0))))
        link = template.copy()
        link.data = template.data
        link.matrix_world = world
        link.name = "SdfInstance%d" % created
        scene_collection.objects.link(link)
        created += 1
        bpy.data.objects.remove(template, do_unlink=True)
    return created, 0


# ---------------------------------------------------------------------------
# Scene setup
# ---------------------------------------------------------------------------

def setup_world_and_light(cfg, opts):
    scene = bpy.context.scene
    if opts["light"]:
        light = cfg.get("light", {})
        direction = Vector(light.get("direction", [-0.35, -0.86, -0.37]))
        color = light.get("color", [1.0, 1.0, 0.9])
        sun_data = bpy.data.lights.new("OctreeSun", type='SUN')
        sun_data.energy = 4.0
        sun_data.color = (color[0], color[1], color[2])
        sun = bpy.data.objects.new("OctreeSun", sun_data)
        scene.collection.objects.link(sun)
        # direction points from the light toward the scene (engine); a sun
        # shines along its local -Z. Convert engine Y-up -> Blender Z-up.
        dir_blender = Vector((direction[0], -direction[2], direction[1]))
        sun.rotation_euler = dir_blender.to_track_quat('-Z', 'Y').to_euler()

    world = scene.world if scene.world is not None else bpy.data.worlds.new("OctreeWorld")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.35, 0.45, 0.6, 1.0)
        bg.inputs["Strength"].default_value = 0.35


def engine_to_blender(v):
    """Engine (x, y, z) Y-up right-handed -> Blender (x, -z, y) Z-up."""
    return Vector((float(v[0]), -float(v[2]), float(v[1])))


def setup_camera(cfg, objects):
    """Create the scene camera: the engine's view when the sidecar carries it,
    otherwise an auto-framed three-quarter view of the meshes."""
    scene = bpy.context.scene
    cam_data = bpy.data.cameras.new("OctreeCamera")
    cam = bpy.data.objects.new("OctreeCamera", cam_data)
    scene.collection.objects.link(cam)

    info = cfg.get("camera") or {}
    if info.get("position") is not None:
        pos = engine_to_blender(info["position"])
        target = engine_to_blender(info.get("target", [0.0, 0.0, 0.0]))
        cam.location = pos
        cam.rotation_euler = (target - pos).to_track_quat('-Z', 'Y').to_euler()
        fov = float(info.get("fovY", 60.0))
        cam_data.sensor_fit = 'VERTICAL'
        cam_data.angle = math.radians(fov)
        cam_data.clip_start = 0.1
        cam_data.clip_end = max(1000.0, (target - pos).length * 20.0)
        print("[octree-import] camera from engine: pos=%s fovY=%.1f" % (tuple(pos), fov))
    else:
        mins = Vector((1e30, 1e30, 1e30))
        maxs = Vector((-1e30, -1e30, -1e30))
        for obj in objects:
            if obj.type != 'MESH':
                continue
            for corner in obj.bound_box:
                p = obj.matrix_world @ Vector(corner)
                mins = Vector((min(mins[i], p[i]) for i in range(3)))
                maxs = Vector((max(maxs[i], p[i]) for i in range(3)))
        if mins.x > maxs.x:
            return
        center = (mins + maxs) * 0.5
        size = max((maxs - mins).length, 1.0)
        cam.location = center + Vector((0.55, -0.85, 0.5)) * size
        cam.rotation_euler = (center - cam.location).to_track_quat('-Z', 'Y').to_euler()
        cam_data.lens = 40.0
        cam_data.clip_end = size * 10.0
        print("[octree-import] camera auto-framed (no engine camera in the export)")

    scene.camera = cam


def configure_render(opts):
    scene = bpy.context.scene
    if opts["engine"] == "cycles":
        scene.render.engine = 'CYCLES'
        scene.cycles.samples = int(opts["samples"])
        scene.cycles.use_denoising = False
        scene.cycles.device = 'CPU'
    else:
        scene.render.engine = 'BLENDER_EEVEE'
        scene.eevee.taa_render_samples = int(opts["samples"])
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    opts = parse_args()
    config_path = resolve_target(opts["target"])
    cfg = load_config(config_path)
    print("[octree-import] config:", config_path)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.name = "Octree Export"

    glb_path = os.path.join(cfg["_dir"], cfg["glb"])
    if not os.path.isfile(glb_path):
        raise SystemExit("glb not found: " + glb_path)
    bpy.ops.import_scene.gltf(filepath=glb_path)
    print("[octree-import] imported", glb_path)

    tiles = extract_tiles(cfg)
    used_layers = sorted(tiles.keys())

    terrain = bpy.data.objects.get(cfg["meshes"]["terrain"])
    water = bpy.data.objects.get(cfg["meshes"]["water"])
    grass = bpy.data.objects.get(cfg["meshes"]["grass"])

    if terrain is not None and terrain.type == 'MESH':
        constants = build_corner_material_attributes(terrain, cfg)
        material = build_terrain_material("OctreeTerrain", cfg, tiles, used_layers,
                                          constants)
        terrain.data.materials.clear()
        terrain.data.materials.append(material)
        for poly in terrain.data.polygons:
            poly.material_index = 0

    if water is not None and water.type == 'MESH':
        mat = bpy.data.materials.new("OctreeWater")
        mat.use_nodes = True
        tree = mat.node_tree
        tree.nodes.clear()
        out = tree.nodes.new("ShaderNodeOutputMaterial")
        bsdf = tree.nodes.new("ShaderNodeBsdfPrincipled")
        water_cfg = cfg.get("water", {})
        color = water_cfg.get("color", [0.1, 0.32, 0.45])
        bsdf.inputs["Base Color"].default_value = (color[0], color[1], color[2], 1.0)
        bsdf.inputs["Roughness"].default_value = 0.05
        if "Transmission Weight" in bsdf.inputs:
            bsdf.inputs["Transmission Weight"].default_value = 0.25
        bsdf.inputs["Alpha"].default_value = float(water_cfg.get("alpha", 0.72))
        mat.blend_method = 'BLEND'
        mat.use_backface_culling = False
        tree.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        water.data.materials.clear()
        water.data.materials.append(mat)

    grass_verts = 0
    if grass is not None:
        bpy.data.objects.remove(grass, do_unlink=True)
        if opts["grass"]:
            grass_verts = build_grass(cfg, scene.collection)
            print("[octree-import] grass verts:", grass_verts)

    sdf_count = 0
    if opts["sdf"]:
        sdf_count, _ = build_sdf(cfg, scene.collection, tiles)
        print("[octree-import] sdf instances:", sdf_count)

    setup_world_and_light(cfg, opts)
    setup_camera(cfg, [o for o in bpy.data.objects if o.type == 'MESH'])
    configure_render(opts)

    scene["octree_export_config"] = config_path
    scene["octree_export_stats"] = json.dumps(cfg.get("stats", {}))

    if opts["save_blend"]:
        # Pack the extracted texture tiles into the .blend so the saved scene
        # is self-contained (the tiles are only needed at import time).
        try:
            bpy.ops.file.pack_all()
            print("[octree-import] packed textures into the .blend")
        except Exception as exc:  # noqa: BLE001 - pack failures are non-fatal
            print("[octree-import] pack_all failed:", exc)
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(opts["save_blend"]))
        print("[octree-import] saved", opts["save_blend"])

    if opts["render"]:
        scene.render.filepath = os.path.abspath(opts["render"])
        bpy.ops.render.render(write_still=True)
        print("[octree-import] rendered", opts["render"])

    print("[octree-import] done: terrain tris =",
          len(terrain.data.polygons) if terrain else 0,
          "grass verts =", grass_verts, "sdf =", sdf_count)


if __name__ == "__main__":
    main()
