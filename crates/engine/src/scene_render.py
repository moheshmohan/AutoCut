"""AutoCut's scene renderer, run by Blender without its window.

AutoCut runs it for `generate.render3d` (a `kind: "scene"` entry in the provider list, see
docs/generation.md) as

    blender --background --factory-startup --disable-autoexec --python-exit-code 3 \
        [scene.blend] --python scene_render.py -- '<request as JSON>'

and imports the frames it writes (PNG or EXR, with alpha) as an image sequence. Blender is a
separate program here: this file is AutoCut's own code (MIT OR Apache-2.0), written for it, and
uses only Blender's documented Python API. Everything Blender-specific is inside `render` and the
functions it calls, so the request handling can be checked with any Python:

    python3 -I scene_render.py -- --check '<request as JSON>'

prints the request as it will be used (or `autocut-error: ...` and exit status 2).

A line `progress <done>/<total>` is printed after each frame is written; AutoCut's progress bar
follows it.
"""

import json
import math
import os
import sys

MAX_REQUEST = 64 * 1024
MAX_FRAME = 9999
MAX_SIDE = 16384
MAX_TEXTS = 32
MAX_TEXT = 1000
MAX_OBJECTS = 64
MAX_TRACK_BYTES = 16 * 1024 * 1024
MAX_TRACK_FRAMES = 100000
LIMIT = 1.0e6
ENGINES = ("eevee", "cycles", "workbench")
FORMATS = ("png", "exr")


class RequestError(ValueError):
    """A request (or camera track) that cannot be rendered."""


def _number(v, what, lo=-LIMIT, hi=LIMIT):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise RequestError(f"{what} must be a number")
    x = float(v)
    if not math.isfinite(x) or x < lo or x > hi:
        raise RequestError(f"{what} must be from {lo:g} to {hi:g}")
    return x


def _whole(v, what, lo, hi):
    if isinstance(v, bool) or not isinstance(v, int) or v < lo or v > hi:
        raise RequestError(f"{what} must be a whole number from {lo} to {hi}")
    return v


def _vector(v, what, n):
    if not isinstance(v, list) or len(v) != n:
        raise RequestError(f"{what} must be a list of {n} numbers")
    return [_number(x, what) for x in v]


def _text(v, what, most):
    if not isinstance(v, str) or len(v) > most or "\0" in v:
        raise RequestError(f"{what} must be text of at most {most} characters")
    return v


def _path(v, what, ending):
    p = _text(v, what, 4096)
    if not os.path.isabs(p) or not p.lower().endswith(ending):
        raise RequestError(f"{what} must be an absolute path to a {ending} file")
    return p


def normalize(req):
    """The request checked and with its defaults: what `render` uses."""
    if not isinstance(req, dict):
        raise RequestError("the request must be a JSON object")
    known = {
        "version", "output", "format", "width", "height", "fps", "frameStart", "frameEnd", "transparent",
        "cryptomatte", "engine", "samples", "device", "text", "texts", "objects", "camera", "seed", "scene",
    }
    unknown = sorted(set(req) - known)
    if unknown:
        raise RequestError("unknown request fields: " + ", ".join(unknown[:5]))
    if req.get("version", 1) != 1:
        raise RequestError("this script reads version 1 requests")
    out = {"version": 1}
    output = _text(req.get("output"), "output", 4096)
    if not os.path.isabs(output):
        raise RequestError("output must be an absolute path (the folder and the frames' name)")
    out["output"] = output
    fmt = req.get("format", "png")
    if fmt not in FORMATS:
        raise RequestError("format must be png or exr")
    out["format"] = fmt
    out["width"] = _whole(req.get("width"), "width", 16, MAX_SIDE)
    out["height"] = _whole(req.get("height"), "height", 16, MAX_SIDE)
    out["fps"] = _number(req.get("fps"), "fps", 1.0, 240.0)
    start = _whole(req.get("frameStart", 1), "frameStart", 0, MAX_FRAME)
    end = _whole(req.get("frameEnd", start), "frameEnd", 0, MAX_FRAME)
    if end < start:
        raise RequestError("frameEnd comes before frameStart")
    out["frameStart"], out["frameEnd"] = start, end
    for flag in ("transparent", "cryptomatte", "scene"):
        v = req.get(flag, False)
        if not isinstance(v, bool):
            raise RequestError(f"{flag} must be true or false")
        out[flag] = v
    if out["cryptomatte"] and fmt != "exr":
        raise RequestError("Cryptomatte passes are written as EXR: give format exr")
    engine = req.get("engine")
    if engine is not None and engine not in ENGINES:
        raise RequestError("engine must be eevee, cycles or workbench")
    out["engine"] = engine
    samples = req.get("samples")
    out["samples"] = None if samples is None else _whole(samples, "samples", 1, 65536)
    device = req.get("device", "gpu")
    if device not in ("gpu", "cpu"):
        raise RequestError("device must be gpu or cpu")
    out["device"] = device
    out["seed"] = _whole(req.get("seed", 0), "seed", 0, 2**31 - 1)
    text = req.get("text")
    out["text"] = None if text is None else _text(text, "text", MAX_TEXT)
    texts = req.get("texts", {})
    if not isinstance(texts, dict) or len(texts) > MAX_TEXTS:
        raise RequestError(f"texts names at most {MAX_TEXTS} text objects")
    out["texts"] = {_text(k, "a text object's name", 64): _text(v, "a text", MAX_TEXT) for k, v in texts.items()}
    objects = req.get("objects", {})
    if not isinstance(objects, dict) or len(objects) > MAX_OBJECTS:
        raise RequestError(f"objects names at most {MAX_OBJECTS} objects")
    out["objects"] = {}
    for name, o in objects.items():
        _text(name, "an object's name", 64)
        if not isinstance(o, dict) or set(o) - {"location", "rotation", "scale", "hide"}:
            raise RequestError(f"object {name}: location, rotation, scale and hide only")
        clean = {}
        if "location" in o:
            clean["location"] = _vector(o["location"], f"{name}.location", 3)
        if "rotation" in o:
            clean["rotation"] = _vector(o["rotation"], f"{name}.rotation", 3)
        if "scale" in o:
            s = o["scale"]
            clean["scale"] = [_number(s, f"{name}.scale")] * 3 if isinstance(s, (int, float)) and not isinstance(s, bool) else _vector(s, f"{name}.scale", 3)
        if "hide" in o:
            if not isinstance(o["hide"], bool):
                raise RequestError(f"{name}.hide must be true or false")
            clean["hide"] = o["hide"]
        out["objects"][name] = clean
    camera = req.get("camera")
    out["camera"] = None if camera is None else _path(camera, "camera", ".json")
    return out


def load_track(path):
    """A camera track: `{"sensorWidth"?: mm, "frames": [{"frame", "position", "rotation" (XYZ
    Euler, degrees) | "quaternion" (w, x, y, z), "focalLength"? (mm)}]}`, in metres on Blender's
    axes (Z up). The frames, checked."""
    try:
        size = os.path.getsize(path)
        if size > MAX_TRACK_BYTES:
            raise RequestError("the camera track is larger than 16 MiB")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        raise RequestError(f"the camera track cannot be read: {e}")
    if not isinstance(data, dict) or not isinstance(data.get("frames"), list):
        raise RequestError("the camera track has no frames list")
    frames = data["frames"]
    if not 1 <= len(frames) <= MAX_TRACK_FRAMES:
        raise RequestError(f"the camera track has 1 to {MAX_TRACK_FRAMES} frames")
    sensor = data.get("sensorWidth")
    track = {"sensorWidth": None if sensor is None else _number(sensor, "sensorWidth", 1.0, 200.0), "frames": []}
    for k, fr in enumerate(frames):
        if not isinstance(fr, dict):
            raise RequestError(f"camera frame {k} is not an object")
        one = {
            "frame": _whole(fr.get("frame"), f"camera frame {k}: frame", -1000000, 1000000),
            "position": _vector(fr.get("position"), f"camera frame {k}: position", 3),
        }
        if "quaternion" in fr:
            q = _vector(fr["quaternion"], f"camera frame {k}: quaternion", 4)
            if sum(x * x for x in q) < 1.0e-12:
                raise RequestError(f"camera frame {k}: the quaternion is zero")
            one["quaternion"] = q
        elif "rotation" in fr:
            one["rotation"] = _vector(fr["rotation"], f"camera frame {k}: rotation", 3)
        else:
            raise RequestError(f"camera frame {k} has no rotation or quaternion")
        if "focalLength" in fr:
            one["focalLength"] = _number(fr["focalLength"], f"camera frame {k}: focalLength", 1.0, 5000.0)
        track["frames"].append(one)
    return track


def request_from_argv(argv):
    """`(check, request)` from the arguments after `--`."""
    args = argv[argv.index("--") + 1:] if "--" in argv else []
    check = bool(args) and args[0] == "--check"
    if check:
        args = args[1:]
    if len(args) != 1:
        raise RequestError("give one request (JSON) after --")
    raw = args[0]
    if len(raw) > MAX_REQUEST:
        raise RequestError("the request is larger than 64 KiB")
    try:
        req = json.loads(raw)
    except ValueError as e:
        raise RequestError(f"the request is not JSON: {e}")
    return check, normalize(req)


def rate(fps):
    """Blender's whole rate and its base for `fps` (29.97 is 30 / 1.001)."""
    whole = max(1, int(math.ceil(fps - 1.0e-6)))
    return whole, whole / fps


# --- inside Blender ------------------------------------------------------------------------


def build_stage(bpy, scene, req):
    """A plain stage for a title when no scene file is given: the text, a camera, light."""
    for obj in list(scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    curve = bpy.data.curves.new("Title", "FONT")
    curve.body = req["text"] or ""
    curve.align_x = "CENTER"
    curve.align_y = "CENTER"
    curve.extrude = 0.06
    curve.bevel_depth = 0.008
    title = bpy.data.objects.new("Title", curve)
    scene.collection.objects.link(title)
    title.rotation_euler = (math.radians(90.0), 0.0, 0.0)
    material = bpy.data.materials.new("Title")
    material.diffuse_color = (0.92, 0.92, 0.92, 1.0)
    curve.materials.append(material)
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = 50.0
    camera = bpy.data.objects.new("Camera", cam_data)
    scene.collection.objects.link(camera)
    camera.location = (0.0, -8.0, 0.0)
    camera.rotation_euler = (math.radians(90.0), 0.0, 0.0)
    scene.camera = camera
    sun_data = bpy.data.lights.new("Sun", "SUN")
    sun_data.energy = 3.0
    sun = bpy.data.objects.new("Sun", sun_data)
    scene.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(50.0), 0.0, math.radians(30.0))
    # the title fills about four fifths of the frame's width (less of a tall frame's)
    bpy.context.view_layer.update()
    seen = 8.0 * 36.0 / 50.0 * min(1.0, req["width"] / req["height"])
    wide = max(title.dimensions.x, 1.0e-3)
    k = 0.8 * seen / wide
    title.scale = (k, k, k)
    # a slow turn and push-in over the shot
    start, end = req["frameStart"], req["frameEnd"]
    if end > start:
        for frame, turn, distance in ((start, -6.0, 8.3), (end, 6.0, 7.7)):
            title.rotation_euler = (math.radians(90.0), 0.0, math.radians(turn))
            title.keyframe_insert("rotation_euler", frame=frame)
            camera.location = (0.0, -distance, 0.0)
            camera.keyframe_insert("location", frame=frame)


def apply_texts(bpy, scene, req):
    texts = dict(req["texts"])
    if req["text"] is not None and req["scene"]:
        texts.setdefault("Text", req["text"])
    for name, body in texts.items():
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != "FONT":
            raise RequestError(f"the scene has no text object named {name}")
        obj.data.body = body


def apply_objects(bpy, scene, req):
    for name, o in req["objects"].items():
        obj = bpy.data.objects.get(name)
        if obj is None:
            raise RequestError(f"the scene has no object named {name}")
        if "location" in o:
            obj.location = o["location"]
        if "rotation" in o:
            obj.rotation_mode = "XYZ"
            obj.rotation_euler = [math.radians(x) for x in o["rotation"]]
        if "scale" in o:
            obj.scale = o["scale"]
        if "hide" in o:
            obj.hide_render = o["hide"]


def apply_track(bpy, scene, track):
    camera = scene.camera
    if camera is None:
        data = bpy.data.cameras.new("Tracked Camera")
        camera = bpy.data.objects.new("Tracked Camera", data)
        scene.collection.objects.link(camera)
        scene.camera = camera
    camera.animation_data_clear()
    camera.data.animation_data_clear()
    if track["sensorWidth"] is not None:
        camera.data.sensor_width = track["sensorWidth"]
    for fr in track["frames"]:
        frame = fr["frame"]
        camera.location = fr["position"]
        camera.keyframe_insert("location", frame=frame)
        if "quaternion" in fr:
            camera.rotation_mode = "QUATERNION"
            camera.rotation_quaternion = fr["quaternion"]
            camera.keyframe_insert("rotation_quaternion", frame=frame)
        else:
            camera.rotation_mode = "XYZ"
            camera.rotation_euler = [math.radians(x) for x in fr["rotation"]]
            camera.keyframe_insert("rotation_euler", frame=frame)
        if "focalLength" in fr:
            camera.data.lens = fr["focalLength"]
            camera.data.keyframe_insert("lens", frame=frame)


def set_engine(bpy, scene, req):
    engine = req["engine"]
    if engine == "cycles":
        scene.render.engine = "CYCLES"
        scene.cycles.seed = req["seed"]
        if req["samples"]:
            scene.cycles.samples = req["samples"]
        if req["device"] == "gpu":
            use_gpu(bpy, scene)
    elif engine == "workbench":
        scene.render.engine = "BLENDER_WORKBENCH"
    elif engine == "eevee":
        for name in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
            try:
                scene.render.engine = name
                break
            except TypeError:
                continue
    if req["samples"] and scene.render.engine.startswith("BLENDER_EEVEE"):
        scene.eevee.taa_render_samples = req["samples"]


def use_gpu(bpy, scene):
    """Cycles on the GPU where this computer has one (Metal, OptiX, CUDA, HIP, oneAPI)."""
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
    except (KeyError, AttributeError):
        print("autocut-note: Cycles renders on the CPU (its preferences are not there)", flush=True)
        return
    for kind in ("METAL", "OPTIX", "CUDA", "HIP", "ONEAPI"):
        try:
            prefs.compute_device_type = kind
        except TypeError:
            continue
        prefs.get_devices()
        devices = [d for d in prefs.devices if d.type == kind]
        if devices:
            for d in prefs.devices:
                d.use = d.type == kind
            scene.cycles.device = "GPU"
            print(f"autocut-note: Cycles renders on the GPU ({kind})", flush=True)
            return
    print("autocut-note: Cycles renders on the CPU (no GPU found)", flush=True)


def set_output(bpy, scene, req):
    r = scene.render
    r.resolution_x, r.resolution_y, r.resolution_percentage = req["width"], req["height"], 100
    r.fps, r.fps_base = rate(req["fps"])
    scene.frame_start, scene.frame_end = req["frameStart"], req["frameEnd"]
    r.film_transparent = req["transparent"]
    r.filepath = req["output"]
    r.use_file_extension = True
    r.use_overwrite = True
    r.use_placeholder = False
    s = r.image_settings
    if req["format"] == "exr":
        s.file_format = "OPEN_EXR"
        s.color_depth = "16"
        s.exr_codec = "ZIP"
    else:
        s.file_format = "PNG"
        s.color_depth = "8"
        s.compression = 15
    s.color_mode = "RGBA" if req["transparent"] else "RGB"


def add_cryptomatte(bpy, scene, req):
    """Cryptomatte passes, as a multilayer EXR a frame in a `cryptomatte` folder beside the
    frames (the frames AutoCut imports stay plain RGBA)."""
    layer = bpy.context.view_layer
    layer.use_pass_cryptomatte_object = True
    layer.use_pass_cryptomatte_material = True
    layer.use_pass_cryptomatte_asset = True
    folder = os.path.join(os.path.dirname(req["output"]), "cryptomatte")
    try:
        scene.use_nodes = True
        tree = scene.node_tree
        source = next((n for n in tree.nodes if n.bl_idname == "CompositorNodeRLayers"), None) or tree.nodes.new("CompositorNodeRLayers")
        out = tree.nodes.new("CompositorNodeOutputFile")
        out.base_path = os.path.join(folder, "frame_")
        out.format.file_format = "OPEN_EXR_MULTILAYER"
        out.format.exr_codec = "ZIP"
        out.layer_slots.clear()
        for socket in source.outputs:
            if socket.enabled and (socket.name == "Image" or socket.name.startswith("Crypto")):
                out.layer_slots.new(socket.name)
                tree.links.new(socket, out.inputs[socket.name])
    except (AttributeError, KeyError, TypeError, RuntimeError) as e:
        print(f"autocut-note: Cryptomatte passes were not written (this Blender's compositor differs: {e})", flush=True)


def render(bpy, req, track):
    scene = bpy.context.scene
    if not req["scene"]:
        build_stage(bpy, scene, req)
    apply_texts(bpy, scene, req)
    apply_objects(bpy, scene, req)
    if track is not None:
        apply_track(bpy, scene, track)
    if scene.camera is None:
        raise RequestError("the scene has no camera")
    set_output(bpy, scene, req)
    set_engine(bpy, scene, req)
    if req["cryptomatte"]:
        add_cryptomatte(bpy, scene, req)
    total = req["frameEnd"] - req["frameStart"] + 1
    done = [0]

    def written(*_):
        done[0] += 1
        print(f"progress {done[0]}/{total}", flush=True)

    bpy.app.handlers.render_write.append(written)
    bpy.ops.render.render(animation=True)


def main(argv=None):
    argv = sys.argv if argv is None else argv
    try:
        check, req = request_from_argv(argv)
        track = load_track(req["camera"]) if req["camera"] else None
    except RequestError as e:
        print(f"autocut-error: {e}", file=sys.stderr, flush=True)
        return 2
    if check:
        print(json.dumps({"request": req, "cameraFrames": len(track["frames"]) if track else 0, "rate": rate(req["fps"])}, sort_keys=True))
        return 0
    try:
        import bpy
    except ImportError:
        print("autocut-error: this script runs inside Blender (bpy is not there)", file=sys.stderr, flush=True)
        return 2
    try:
        render(bpy, req, track)
    except RequestError as e:
        print(f"autocut-error: {e}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    code = main()
    if code:
        sys.exit(code)
