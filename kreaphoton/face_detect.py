"""
Face Detailer backends (v1.6): YOLO face detection through ultralytics and the
ArcFace identity gate through insightface. Both libraries are imported lazily
inside the functions and their model objects are cached once per process, so the
node pack loads (and every other node runs) without them. The pure helpers at
the top have no third-party imports and are unit-tested.

Model files (resolved through comfy's folder_paths, models_dir relative):
  ultralytics/bbox/face_yolov8m.pt   detector - REQUIRED (RuntimeError names the paths tried)
  insightface/models/buffalo_l/      ArcFace pack - optional (gate reports OFF when absent)
"""
import os

import torch

DEFAULT_FACE_MODEL = "face_yolov8m.pt"
ARCFACE_PACK = "buffalo_l"

_YOLO_CACHE = {}       # path -> ultralytics.YOLO
_GATE = None           # ArcFaceGate singleton


# ------------------------------------------------------------------ pure helpers
def _largest_face_index(faces):
    """Index of the face object with the largest bbox area (objects expose .bbox as
    [x0, y0, x1, y1]); None when the list is empty."""
    best, best_area = None, -1.0
    for i, f in enumerate(faces):
        x0, y0, x1, y1 = [float(v) for v in f.bbox[:4]]
        area = max(x1 - x0, 0.0) * max(y1 - y0, 0.0)
        if area > best_area:
            best, best_area = i, area
    return best


def _to_bgr_uint8(image_bhwc: torch.Tensor):
    """comfy IMAGE (1, H, W, C) float 0..1 -> numpy (H, W, 3) uint8 BGR (what both
    ultralytics and insightface expect from a cv2-style array)."""
    img = image_bhwc[0, ..., :3].detach().float().clamp(0.0, 1.0).cpu()
    rgb = (img * 255.0 + 0.5).to(torch.uint8).numpy()
    return rgb[..., ::-1].copy()


def _bbox_folder_candidates(models_dir: str, registered):
    """Directories to look for the detector in, in order: comfy-registered
    'ultralytics_bbox' folders (Impact Subpack registers it), then the conventional
    models/ultralytics/bbox, then models/ultralytics."""
    out = list(registered or [])
    for sub in (("ultralytics", "bbox"), ("ultralytics",)):
        p = os.path.join(models_dir, *sub)
        if p not in out:
            out.append(p)
    return out


# ------------------------------------------------------------------ YOLO detector
def yolo_model_path(model_name: str = DEFAULT_FACE_MODEL) -> str:
    import folder_paths  # top-level ComfyUI module
    try:
        registered = folder_paths.get_folder_paths("ultralytics_bbox")
    except Exception:
        registered = []
    tried = []
    for d in _bbox_folder_candidates(folder_paths.models_dir, registered):
        p = os.path.join(d, model_name)
        tried.append(p)
        if os.path.isfile(p):
            return p
    raise RuntimeError("KreaPhoton Face Detailer: detector %s not found; tried: %s. Download it from "
                       "https://huggingface.co/Bingsu/adetailer into models/ultralytics/bbox/"
                       % (model_name, ", ".join(tried)))


def _yolo(path: str):
    m = _YOLO_CACHE.get(path)
    if m is None:
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError("KreaPhoton Face Detailer needs the `ultralytics` package "
                               "(pip install ultralytics): %s" % e)
        m = YOLO(path)
        _YOLO_CACHE[path] = m
    return m


def detect_faces(image_bhwc: torch.Tensor, *, model_name: str = DEFAULT_FACE_MODEL, threshold: float = 0.25):
    """(boxes [(x0, y0, x1, y1) px], confs) for one image (1, H, W, C). `threshold` is
    the detector's own confidence cut - the node filters again with bbox_threshold, so
    this stays low."""
    model = _yolo(yolo_model_path(model_name))
    bgr = _to_bgr_uint8(image_bhwc)
    results = model.predict(bgr, conf=float(threshold), verbose=False)
    boxes, confs = [], []
    for r in results:
        if r.boxes is None:
            continue
        xyxy = r.boxes.xyxy.detach().float().cpu()
        conf = r.boxes.conf.detach().float().cpu()
        for k in range(int(xyxy.shape[0])):
            boxes.append(tuple(float(v) for v in xyxy[k].tolist()))
            confs.append(float(conf[k]))
    return boxes, confs


# ------------------------------------------------------------------ ArcFace gate
class ArcFaceGate:
    """Identity similarity through insightface (buffalo_l, CPU onnxruntime). `available`
    is False - with `reason` - when the package or the model pack is missing; the node
    then runs with the gate OFF instead of failing."""

    def __init__(self, pack: str = ARCFACE_PACK):
        self.available = False
        self.reason = ""
        self._app = None
        try:
            import folder_paths  # top-level ComfyUI module
            from insightface.app import FaceAnalysis
            root = os.path.join(folder_paths.models_dir, "insightface")
            pack_dir = os.path.join(root, "models", pack)
            if not os.path.isdir(pack_dir):
                raise RuntimeError("model pack %s not found at %s" % (pack, pack_dir))
            app = FaceAnalysis(name=pack, root=root, providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1, det_size=(640, 640))
            self._app = app
            self.available = True
        except Exception as e:  # ImportError, missing pack, onnxruntime trouble
            self.reason = "%s: %s" % (type(e).__name__, e)

    def embed(self, image_bhwc: torch.Tensor):
        """L2-normalised embedding of the largest face in the image, None when the
        detector finds no face (the caller scores that as 0.0 - a destroyed face)."""
        if not self.available:
            return None
        faces = self._app.get(_to_bgr_uint8(image_bhwc))
        i = _largest_face_index(faces)
        if i is None:
            return None
        emb = torch.as_tensor(faces[i].normed_embedding, dtype=torch.float32)
        return emb / emb.norm().clamp_min(1e-8)

    @staticmethod
    def sim(a, b) -> float:
        """Cosine similarity; 0.0 when either side has no face."""
        if a is None or b is None:
            return 0.0
        return float(torch.dot(a, b))


def get_gate() -> ArcFaceGate:
    global _GATE
    if _GATE is None:
        _GATE = ArcFaceGate()
    return _GATE
