# -*- coding: utf-8 -*-
"""KreaPhoton Save Image node (KREA2-NODES 2026-07-08).

Saves the IMAGE batch to an arbitrary folder on the server filesystem, with a
JS "Browse" folder picker (web/kreaphoton_save.js + the /kreaphoton/browse
endpoint in server_routes.py) and auto-generated unique filenames.

The module is split into PURE helpers (no ComfyUI, unit-tested in
tests/test_save.py) and the node class (torch/ComfyUI, lazy imports only so the
plain-assert test suite runs in an interpreter without a ComfyUI tree)."""
import json
import os
import string

# ---------------------------------------------------------------------------
# Pure helpers (no torch, no ComfyUI) - directly unit-tested
# ---------------------------------------------------------------------------

FORMAT_EXT = {"png": "png", "jpg": "jpg", "webp": "webp"}
_DEFAULT_PREFIX = "KreaPhoton"
# characters that are illegal in a Windows filename component
_ILLEGAL = set('<>:"/\\|?*') | {chr(c) for c in range(32)}


def ext_for_format(fmt):
    """Map a format widget value to a file extension, defaulting to png."""
    return FORMAT_EXT.get(str(fmt).lower(), "png")


def sanitize_prefix(prefix):
    """Strip path separators / illegal filename chars from the user prefix.
    Empty or all-illegal input falls back to the default prefix."""
    cleaned = "".join(c for c in str(prefix) if c not in _ILLEGAL).strip().rstrip(". ")
    return cleaned or _DEFAULT_PREFIX


def make_timestamp(dt):
    """`YYYYmmdd-HHMMSS-mmm` from a datetime (millisecond precision so two runs
    inside the same second still differ)."""
    return dt.strftime("%Y%m%d-%H%M%S-") + "%03d" % (dt.microsecond // 1000)


def build_filename(prefix, timestamp, index, ext):
    """`{prefix}_{timestamp}_{NNNNN}.{ext}` - the canonical unique name."""
    return "%s_%s_%05d.%s" % (prefix, timestamp, index, ext)


def next_free_path(folder, prefix, timestamp, index, ext):
    """Absolute path for (prefix, timestamp, index, ext), bumping `index` until
    the file does not exist. Returns (path, final_index) so the caller can
    continue the batch counter past any collision."""
    n = int(index)
    while True:
        path = os.path.join(folder, build_filename(prefix, timestamp, n, ext))
        if not os.path.exists(path):
            return path, n
        n += 1


def list_drives():
    """Existing Windows drive roots (`C:\\`, `D:\\`, ...). Empty on POSIX."""
    if os.name != "nt":
        return []
    return ["%s:\\" % d for d in string.ascii_uppercase if os.path.exists("%s:\\" % d)]


def list_dirs(path):
    """Directory listing for the Browse modal. `path` empty -> the drive roots
    (Windows) or `/` (POSIX). Returns only sub-directories (never files), each
    as {name, path}. `parent` is None at the top of the tree.

    Errors (missing dir, permission denied) degrade to an empty listing with an
    `error` message rather than raising - the endpoint must never 500."""
    path = (path or "").strip()
    if not path:
        drives = list_drives()
        if drives:  # Windows: top level is the set of drives
            return {"path": "", "parent": None,
                    "dirs": [{"name": d, "path": d} for d in drives]}
        path = os.path.abspath(os.sep)  # POSIX: start at /

    path = os.path.abspath(path)
    parent = os.path.dirname(path)
    if parent == path:  # filesystem root -> parent is the drive list ("" sentinel)
        parent = "" if os.name == "nt" else None

    dirs = []
    error = None
    try:
        with os.scandir(path) as it:
            for entry in it:
                try:
                    if entry.is_dir():
                        dirs.append({"name": entry.name, "path": entry.path})
                except OSError:
                    continue  # unreadable entry - skip
    except OSError as exc:
        error = str(exc)
    dirs.sort(key=lambda d: d["name"].lower())
    out = {"path": path, "parent": parent, "dirs": dirs}
    if error:
        out["error"] = error
    return out


def exif_bytes(metadata):
    """EXIF blob carrying the workflow/prompt JSON in ImageDescription (0x010e)
    - the dependency-free way to tag JPEG/WebP (PNG uses text chunks instead)."""
    from PIL import Image
    exif = Image.Exif()
    exif[0x010e] = metadata if isinstance(metadata, str) else json.dumps(metadata)
    return exif.tobytes()


def save_pil(img, path, fmt, quality=90, metadata=None):
    """Write a PIL image to `path` in `fmt` with embedded metadata.
    PNG -> tEXt chunks (drag-drop-reloadable in ComfyUI); JPEG/WebP -> EXIF."""
    ext = ext_for_format(fmt)
    if ext == "png":
        pnginfo = None
        if metadata:
            from PIL.PngImagePlugin import PngInfo
            pnginfo = PngInfo()
            for k, v in metadata.items():
                pnginfo.add_text(k, v if isinstance(v, str) else json.dumps(v))
        img.save(path, format="PNG", pnginfo=pnginfo, compress_level=4)
        return
    params = {}
    if metadata:
        params["exif"] = exif_bytes(metadata)
    if ext == "jpg":
        img.save(path, format="JPEG", quality=int(quality), **params)
    else:
        img.save(path, format="WEBP", quality=int(quality), **params)


# ---------------------------------------------------------------------------
# The node (torch + lazy ComfyUI imports)
# ---------------------------------------------------------------------------

_FOLDER_TOOLTIP = ("Absolute folder on the server to save into (use the Browse button). "
                   "Empty = the standard ComfyUI output/ directory. Created if missing.")
_FORMAT_TOOLTIP = "png keeps the reloadable workflow; jpg/webp are smaller (quality applies)."
_META_TOOLTIP = ("Embed the workflow + prompt: PNG as text chunks (drag the PNG back into "
                 "ComfyUI to restore the graph), JPEG/WebP into EXIF. Off = clean file.")


def _tensor_to_pil(image):
    """A single ComfyUI IMAGE item [H,W,C] float 0..1 -> PIL.Image (RGB/L)."""
    import numpy as np
    from PIL import Image
    arr = np.clip(image.cpu().numpy() * 255.0, 0, 255).astype("uint8")
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[:, :, 0]
    return Image.fromarray(arr)


class KreaPhotonSaveImage:
    """Save the IMAGE batch to any server folder with unique timestamped names,
    embedded workflow metadata, and an on-node thumbnail."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "folder_path": ("STRING", {"default": "", "tooltip": _FOLDER_TOOLTIP}),
                "filename_prefix": ("STRING", {"default": _DEFAULT_PREFIX}),
                "format": (list(FORMAT_EXT.keys()), {"default": "png",
                                                     "tooltip": _FORMAT_TOOLTIP}),
                "quality": ("INT", {"default": 90, "min": 1, "max": 100, "step": 1}),
                "save_metadata": ("BOOLEAN", {"default": True, "tooltip": _META_TOOLTIP}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "KreaPhoton"

    def save(self, images, folder_path, filename_prefix, format, quality,
             save_metadata, prompt=None, extra_pnginfo=None):
        from datetime import datetime

        folder = (folder_path or "").strip()
        if not folder:
            import folder_paths  # lazy: ComfyUI top-level module
            folder = folder_paths.get_output_directory()
        os.makedirs(folder, exist_ok=True)

        prefix = sanitize_prefix(filename_prefix)
        ext = ext_for_format(format)
        timestamp = make_timestamp(datetime.now())

        metadata = None
        if save_metadata:
            metadata = {}
            if prompt is not None:
                metadata["prompt"] = prompt
            if extra_pnginfo is not None:
                metadata.update(extra_pnginfo)  # carries 'workflow'

        saved = []
        n = 1
        for image in images:
            path, n = next_free_path(folder, prefix, timestamp, n, ext)
            save_pil(_tensor_to_pil(image), path, format, quality, metadata)
            saved.append(path)
            n += 1

        return self._ui(images, saved)

    @staticmethod
    def _ui(images, saved):
        """On-node thumbnail: the chosen folder is usually outside ComfyUI's
        output/ (which /view serves), so mirror the batch into temp via the
        stock PreviewImage and show that. `saved` paths returned as text."""
        try:
            import nodes as comfy_nodes  # lazy: ComfyUI top-level module
            ui = comfy_nodes.PreviewImage().save_images(
                images, filename_prefix="KreaPhoton_saved")["ui"]
        except Exception:
            ui = {}
        ui["saved_paths"] = saved
        return {"ui": ui}
