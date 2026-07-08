# -*- coding: utf-8 -*-
"""Plain-assert test for kreaphoton/save.py (pure helpers + PIL round-trip).

No ComfyUI needed - only the pure name/dir/metadata helpers and Pillow.
Run: <embedded python> tests/test_save.py
"""
import importlib.util
import json
import os
import sys
import tempfile
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO_ROOT, relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    save = _load("kreaphoton_save", "kreaphoton/save.py")

    print("=" * 78)
    print("test_save: kreaphoton.save pure helpers")
    print("=" * 78)

    # --- ext_for_format: mapping + fallback ---
    assert save.ext_for_format("png") == "png"
    assert save.ext_for_format("jpg") == "jpg"
    assert save.ext_for_format("webp") == "webp"
    assert save.ext_for_format("PNG") == "png"          # case-insensitive
    assert save.ext_for_format("bmp") == "png"          # unknown -> png fallback
    print("[1] ext_for_format: mapping + fallback  OK")

    # --- sanitize_prefix: strips separators, never empty ---
    assert save.sanitize_prefix("KreaPhoton") == "KreaPhoton"
    assert save.sanitize_prefix("a/b\\c") == "abc"      # path separators gone
    assert save.sanitize_prefix('bad:*?"name') == "badname"
    assert save.sanitize_prefix("") == "KreaPhoton"     # empty -> default
    assert save.sanitize_prefix("///") == "KreaPhoton"  # all-illegal -> default
    assert save.sanitize_prefix("trail. ") == "trail"   # trailing dot/space trimmed
    print("[2] sanitize_prefix: illegal chars stripped, never empty  OK")

    # --- make_timestamp: millisecond precision, sortable ---
    dt = datetime(2026, 7, 8, 11, 45, 30, 812000)
    assert save.make_timestamp(dt) == "20260708-114530-812", save.make_timestamp(dt)
    print("[3] make_timestamp: %s  OK" % save.make_timestamp(dt))

    # --- build_filename: canonical name ---
    name = save.build_filename("KreaPhoton", "20260708-114530-812", 1, "png")
    assert name == "KreaPhoton_20260708-114530-812_00001.png", name
    print("[4] build_filename: %s  OK" % name)

    # --- next_free_path: collision bump ---
    with tempfile.TemporaryDirectory() as d:
        ts = "20260708-114530-812"
        p1, n1 = save.next_free_path(d, "KP", ts, 1, "png")
        assert n1 == 1
        open(p1, "w").close()                            # occupy index 1
        p2, n2 = save.next_free_path(d, "KP", ts, 1, "png")
        assert n2 == 2, n2                               # bumped past the taken index
        assert p1 != p2 and p2.endswith("_00002.png")
        print("[5] next_free_path: collision bump 1 -> 2  OK")

    # --- list_dirs: only sub-directories, sorted, parent link ---
    with tempfile.TemporaryDirectory() as d:
        os.mkdir(os.path.join(d, "zeta"))
        os.mkdir(os.path.join(d, "alpha"))
        open(os.path.join(d, "afile.txt"), "w").close()  # a file - must NOT appear
        data = save.list_dirs(d)
        names = [x["name"] for x in data["dirs"]]
        assert names == ["alpha", "zeta"], names         # sorted, file excluded
        assert data["path"] == os.path.abspath(d)
        assert data["parent"] == os.path.dirname(os.path.abspath(d))
        # navigating into a child returns a parent link back
        child = save.list_dirs(os.path.join(d, "alpha"))
        assert child["parent"] == os.path.abspath(d)
        print("[6] list_dirs: dirs-only, sorted, parent link  OK")

    # --- list_dirs: missing path degrades to error, never raises ---
    bad = save.list_dirs(os.path.join(tempfile.gettempdir(), "kp_nope_%d" % os.getpid()))
    assert "error" in bad and bad["dirs"] == []
    print("[7] list_dirs: missing path -> error field, no raise  OK")

    # --- empty path -> drives (Windows) / root (POSIX), no raise ---
    top = save.list_dirs("")
    assert top["parent"] is None and isinstance(top["dirs"], list)
    if os.name == "nt":
        assert top["dirs"] and all(x["path"].endswith(":\\") for x in top["dirs"])
    print("[8] list_dirs: empty path -> top of tree  OK")

    # --- PIL round-trip: PNG text chunk + JPEG/WebP EXIF metadata ---
    try:
        from PIL import Image
    except ImportError:
        print("\n[skip] Pillow not present - metadata round-trip skipped")
        print("\ntest_save: ALL ASSERTS PASSED (metadata skipped)")
        return

    meta = {"prompt": {"1": {"class_type": "X"}}, "workflow": {"nodes": []}}
    img = Image.new("RGB", (8, 6), (120, 60, 30))
    with tempfile.TemporaryDirectory() as d:
        # PNG -> tEXt chunks readable back
        ppng = os.path.join(d, "a.png")
        save.save_pil(img, ppng, "png", metadata=meta)
        reop = Image.open(ppng)
        assert json.loads(reop.text["prompt"]) == meta["prompt"]
        assert json.loads(reop.text["workflow"]) == meta["workflow"]
        print("[9] save_pil PNG: prompt+workflow in text chunks  OK")

        # JPEG -> ImageDescription EXIF carries the JSON
        pjpg = os.path.join(d, "a.jpg")
        save.save_pil(img, pjpg, "jpg", quality=85, metadata=meta)
        desc = Image.open(pjpg).getexif().get(0x010e)
        assert json.loads(desc) == meta, desc
        print("[10] save_pil JPEG: metadata in EXIF ImageDescription  OK")

        # WebP -> same EXIF path
        pwebp = os.path.join(d, "a.webp")
        save.save_pil(img, pwebp, "webp", quality=85, metadata=meta)
        assert os.path.exists(pwebp) and os.path.getsize(pwebp) > 0
        print("[11] save_pil WebP: file written  OK")

        # metadata=None -> clean PNG (no prompt chunk)
        pclean = os.path.join(d, "clean.png")
        save.save_pil(img, pclean, "png", metadata=None)
        assert "prompt" not in Image.open(pclean).text
        print("[12] save_pil PNG: metadata=None -> no chunks  OK")

    print("\ntest_save: ALL ASSERTS PASSED")


if __name__ == "__main__":
    main()
