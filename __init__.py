"""
ComfyUI-KreaPhoton
Photorealism-focused sampling nodes for Krea 2 Turbo.
"""
from .kreaphoton.nodes import (NODE_CLASS_MAPPINGS as _NODES,
                               NODE_DISPLAY_NAME_MAPPINGS as _NAMES)

# Register server-side routes for the Save Image folder picker (guarded: a bare
# import without a running ComfyUI server is a no-op, see server_routes.py).
from .kreaphoton import server_routes as _server_routes  # noqa: F401

NODE_CLASS_MAPPINGS = {**_NODES}
NODE_DISPLAY_NAME_MAPPINGS = {**_NAMES}

# JS folder-picker extension lives in ./web (web/kreaphoton_save.js).
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
