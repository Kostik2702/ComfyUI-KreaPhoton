"""
ComfyUI-KreaPhoton
Photorealism-focused sampling nodes for Krea 2 Turbo.
"""
from .kreaphoton.nodes import (NODE_CLASS_MAPPINGS as _NODES,
                               NODE_DISPLAY_NAME_MAPPINGS as _NAMES)

NODE_CLASS_MAPPINGS = {**_NODES}
NODE_DISPLAY_NAME_MAPPINGS = {**_NAMES}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
