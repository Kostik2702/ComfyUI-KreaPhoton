# -*- coding: utf-8 -*-
"""Server-side routes for KreaPhoton (KREA2-NODES 2026-07-08).

Registers GET /kreaphoton/browse - the directory listing that backs the JS
folder picker in web/kreaphoton_save.js. The browser cannot open a native OS
folder dialog against the server filesystem, so the modal walks the tree via
this endpoint. Read-only: it lists sub-directories only, never reads or writes
files.

Import is wrapped so the plain-assert test suite (no ComfyUI/aiohttp) can import
the package without a running server."""
from .save import list_dirs

_REGISTERED = False
try:
    from server import PromptServer
    from aiohttp import web

    # PromptServer.instance is created at ComfyUI startup, before custom nodes
    # load - but may be absent when the module is imported standalone (unit
    # tests / tooling). Guard on it so a bare import is always a safe no-op.
    if getattr(PromptServer, "instance", None) is not None:

        @PromptServer.instance.routes.get("/kreaphoton/browse")
        async def kreaphoton_browse(request):
            """?path=<dir> -> {path, parent, dirs:[{name,path}], error?}.
            Empty path -> drive roots (Windows) / filesystem root (POSIX)."""
            return web.json_response(list_dirs(request.query.get("path", "")))

        _REGISTERED = True
except ImportError:
    # No ComfyUI server (e.g. unit tests) - routes simply not registered.
    pass
