// KreaPhoton Save Image - folder picker (KREA2-NODES 2026-07-08).
// Adds a "Browse" button to the KreaPhotonSaveImage node that opens a modal
// walking the server filesystem via GET /kreaphoton/browse, then writes the
// chosen absolute path into the node's `folder_path` widget. The browser can
// not open a native OS folder dialog against the server FS, hence this custom
// modal backed by kreaphoton/server_routes.py.
import { app } from "../../scripts/app.js";

async function fetchDirs(path) {
    const res = await fetch(`/kreaphoton/browse?path=${encodeURIComponent(path || "")}`);
    if (!res.ok) throw new Error(`browse failed: ${res.status}`);
    return res.json();
}

function styleButton(b) {
    b.style.cssText =
        "padding:4px 10px;border-radius:6px;border:1px solid #555;background:#2b2b2b;" +
        "color:#eee;cursor:pointer;font-size:12px;";
}

function openFolderModal(current, onSelect) {
    const overlay = document.createElement("div");
    overlay.style.cssText =
        "position:fixed;inset:0;background:rgba(0,0,0,.6);z-index:10000;" +
        "display:flex;align-items:center;justify-content:center;";

    const box = document.createElement("div");
    box.style.cssText =
        "width:min(560px,90vw);max-height:80vh;display:flex;flex-direction:column;" +
        "background:#1e1e1e;border:1px solid #444;border-radius:10px;color:#eee;" +
        "font-family:sans-serif;box-shadow:0 8px 30px rgba(0,0,0,.5);overflow:hidden;";
    overlay.appendChild(box);

    const header = document.createElement("div");
    header.style.cssText = "padding:10px 14px;border-bottom:1px solid #333;font-weight:600;";
    header.textContent = "Select save folder";
    box.appendChild(header);

    const crumb = document.createElement("div");
    crumb.style.cssText =
        "padding:8px 14px;font-size:12px;color:#9cd;word-break:break-all;" +
        "border-bottom:1px solid #2a2a2a;min-height:18px;";
    box.appendChild(crumb);

    const listWrap = document.createElement("div");
    listWrap.style.cssText = "flex:1;overflow-y:auto;padding:6px 0;min-height:120px;";
    box.appendChild(listWrap);

    const footer = document.createElement("div");
    footer.style.cssText =
        "padding:10px 14px;border-top:1px solid #333;display:flex;gap:8px;justify-content:flex-end;";
    box.appendChild(footer);

    let state = { path: current || "", parent: null };

    const close = () => document.body.removeChild(overlay);

    function row(label, onClick, icon) {
        const r = document.createElement("div");
        r.style.cssText =
            "padding:6px 14px;cursor:pointer;font-size:13px;display:flex;gap:8px;align-items:center;";
        r.onmouseenter = () => (r.style.background = "#2d2d2d");
        r.onmouseleave = () => (r.style.background = "");
        r.textContent = `${icon} ${label}`;
        r.onclick = onClick;
        return r;
    }

    async function render(path) {
        listWrap.innerHTML = "";
        let data;
        try {
            data = await fetchDirs(path);
        } catch (e) {
            crumb.textContent = "Error: " + e.message;
            return;
        }
        state = { path: data.path, parent: data.parent };
        crumb.textContent = data.path || "(drives)";
        if (data.parent !== null) {
            listWrap.appendChild(row("..", () => render(data.parent), "⬆"));
        }
        if (data.error) {
            const e = document.createElement("div");
            e.style.cssText = "padding:6px 14px;color:#e88;font-size:12px;";
            e.textContent = data.error;
            listWrap.appendChild(e);
        }
        for (const d of data.dirs) {
            listWrap.appendChild(row(d.name, () => render(d.path), "📁"));
        }
        if (!data.dirs.length && !data.error && data.parent !== null) {
            const e = document.createElement("div");
            e.style.cssText = "padding:6px 14px;color:#888;font-size:12px;";
            e.textContent = "(no sub-folders)";
            listWrap.appendChild(e);
        }
    }

    const cancel = document.createElement("button");
    styleButton(cancel);
    cancel.textContent = "Cancel";
    cancel.onclick = close;

    const select = document.createElement("button");
    styleButton(select);
    select.style.borderColor = "#3a7";
    select.style.background = "#254";
    select.textContent = "Select this folder";
    select.onclick = () => {
        if (state.path) {
            onSelect(state.path);
            close();
        }
    };
    footer.appendChild(cancel);
    footer.appendChild(select);

    overlay.onclick = (e) => {
        if (e.target === overlay) close();
    };
    document.body.appendChild(overlay);
    render(state.path);
}

app.registerExtension({
    name: "KreaPhoton.SaveImage.FolderPicker",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "KreaPhotonSaveImage") return;
        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated ? onCreated.apply(this, arguments) : undefined;
            const pathWidget = this.widgets.find((w) => w.name === "folder_path");
            this.addWidget("button", "📁 Browse folder", null, () => {
                openFolderModal(pathWidget?.value || "", (p) => {
                    if (pathWidget) {
                        pathWidget.value = p;
                        this.setDirtyCanvas(true, true);
                    }
                });
            });
            return r;
        };
    },
});
