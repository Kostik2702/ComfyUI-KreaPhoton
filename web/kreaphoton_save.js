// KreaPhoton Save Image - folder picker (KREA2-NODES 2026-07-08; rebuilt for node-ui-concept P1,
// 2026-10-17). Adds a "Choose folder…" button to the KreaPhotonSaveImage node that opens a modal
// walking the server filesystem via GET /kreaphoton/browse, then writes the chosen absolute path
// into the node's `folder_path` widget. The browser can not open a native OS folder dialog against
// the server FS, hence this custom modal backed by kreaphoton/server_routes.py (read-only route:
// no "new folder").
//
// Theme: ComfyUI CSS variables with dark fallbacks. Keyboard: Up/Down move, Enter opens the
// highlighted folder, Backspace goes up, Ctrl+Enter selects the current folder, Esc closes; focus
// stays inside the dialog. Recent folders live in localStorage (per browser, best effort).
import { app } from "../../scripts/app.js";

const RECENT_KEY = "kreaphoton.save.recentFolders";
const RECENT_MAX = 6;

async function fetchDirs(path) {
    const res = await fetch(`/kreaphoton/browse?path=${encodeURIComponent(path || "")}`);
    if (!res.ok) throw new Error(`browse failed: ${res.status}`);
    return res.json();
}

function loadRecent() {
    try {
        const v = JSON.parse(localStorage.getItem(RECENT_KEY) || "[]");
        return Array.isArray(v) ? v.filter((x) => typeof x === "string").slice(0, RECENT_MAX) : [];
    } catch {
        return [];
    }
}

function rememberRecent(path) {
    try {
        const next = [path, ...loadRecent().filter((p) => p !== path)].slice(0, RECENT_MAX);
        localStorage.setItem(RECENT_KEY, JSON.stringify(next));
    } catch {
        /* storage blocked: the picker still works, it just forgets */
    }
}

const V = {
    bg: "var(--comfy-menu-bg, #1e1e1e)",
    input: "var(--comfy-input-bg, #2b2b2b)",
    fg: "var(--fg-color, #e6e6e6)",
    muted: "var(--descrip-text, #9a9a9a)",
    border: "var(--border-color, #444)",
    accent: "#2a78d6",     // phase.composition (light) - readable on both themes for focus rings
};

function el(tag, css, text) {
    const e = document.createElement(tag);
    if (css) e.style.cssText = css;
    if (text !== undefined) e.textContent = text;
    return e;
}

function button(label, primary) {
    const b = el("button",
        `padding:5px 12px;border-radius:6px;font-size:12px;cursor:pointer;color:${V.fg};` +
        `border:1px solid ${primary ? V.accent : V.border};background:${primary ? V.accent : V.input};`, label);
    if (primary) b.style.color = "#fff";
    b.type = "button";
    return b;
}

function splitPath(path) {
    // -> [{label, path}] for every ancestor, Windows drive or POSIX root first
    if (!path) return [];
    const sep = path.includes("\\") ? "\\" : "/";
    const parts = path.split(sep).filter((x, i) => x !== "" || i === 0);
    const out = [];
    let acc = "";
    parts.forEach((p, i) => {
        if (i === 0) {
            acc = p === "" ? sep : p + sep;           // "/" or "E:\"
            out.push({ label: p === "" ? sep : p, path: acc });
        } else {
            acc = acc.endsWith(sep) ? acc + p : acc + sep + p;
            out.push({ label: p, path: acc });
        }
    });
    return out;
}

function openFolderModal(current, onSelect) {
    const overlay = el("div",
        "position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:10000;display:flex;align-items:center;" +
        "justify-content:center;");
    const box = el("div",
        `width:min(580px,92vw);max-height:80vh;display:flex;flex-direction:column;background:${V.bg};` +
        `border:1px solid ${V.border};border-radius:6px;color:${V.fg};font-family:inherit;` +
        "box-shadow:0 8px 30px rgba(0,0,0,.45);overflow:hidden;");
    box.setAttribute("role", "dialog");
    box.setAttribute("aria-modal", "true");
    box.setAttribute("aria-label", "Choose save folder");
    box.tabIndex = -1;
    overlay.appendChild(box);

    box.appendChild(el("div", `padding:10px 14px;border-bottom:1px solid ${V.border};font-weight:600;`,
        "Choose save folder"));
    const crumbs = el("div",
        `padding:8px 14px;font-size:12px;border-bottom:1px solid ${V.border};display:flex;flex-wrap:wrap;gap:2px;` +
        "min-height:18px;");
    box.appendChild(crumbs);
    const recentBar = el("div", `padding:6px 14px;font-size:12px;color:${V.muted};display:flex;flex-wrap:wrap;gap:6px;`);
    box.appendChild(recentBar);
    const list = el("div", "flex:1;overflow-y:auto;padding:4px 0;min-height:140px;outline:none;");
    list.setAttribute("role", "listbox");
    list.tabIndex = 0;
    box.appendChild(list);
    const hint = el("div", `padding:4px 14px;font-size:11px;color:${V.muted};`,
        "Up / Down move · Enter open · Backspace up · Ctrl+Enter select · Esc close");
    box.appendChild(hint);
    const footer = el("div", `padding:10px 14px;border-top:1px solid ${V.border};display:flex;gap:8px;` +
        "justify-content:flex-end;");
    box.appendChild(footer);

    let state = { path: current || "", parent: null };
    let rows = [];        // [{el, path}]
    let active = -1;

    const close = () => {
        document.removeEventListener("keydown", onKey, true);
        overlay.remove();
    };
    const choose = () => {
        if (!state.path) return;
        rememberRecent(state.path);
        onSelect(state.path);
        close();
    };

    function highlight(i) {
        if (!rows.length) return;
        active = Math.max(0, Math.min(rows.length - 1, i));
        rows.forEach((r, k) => {
            r.el.style.background = k === active ? V.input : "";
            r.el.setAttribute("aria-selected", k === active ? "true" : "false");
        });
        rows[active].el.scrollIntoView({ block: "nearest" });
    }

    function addRow(label, path, title) {
        const r = el("div", "padding:5px 14px;cursor:pointer;font-size:13px;", label);
        r.setAttribute("role", "option");
        if (title) r.title = title;
        r.onmouseenter = () => highlight(rows.findIndex((x) => x.el === r));
        r.onclick = () => render(path);
        list.appendChild(r);
        rows.push({ el: r, path });
    }

    function renderCrumbs(path) {
        crumbs.innerHTML = "";
        const segs = splitPath(path);
        if (!segs.length) {
            crumbs.appendChild(el("span", `color:${V.muted};`, "Drives"));
            return;
        }
        segs.forEach((s, i) => {
            if (i > 0) crumbs.appendChild(el("span", `color:${V.muted};padding:0 2px;`, "›"));
            const a = el("button",
                `background:none;border:none;padding:0 2px;font-size:12px;cursor:pointer;color:${V.fg};` +
                (i === segs.length - 1 ? "font-weight:600;" : "text-decoration:underline;"), s.label);
            a.type = "button";
            a.onclick = () => render(s.path);
            crumbs.appendChild(a);
        });
    }

    function renderRecent() {
        recentBar.innerHTML = "";
        const rec = loadRecent();
        if (!rec.length) {
            recentBar.style.display = "none";
            return;
        }
        recentBar.style.display = "flex";
        recentBar.appendChild(el("span", "", "Recent:"));
        for (const p of rec) {
            const b = el("button",
                `background:${V.input};border:1px solid ${V.border};border-radius:4px;padding:1px 6px;` +
                `font-size:11px;cursor:pointer;color:${V.fg};max-width:220px;overflow:hidden;text-overflow:ellipsis;` +
                "white-space:nowrap;", p.split(/[\\/]/).filter(Boolean).pop() || p);
            b.type = "button";
            b.title = p;
            b.onclick = () => render(p);
            recentBar.appendChild(b);
        }
    }

    async function render(path) {
        list.innerHTML = "";
        rows = [];
        active = -1;
        let data;
        try {
            data = await fetchDirs(path);
        } catch (e) {
            list.appendChild(el("div", "padding:6px 14px;color:#d9534f;font-size:12px;", "Error: " + e.message));
            return;
        }
        state = { path: data.path, parent: data.parent };
        renderCrumbs(data.path);
        if (data.parent !== null) addRow("..", data.parent, "Up one level");
        if (data.error) list.appendChild(el("div", "padding:6px 14px;color:#d9534f;font-size:12px;", data.error));
        for (const d of data.dirs) addRow(d.name, d.path);
        if (!data.dirs.length && !data.error && data.parent !== null) {
            list.appendChild(el("div", `padding:6px 14px;color:${V.muted};font-size:12px;`, "No sub-folders"));
        }
        select.disabled = !state.path;
        select.style.opacity = state.path ? "1" : "0.5";
        highlight(data.parent !== null && rows.length > 1 ? 1 : 0);
        list.focus({ preventScroll: true });
    }

    function onKey(e) {
        if (e.key === "Escape") {
            e.preventDefault();
            close();
        } else if (e.key === "Tab") {
            // focus trap: cycle inside the dialog
            const f = [...box.querySelectorAll("button, [tabindex='0']")].filter((x) => !x.disabled);
            if (!f.length) return;
            const i = f.indexOf(document.activeElement);
            const next = e.shiftKey ? (i <= 0 ? f.length - 1 : i - 1) : (i === f.length - 1 ? 0 : i + 1);
            e.preventDefault();
            f[next].focus();
        } else if (document.activeElement === list || box.contains(document.activeElement)) {
            if (e.key === "ArrowDown") {
                e.preventDefault();
                highlight(active + 1);
            } else if (e.key === "ArrowUp") {
                e.preventDefault();
                highlight(active - 1);
            } else if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                choose();
            } else if (e.key === "Enter" && document.activeElement === list && rows[active]) {
                e.preventDefault();
                render(rows[active].path);
            } else if (e.key === "Backspace" && state.parent !== null) {
                e.preventDefault();
                render(state.parent);
            }
            e.stopPropagation();   // keep the canvas shortcuts out while the dialog is open
        }
    }

    const cancel = button("Cancel", false);
    cancel.onclick = close;
    const select = button("Select this folder", true);
    select.onclick = choose;
    footer.appendChild(cancel);
    footer.appendChild(select);

    overlay.onclick = (e) => {
        if (e.target === overlay) close();
    };
    document.addEventListener("keydown", onKey, true);
    document.body.appendChild(overlay);
    renderRecent();
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
            this.addWidget("button", "Choose folder…", null, () => {
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
