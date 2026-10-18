// KreaPhoton sigma strip, Empty Latent readout and lint chips (node-ui-concept P2, 2026-10-18).
//
// The strip shows the schedule a node will run: the sigma curve, the restart band and one cell per
// model call coloured by phase (composition blue, identity green, draft grey, texture amber;
// self-refine cells hatched when the loop may skip them), with the counts in words underneath.
// Nothing is computed here: /kreaphoton/schedule runs the same Python functions as the samplers
// and /kreaphoton/ui serves the resolution table, LoRA phases and media nouns.
//
// Every element is a DOM widget with serialize = false: the frontend (1.53.6) skips those when it
// writes widgets_values, so saved workflows keep their positional values. Each widget polls its
// node every 400 ms, builds the query from the graph (own widgets, the Empty Latent upstream, the
// sampler downstream of a LoRA Phase) and fetches only when the query changed - one mechanism
// that follows widget edits, links, upstream nodes and workflow loads alike. The helpers are exported
// for kreaphoton_sections.js (P3).
import { app } from "../../scripts/app.js";
import { themeName } from "./kreaphoton_theme.js";

const SETTING_STRIP = "KreaPhoton.SigmaStrip";
const POLL_MS = 400;
const STRIP_H = 88;           // curve + cells 48 px, a two-line caption 28 px
const COMPACT_H = 34;
const LINE_H = 22;
const CHIP_H = 18;

// docs/design/node-ui-concept.md section 3
export const TOKENS = {
    dark: {
        composition: "#3987e5", identity: "#199e70", texture: "#d95926", draft: "#b3b9b5",
        band: "rgba(74, 42, 32, 0.55)", warn: "#e0a33a", curve: "#c8ccc9", off: "#3a3f41",
    },
    light: {
        composition: "#2a78d6", identity: "#13845d", texture: "#c24d1f", draft: "#6f7578",
        band: "#f3e3de", warn: "#9a6200", curve: "#4a5053", off: "#d3d7d2",
    },
};
const PHASE_COLOR = { composition: "composition", identity: "identity", draft: "draft", texture: "texture",
    refine: "identity" };
const SVG_NS = "http://www.w3.org/2000/svg";

let uiPromise = null;
export function loadUI() {
    uiPromise ??= fetch("/kreaphoton/ui").then((r) => (r.ok ? r.json() : null)).catch(() => null);
    return uiPromise;
}

const scheduleCache = new Map();
const CACHE_MAX = 256;
function fetchSchedule(query) {
    if (!scheduleCache.has(query)) {
        if (scheduleCache.size >= CACHE_MAX) scheduleCache.delete(scheduleCache.keys().next().value);
        scheduleCache.set(query, fetch(`/kreaphoton/schedule?${query}`)
            .then(async (r) => (r.ok ? r.json() : { error: (await r.json().catch(() => ({}))).error || r.statusText }))
            .catch((e) => ({ error: String(e) })));
    }
    return scheduleCache.get(query);
}

function stripEnabled() {
    try {
        return app.extensionManager?.setting?.get(SETTING_STRIP) ?? true;
    } catch {
        return true;
    }
}

// ---------------------------------------------------------------- graph helpers

export const widgetValue = (node, name) => node?.widgets?.find((w) => w.name === name)?.value;

// links and nodes are looked up in the node's own graph: inside a subgraph that is not app.graph
function linkById(graph, id) {
    const links = graph?.links;
    return links?.get?.(id) ?? links?.[id];
}

// the node feeding input `name`, through Reroute nodes
export function upstream(node, name) {
    let cur = node;
    let slot = cur?.inputs?.findIndex((i) => i.name === name);
    for (let hop = 0; hop < 16 && cur && slot >= 0; hop++) {
        const id = cur.inputs[slot]?.link;
        if (id == null) return null;
        const link = linkById(cur.graph, id);
        const origin = link ? cur.graph.getNodeById(link.origin_id) : null;
        if (!origin || origin.type !== "Reroute") return origin;
        cur = origin;
        slot = 0;
    }
    return null;
}

// nodes fed by output `slot`, through Reroute nodes
export function downstream(node, slot) {
    const out = [];
    const seen = new Set();
    const walk = (n, s, depth) => {
        for (const id of n?.outputs?.[s]?.links ?? []) {
            const link = linkById(n.graph, id);
            const target = link ? n.graph.getNodeById(link.target_id) : null;
            if (!target || seen.has(target.id) || depth > 16) continue;
            seen.add(target.id);
            if (target.type === "Reroute") walk(target, 0, depth + 1);
            else out.push({ node: target, input: target.inputs?.[link.target_slot]?.name });
        }
    };
    walk(node, slot, 0);
    return out;
}

function latentSize(node, inputName, ui) {
    const src = upstream(node, inputName);
    if (src?.comfyClass !== "KreaPhotonEmptyLatent" && src?.type !== "KreaPhotonEmptyLatent") return null;
    const wh = ui?.resolution?.buckets?.[widgetValue(src, "size")]?.[widgetValue(src, "aspect")];
    return wh ? { w: wh[0], h: wh[1] } : null;
}

export function params(obj) {
    const p = new URLSearchParams();
    for (const [k, v] of Object.entries(obj)) if (v !== undefined && v !== null) p.set(k, String(v));
    return p.toString();
}

function samplerQuery(node, ui) {
    const lat = latentSize(node, "latent_image", ui);
    return params({
        node: "sampler", preset: widgetValue(node, "preset"), coherence: widgetValue(node, "coherence"),
        denoise: widgetValue(node, "denoise"), profile: widgetValue(node, "precision_profile"),
        w: lat?.w, h: lat?.h,
    });
}

const SCHEDULER_KEYS = ["steps", "alpha", "restart_frac", "sigma_r", "plunge", "refine_steps", "refine_sigma",
    "plunge_sigma", "soft_plunge"];

function schedulerQuery(node, ui) {
    const lat = latentSize(node, "latent", ui);
    const q = { node: "scheduler", w: lat?.w, h: lat?.h };
    for (const k of SCHEDULER_KEYS) q[k] = widgetValue(node, k);
    return params(q);
}

// the schedule a LoRA Phase's LoRA ends up in: the first KreaPhoton sampler downstream on MODEL
// (through other LoRA Phase nodes); the Advanced sampler is described by its Scheduler
function loraTarget(node, ui) {
    const queue = [node];
    const seen = new Set([node.id]);
    while (queue.length) {
        const cur = queue.shift();
        const slot = cur.outputs?.findIndex((o) => o.type === "MODEL") ?? -1;
        if (slot < 0) continue;
        for (const { node: t } of downstream(cur, slot)) {
            if (seen.has(t.id)) continue;
            seen.add(t.id);
            const cls = t.comfyClass ?? t.type;
            if (cls === "KreaPhotonSampler") return { query: samplerQuery(t, ui), source: t.title || "Sampler" };
            if (cls === "KreaPhotonSamplerAdvanced") {
                const sched = upstream(t, "sigmas");
                if ((sched?.comfyClass ?? sched?.type) === "KreaPhotonScheduler") {
                    return { query: schedulerQuery(sched, ui), source: sched.title || "Scheduler" };
                }
            }
            if (cls === "KreaPhotonLoraPhase") queue.push(t);
        }
    }
    return { query: params({ node: "sampler", preset: ui?.default_preset }), source: null };
}

// ---------------------------------------------------------------- drawing

export function el(tag, style, text) {
    const e = document.createElement(tag);
    if (style) Object.assign(e.style, style);
    if (text !== undefined) e.textContent = text;
    return e;
}

function svgEl(tag, attrs) {
    const e = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
    return e;
}

function phaseWord(c) {
    if (c.phase === "refine") return c.conditional ? "self-refine (skipped on a stable draft)" : "self-refine";
    return c.phase;
}

// the strip body: sigma curve over a row of cells; `lit` (Set of call indices) dims the rest
function drawStrip(svg, data, { compact = false, lit = null } = {}) {
    const t = TOKENS[themeName()];
    svg.replaceChildren();
    const calls = data.calls ?? [];
    const n = Math.max(calls.length, 1);
    const W = 100 * n;                       // 100 user units per cell; the svg stretches horizontally
    const cellY = compact ? 0 : 34;
    const cellH = compact ? 14 : 14;
    const H = cellY + cellH;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("preserveAspectRatio", "none");
    const defs = svgEl("defs", {});
    const hatch = svgEl("pattern", { id: "kp-hatch-" + Math.random().toString(36).slice(2), width: 24, height: 24,
        patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" });
    hatch.appendChild(svgEl("rect", { width: 24, height: 24, fill: t.off }));
    hatch.appendChild(svgEl("rect", { width: 11, height: 24, fill: t.identity }));
    defs.appendChild(hatch);
    svg.appendChild(defs);
    const ri = data.restart ? calls.findIndex((c) => c.i >= data.restart.index) : -1;
    if (ri >= 0) {
        svg.appendChild(svgEl("rect", { x: ri * 100, y: 0, width: (n - ri) * 100, height: H, fill: t.band }));
    }
    if (!compact) {
        // each call is a segment sigma -> next over its cell: the draft readout drops to 0, a
        // restart / self-refine jump shows as the rise at the next cell's left edge
        const y = (v) => 2 + (1 - v) * 28;
        const pts = calls.map((c, k) => `${k * 100},${y(c.sigma)} ${(k + 1) * 100},${y(c.next)}`).join(" ");
        svg.appendChild(svgEl("polyline", { points: pts, fill: "none", stroke: t.curve, "stroke-width": 1.5,
            "vector-effect": "non-scaling-stroke", "stroke-linejoin": "round" }));
    }
    calls.forEach((c, k) => {
        const on = !lit || lit.has(c.i);
        const fill = !on ? t.off : c.conditional ? `url(#${hatch.id})` : t[PHASE_COLOR[c.phase]];
        const r = svgEl("rect", { x: k * 100 + 6, y: cellY, width: 88, height: cellH, rx: 10, fill });
        const tip = svgEl("title", {});
        tip.textContent = `call ${k + 1}: sigma ${c.sigma.toFixed(3)} -> ${c.next.toFixed(3)}, ${phaseWord(c)}`
            + (c.pc ? ", + corrector call" : "");
        r.appendChild(tip);
        svg.appendChild(r);
        if (c.pc && !compact) {
            svg.appendChild(svgEl("rect", { x: k * 100 + 40, y: cellY - 5, width: 20, height: 3, fill: t.curve }));
        }
    });
}

export function makeBox(height) {
    const box = el("div", {
        display: "flex", flexDirection: "column", gap: "2px", boxSizing: "border-box",
        height: `${height}px`, padding: "2px 0", overflow: "hidden", pointerEvents: "auto",
    });
    return box;
}

export function captionEl(lines = 1) {
    return el("div", {
        font: "11px/14px var(--comfy-font, sans-serif)", color: "var(--descrip-text, #9a9a9a)",
        fontVariantNumeric: "tabular-nums", overflow: "hidden", maxHeight: `${14 * lines}px`,
        ...(lines === 1 ? { whiteSpace: "nowrap", textOverflow: "ellipsis" }
            : { display: "-webkit-box", webkitBoxOrient: "vertical", webkitLineClamp: String(lines) }),
    });
}

export function addWidget(node, name, element, height, margin = 4) {
    const w = node.addDOMWidget(name, "kp_ui", element, {
        serialize: false,
        hideOnZoom: false,
        margin,                    // the frontend default (10 px) eats a third of a 48 px chip row
        getMinHeight: () => w.kpHeight ?? height,
        getMaxHeight: () => w.kpHeight ?? height,
    });
    w.serialize = false;       // LGraphNode skips these in widgets_values (frontend 1.53.6)
    w.kpHeight = height;
    return w;
}

export function setHeight(node, w, height) {
    if (w.kpHeight === height) return;
    w.kpHeight = height;
    w.element.style.display = height ? "" : "none";
    const min = node.computeSize?.();
    if (min) node.setSize([node.size[0], height ? Math.max(node.size[1], min[1]) : min[1]]);
    node.setDirtyCanvas?.(true, true);
}

export function poll(node, w, tick) {
    let last = null;
    let attached = false;
    let timer = null;
    const run = async () => {
        // nodeCreated runs before the node joins a graph; once it has been in one, leaving it
        // means it was removed (graph.clear() and workflow switches may skip onRemoved)
        if (!node.graph) {
            if (attached) clearInterval(timer);
            return;
        }
        attached = true;
        const ui = await loadUI();
        const key = await tick(ui, last);
        if (key !== undefined) last = key;
    };
    timer = setInterval(run, POLL_MS);
    run();
    const prev = node.onRemoved;
    node.onRemoved = function () {
        clearInterval(timer);
        return prev?.apply(this, arguments);
    };
}

// ---------------------------------------------------------------- the widgets

function attachStrip(node, kind) {
    const compact = kind === "lora";
    const height = compact ? COMPACT_H : STRIP_H;
    const box = makeBox(height);
    const svg = svgEl("svg", { width: "100%" });
    svg.style.height = compact ? "14px" : "48px";
    svg.style.display = "block";
    svg.setAttribute("role", "img");
    const cap = captionEl(compact ? 1 : 2);
    box.append(svg, cap);
    const w = addWidget(node, "kp_sigma_strip", box, height);
    if (kind === "sampler") {
        // under seed / preset / variety (concept section 5); a non-serialized widget does not
        // take part in widgets_values, so its place in the list is free
        const at = node.widgets.findIndex((x) => x.name === "variety");
        if (at >= 0 && node.widgets[node.widgets.length - 1] === w) {
            node.widgets.splice(node.widgets.length - 1, 1);
            node.widgets.splice(at + 1, 0, w);
        }
    }
    let theme = null;
    poll(node, w, async (ui, last) => {
        if (!stripEnabled()) {
            setHeight(node, w, 0);
            return null;
        }
        setHeight(node, w, height);
        let query;
        let source = null;
        let lit = null;
        if (kind === "sampler") query = samplerQuery(node, ui);
        else if (kind === "scheduler") query = schedulerQuery(node, ui);
        else ({ query, source } = loraTarget(node, ui));
        const phase = kind === "lora" ? widgetValue(node, "phase") : null;
        const key = `${query}|${phase}|${themeName()}`;
        if (key === last && theme === themeName()) return key;
        theme = themeName();
        const data = await fetchSchedule(query);
        if (data.error) {
            svg.replaceChildren();
            cap.textContent = `Schedule unavailable: ${data.error}`;
            return key;
        }
        let caption = data.caption;
        if (kind === "lora") {
            const segs = new Set(ui?.lora_phases?.[phase] ?? []);
            lit = new Set(data.calls.filter((c) => segs.has(c.segment)).map((c) => c.i));
            caption = `${phase}: ${lit.size} of ${data.calls.length} calls`
                + (source ? `, schedule of ${source}` : `, ${ui?.default_preset ?? "default"} schedule`);
        } else if (kind === "sampler" && data.latent?.assumed) {
            // the Scheduler without a latent runs its alpha widget as typed: no size involved
            caption += ` (size assumed ${data.latent.w} × ${data.latent.h})`;
        }
        drawStrip(svg, data, { compact, lit });
        cap.textContent = caption;
        cap.title = caption;
        svg.setAttribute("aria-label", caption);
        return key;
    });
}

function orientation(w, h) {
    return w === h ? "square" : w > h ? "landscape" : "portrait";
}

function attachReadout(node) {
    const box = makeBox(LINE_H);
    const cap = captionEl();
    box.append(cap);
    const w = addWidget(node, "kp_readout", box, LINE_H);
    poll(node, w, async (ui, last) => {
        const size = widgetValue(node, "size");
        const aspect = widgetValue(node, "aspect");
        const key = `${size}|${aspect}`;
        if (key === last) return key;
        const wh = ui?.resolution?.buckets?.[size]?.[aspect];
        cap.textContent = wh
            ? `${wh[0]} × ${wh[1]}, ${((wh[0] * wh[1]) / 1e6).toFixed(2)} MP, ${orientation(wh[0], wh[1])} ${aspect}`
            : "";
        return key;
    });
}

function chipEl() {
    const t = TOKENS[themeName()];
    return el("div", {
        font: "11px/14px var(--comfy-font, sans-serif)", color: "var(--fg-color, #e6e6e6)",
        border: `1px solid ${t.warn}`, borderRadius: "4px", padding: "1px 6px", whiteSpace: "nowrap",
        overflow: "hidden", textOverflow: "ellipsis", boxSizing: "border-box", height: `${CHIP_H}px`,
        flex: "0 0 auto",
    });
}

function attachChips(node, compute) {
    const box = makeBox(0);
    box.style.height = "100%";
    const w = addWidget(node, "kp_lint", box, 0);
    w.element.style.display = "none";
    poll(node, w, async (ui, last) => {
        const notes = await compute(ui);
        const key = notes.join("\n") + "|" + themeName();
        if (key === last) return key;
        box.replaceChildren(...notes.map((n) => {
            const c = chipEl();
            c.textContent = n;
            c.title = n;
            return c;
        }));
        setHeight(node, w, notes.length ? notes.length * (CHIP_H + 2) + 4 : 0);
        return key;
    });
}

function mediaNouns(text, nouns) {
    const found = [];
    for (const word of nouns ?? []) {
        if (new RegExp(`\\b${word}s?\\b`, "i").test(text || "")) found.push(word);
    }
    return found;
}

const encodeNotes = async (node, ui) => {
    const style = widgetValue(node, "style");
    const text = [widgetValue(node, "text"), style === "custom" ? widgetValue(node, "custom_directive") : ""]
        .join("\n");
    return mediaNouns(text, ui?.media_nouns).map((w) => `'${w}' may render as print or text`);
};

const schedulerNotes = async (node, ui) => {
    const notes = [];
    if (Number(widgetValue(node, "refine_steps")) > 0 && !widgetValue(node, "plunge")) {
        notes.push("Self-refine needs plunge: refine_steps is ignored");
    }
    const data = await fetchSchedule(schedulerQuery(node, ui));
    if (data?.restart) {
        for (const { node: t } of downstream(node, 0)) {
            const cls = t.comfyClass ?? t.type;
            if (cls !== "KreaPhotonSamplerAdvanced") notes.push(`Feeds ${t.title || cls}: restart will be ignored`);
        }
    }
    return notes;
};

app.registerExtension({
    name: "KreaPhoton.SigmaStrip",
    settings: [
        {
            id: SETTING_STRIP,
            category: ["KreaPhoton", "Nodes", "Sigma strip"],
            name: "KreaPhoton: sigma strip on samplers, Scheduler and LoRA Phase",
            tooltip: "One cell per model call, coloured by phase: blue composition, green identity, grey draft, "
                + "amber texture; hatched cells run only if the draft still moves.",
            type: "boolean",
            defaultValue: true,
        },
    ],
    nodeCreated(node) {
        const cls = node.comfyClass ?? node.type;
        if (cls === "KreaPhotonSampler") attachStrip(node, "sampler");
        else if (cls === "KreaPhotonScheduler") {
            attachStrip(node, "scheduler");
            attachChips(node, (ui) => schedulerNotes(node, ui));
        } else if (cls === "KreaPhotonLoraPhase") attachStrip(node, "lora");
        else if (cls === "KreaPhotonEmptyLatent") attachReadout(node);
        else if (cls === "KreaPhotonEncode") attachChips(node, (ui) => encodeNotes(node, ui));
    },
});
