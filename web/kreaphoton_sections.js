// KreaPhoton node UI P3 (node-ui-concept s5-s6, 2026-10-18): Sampler (Advanced) sections, changed
// dots, relevance dimming, "Start from preset..."; "More inputs" on the simple Sampler; dimming on
// the Scheduler.
//
// Nothing here is decided in JS: the sections, the relevance rules, the experimental list, the
// widget defaults and the values a preset writes come from /kreaphoton/ui and
// /kreaphoton/preset_values (kreaphoton/ui_data.py, every rule proven by tests/test_ui_relevance.py).
// JS only evaluates the rule conditions on the current widget values.
//
// The widget order is frozen (widgets_values is positional), and a section's widgets are scattered
// over it, so sections are an accordion: one open at a time, its widgets shown (in their frozen
// order), the others hidden with widget.hidden - still serialized and still sent to the server.
// Headers, the preset picker and the notes are DOM widgets with serialize = false; moving those in
// node.widgets does not touch widgets_values. A dimmed widget keeps its value and stays editable:
// the frontend's own `disabled` (1.53.6) hides the value and swallows clicks, so the canvas dims it
// by wrapping its drawWidget, and the Vue renderer by the opacity of its row.
import { app } from "../../scripts/app.js";
import { themeName } from "./kreaphoton_theme.js";
import { TOKENS, addWidget, captionEl, el, loadUI, makeBox, params, poll, upstream, widgetValue }
    from "./kreaphoton_strip.js";

const SETTING_SECTIONS = "KreaPhoton.AdvancedSections";
const HEADER_H = 22;
const NOTE_LINE = 14;
const DIM_ALPHA = 0.42;
const DOT = "• ";
const EXP = " · exp";

function setting(id, fallback) {
    try {
        return app.extensionManager?.setting?.get(id) ?? fallback;
    } catch {
        return fallback;
    }
}

// ---------------------------------------------------------------- rule conditions (ui_data.condition_holds)

function holds(cond, values, ctx) {
    if (cond.all) return cond.all.every((c) => holds(c, values, ctx));
    if (cond.any) return cond.any.some((c) => holds(c, values, ctx));
    if (cond.context) return !!ctx?.[cond.context];
    const v = values[cond.widget];
    if (cond.in) return cond.in.includes(v);
    if (cond.is !== undefined) return !!v === cond.is;
    if (cond.ne !== undefined) return Number(v) !== Number(cond.ne);
    if (cond.gte !== undefined) return Number(v) >= Number(cond.gte);
    if (cond.gt !== undefined) return Number(v) > Number(cond.gt);
    return true;
}

function dimmedMap(rules, values, ctx) {
    const out = new Map();
    for (const r of rules ?? []) {
        if (holds(r.when, values, ctx)) continue;
        for (const w of r.widgets) if (!out.has(w)) out.set(w, r.why);
    }
    return out;
}

// ---------------------------------------------------------------- graph facts

const cls = (n) => n?.comfyClass ?? n?.type;
const inputLinked = (node, name) => node.inputs?.find((i) => i.name === name)?.link != null;
// a widget whose socket is linked shows its socket on its row: never hide that row
const widgetLinked = (node, w) => node.inputs?.find((i) => i.widget?.name === w.name)?.link != null;

// texture_start acts only with a texture model: texture_model linked, or a LoRA Phase upstream on
// `model` whose phase splits the texture set off (ui.texture_split_phases)
function hasTextureModel(node, ui) {
    if (inputLinked(node, "texture_model")) return true;
    let cur = upstream(node, "model");
    for (let hop = 0; cur && cls(cur) === "KreaPhotonLoraPhase" && hop < 64; hop++) {
        if ((ui?.texture_split_phases ?? []).includes(widgetValue(cur, "phase"))
            && Number(widgetValue(cur, "strength")) !== 0) return true;
        cur = upstream(cur, "model");
    }
    return false;
}

function values(node) {
    const out = {};
    for (const w of node.widgets ?? []) if (w.serialize !== false && w.name) out[w.name] = w.value;
    return out;
}

// LGraphNode.computeSize (1.53.6) counts every widget as its height + 4 px, arrange() gives a DOM
// widget its minHeight and spreads the free space up to its maxHeight: a minHeight 4 px short makes
// the node exactly as tall as its rows instead of 4 px too tall per DOM widget
function snug(w) {
    w.options.getMinHeight = () => Math.max(0, (w.kpHeight ?? 0) - 4);
    w.options.getMaxHeight = () => w.kpHeight ?? 0;
    return w;
}

const same = (a, b) => (typeof a === "number" && typeof b === "number" ? Math.abs(a - b) < 1e-6 : a === b);

// ---------------------------------------------------------------- widget decoration

function setDim(w, on) {
    w.kpDim = on;
    if (w.kpDimWrapped || typeof w.drawWidget !== "function") return;
    const draw = w.drawWidget;
    w.drawWidget = function (ctx, opts) {
        if (!this.kpDim) return draw.call(this, ctx, opts);
        ctx.save();
        ctx.globalAlpha *= DIM_ALPHA;
        try {
            return draw.call(this, ctx, opts);
        } finally {
            ctx.restore();
        }
    };
    w.kpDimWrapped = true;
}

function setLabel(w, changed, experimental) {
    const label = (changed ? DOT : "") + w.name + (experimental ? EXP : "");
    if (label === w.name) {
        if (w.label?.startsWith(DOT) || w.label?.endsWith(EXP)) w.label = undefined;
    } else if (w.label !== label) w.label = label;
}

// the Vue node renderer draws its own rows; they carry the label text, not the widget
function vueDim(node) {
    if (!setting("Comfy.VueNodes.Enabled", false)) return;
    const root = document.querySelector(`[data-node-id="${node.id}"]`);
    if (!root) return;
    const byLabel = new Map((node.widgets ?? []).map((w) => [w.label ?? w.name, w]));
    for (const row of root.querySelectorAll('[data-testid="node-widget"]')) {
        const text = row.querySelector('[data-testid="widget-layout-field-label"]')?.textContent?.trim();
        const w = byLabel.get(text);
        if (!w) continue;
        const want = w.kpDim ? String(DIM_ALPHA) : "";
        if (row.style.opacity !== want) row.style.opacity = want;
    }
}

function relayout(node) {
    const min = node.computeSize?.();
    if (min) node.setSize([node.size[0], min[1]]);
    node.setDirtyCanvas?.(true, true);
}

function moveWidget(node, w, index) {
    const at = node.widgets.indexOf(w);
    if (at < 0) return;
    node.widgets.splice(at, 1);
    node.widgets.splice(Math.min(index, node.widgets.length), 0, w);
}

// ---------------------------------------------------------------- section headers

function summaryText(sec, node, vals, changedCount) {
    const s = sec.summary ?? {};
    const fmt = (v) => (typeof v === "number" ? String(Math.round(v * 1000) / 1000) : String(v));
    if (s.kind === "value") return fmt(vals[s.widget]);
    if (s.kind === "switch") return Number(vals[s.widget]) ? (s.on ?? "on").replace("{v}", fmt(vals[s.widget])) : "off";
    if (s.kind === "any_on") return s.widgets.some((w) => Number(vals[w]) > 0) ? "on" : "off";
    if (s.kind === "inputs") return s.inputs.filter((i) => inputLinked(node, i)).join(", ") || "none";
    return changedCount ? `${changedCount} of ${sec.widgets.length} changed` : "";
}

function headerWidget(node, sec, onToggle) {
    const box = makeBox(HEADER_H);
    // the DOM widget's height follows kpHeight (a second line for the reasons)
    Object.assign(box.style, { gap: "0", padding: "0", cursor: "pointer", userSelect: "none", height: "100%" });
    box.setAttribute("role", "button");
    box.tabIndex = 0;
    const row = el("div", {
        display: "flex", alignItems: "center", gap: "6px", height: `${HEADER_H - 2}px`, flex: "0 0 auto",
        font: "12px/16px var(--comfy-font, sans-serif)", color: "var(--fg-color, #e6e6e6)",
        borderTop: "1px solid var(--border-color, #4a4a4a)", padding: "0 4px", boxSizing: "border-box",
    });
    const chev = el("span", { width: "10px", flex: "0 0 auto", color: "var(--descrip-text, #9a9a9a)" });
    const title = el("span", { flex: "1 1 auto", overflow: "hidden", whiteSpace: "nowrap", textOverflow: "ellipsis" });
    const sum = el("span", {
        flex: "0 1 auto", color: "var(--descrip-text, #9a9a9a)", fontVariantNumeric: "tabular-nums",
        overflow: "hidden", whiteSpace: "nowrap", textOverflow: "ellipsis", maxWidth: "55%",
    });
    row.append(chev, title, sum);
    const note = captionEl();
    note.style.padding = "0 4px 0 20px";
    box.append(row, note);
    box.addEventListener("click", (e) => {
        e.stopPropagation();
        onToggle(sec.id);
    });
    box.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            onToggle(sec.id);
        }
    });
    box.addEventListener("pointerdown", (e) => e.stopPropagation());
    // no DOM margin: the element gets the whole row (with margin 4 it would get kpHeight - 8)
    const w = snug(addWidget(node, `kp_section_${sec.id}`, box, HEADER_H, 0));
    w.kpParts = { box, chev, title, sum, note };
    return w;
}

function paintHeader(w, { open, title, summary, changed, note, dotColor }) {
    const p = w.kpParts;
    p.chev.textContent = open ? "▾" : "▸";
    p.box.setAttribute("aria-expanded", String(open));
    p.title.textContent = title;
    p.title.style.color = changed ? dotColor : "";
    p.sum.textContent = summary;
    p.sum.title = summary;
    p.note.textContent = note;
    p.note.title = note;
    p.note.style.display = note ? "" : "none";
    w.kpHeight = HEADER_H + (note ? NOTE_LINE : 0);
}

// ---------------------------------------------------------------- the sections controller

function attachSections(node, kind) {
    const nodeCls = cls(node);
    const props = (node.properties ??= {});
    const headers = new Map();
    let picker = null;
    let dimNote = null;
    let sections = [];
    let lastLayout = null;
    let flash = "";

    const toggle = (id) => {
        props.kp_open_section = props.kp_open_section === id ? null : id;
        lastLayout = null;
    };

    if (kind === "advanced") picker = attachPicker(node, (msg) => {
        flash = msg;
        lastLayout = null;
    });
    if (kind === "scheduler") {
        const box = makeBox(0);
        box.style.height = "100%";
        const cap = captionEl(3);
        box.append(cap);
        dimNote = snug(addWidget(node, "kp_dim_note", box, 0, 0));
        dimNote.kpCap = cap;
        dimNote.element.style.display = "none";
    }

    poll(node, picker ?? dimNote, async (ui) => {
        if (!ui) return null;
        if (!sections.length && kind !== "scheduler") {
            sections = ui.sections?.[nodeCls] ?? [];
            for (const sec of sections) headers.set(sec.id, headerWidget(node, sec, toggle));
        }
        const on = kind !== "advanced" || setting(SETTING_SECTIONS, true);
        const vals = values(node);
        const ctx = { texture_model: kind === "advanced" && hasTextureModel(node, ui) };
        const dim = dimmedMap(ui.relevance?.[nodeCls], vals, ctx);
        const headline = new Set((ui.relevance?.[nodeCls] ?? []).filter((r) => r.headline).map((r) => r.why));
        const ref = { ...(ui.defaults?.[nodeCls] ?? {}), ...(props.kp_reference?.values ?? {}) };
        const exp = new Set(kind === "advanced" ? ui.experimental ?? [] : []);
        const color = TOKENS[themeName()].composition;
        // dots only inside sections: seed and its control are run choices, not settings
        const inSection = new Set(kind === "advanced" ? sections.flatMap((sec) => sec.widgets) : []);

        for (const w of node.widgets ?? []) {
            if (w.serialize === false || !w.name) continue;
            setDim(w, dim.has(w.name));
            const changed = inSection.has(w.name) && w.name in ref && !same(w.value, ref[w.name]);
            setLabel(w, changed, exp.has(w.name));
        }

        const open = on ? props.kp_open_section ?? null : null;
        const layoutKey = JSON.stringify([on, open, flash, themeName(), sections.map((s) => s.id),
            sections.map((sec) => sec.widgets.map((n) => [vals[n], dim.has(n)])),
            ["clean_model", "texture_model", "vae"].map((i) => inputLinked(node, i)),
            node.inputs?.filter((i) => i.widget && i.link != null).map((i) => i.name), props.kp_reference?.preset]);

        if (kind === "scheduler") {
            const notes = [...new Set(dim.values())];
            const text = notes.join(". ");
            if (dimNote.kpCap.textContent !== text) {
                dimNote.kpCap.textContent = text;
                dimNote.kpCap.title = text;
                dimNote.kpHeight = notes.length ? NOTE_LINE * Math.min(3, notes.length) + 4 : 0;
                dimNote.element.style.display = notes.length ? "" : "none";
                relayout(node);
            }
        } else if (layoutKey !== lastLayout) {
            lastLayout = layoutKey;
            for (const sec of sections) {
                const hw = headers.get(sec.id);
                const changed = sec.widgets.filter((n) => n in ref && !same(vals[n], ref[n])).length;
                // an open section lists why its widgets are dimmed; a closed one only the headline
                // rules (the simple Sampler's denoise note)
                const reasons = [...new Set(sec.widgets.filter((n) => dim.has(n))
                    .map((n) => dim.get(n)).filter((why) => open === sec.id || headline.has(why)))];
                paintHeader(hw, {
                    open: open === sec.id, title: (changed && kind === "advanced" ? DOT : "") + sec.title,
                    summary: summaryText(sec, node, vals, changed), changed: changed > 0 && kind === "advanced",
                    note: reasons.join(". "), dotColor: color,
                });
                hw.hidden = !on;
                hw.element.style.display = on ? "" : "none";
                for (const name of sec.widgets) {
                    const w = node.widgets.find((x) => x.name === name);
                    if (w) w.hidden = on && open !== sec.id && !widgetLinked(node, w);
                }
            }
            placeHeaders(node, sections, headers, open, picker);
            if (picker) picker.kpPaint(props, flash);
            relayout(node);
        }
        vueDim(node);
        return null;
    });
}

// headers sit around the open section: the ones before it (and its own) right above its first
// widget, the ones after it right below its last; with none open they follow the preset picker
function placeHeaders(node, sections, headers, open, picker) {
    const hws = sections.map((s) => headers.get(s.id));
    for (const hw of hws) node.widgets.splice(node.widgets.indexOf(hw), 1);
    const indexOf = (name) => node.widgets.findIndex((w) => w.name === name);
    let anchor = picker ? node.widgets.indexOf(picker) + 1 : -1;
    if (anchor < 0) {
        const first = Math.min(...sections.flatMap((s) => s.widgets.map(indexOf)).filter((i) => i >= 0));
        anchor = Number.isFinite(first) ? first : node.widgets.length;
    }
    const k = sections.findIndex((s) => s.id === open);
    if (k < 0) {
        node.widgets.splice(anchor, 0, ...hws);
        return;
    }
    const members = sections[k].widgets.map(indexOf).filter((i) => i >= 0);
    const first = Math.min(...members);
    node.widgets.splice(first, 0, ...hws.slice(0, k + 1));
    const last = Math.max(...sections[k].widgets.map(indexOf).filter((i) => i >= 0));
    node.widgets.splice(last + 1, 0, ...hws.slice(k + 1));
}

// ---------------------------------------------------------------- Start from preset...

function attachPicker(node, onApplied) {
    const box = makeBox(HEADER_H + NOTE_LINE);
    Object.assign(box.style, { gap: "1px", padding: "1px 0" });
    const select = el("select", {
        height: `${HEADER_H - 2}px`, font: "12px var(--comfy-font, sans-serif)", flex: "0 0 auto",
        color: "var(--input-text, var(--fg-color, #e6e6e6))", background: "var(--comfy-input-bg, #222)",
        border: "1px solid var(--border-color, #4a4a4a)", borderRadius: "4px", padding: "0 4px",
    });
    select.setAttribute("aria-label", "Start from preset");
    const note = captionEl();
    note.style.padding = "0 2px";
    box.append(select, note);
    for (const ev of ["pointerdown", "mousedown", "click", "wheel"]) box.addEventListener(ev, (e) => e.stopPropagation());
    const w = snug(addWidget(node, "kp_start_from_preset", box, HEADER_H + NOTE_LINE, 0));
    // under seed and its control widget (concept s5)
    const seedAt = Math.max(node.widgets.findIndex((x) => x.name === "control_after_generate"),
        node.widgets.findIndex((x) => x.name === "seed"));
    moveWidget(node, w, seedAt + 1);

    let filled = false;
    loadUI().then((ui) => {
        const opts = [el("option", {}, "Start from preset…")];
        opts[0].value = "";
        for (const name of Object.keys(ui?.presets ?? {})) {
            const o = el("option", {}, name);
            o.value = name;
            opts.push(o);
        }
        select.replaceChildren(...opts);
        filled = true;
    });
    select.addEventListener("change", async () => {
        const preset = select.value;
        select.value = "";
        if (!preset || !filled) return;
        onApplied(await applyPreset(node, preset));
    });
    w.kpPaint = (props, flash) => {
        const ref = props.kp_reference?.preset;
        const text = flash || (ref ? `Changed dots compare with ${ref}` : "Changed dots compare with the defaults");
        note.textContent = text;
        note.title = text;
    };
    return w;
}

function setWidget(n, name, v) {
    const w = n.widgets?.find((x) => x.name === name);
    if (!w || same(w.value, v)) return 0;
    w.value = v;
    try {
        w.callback?.(w.value, app.canvas, n);
    } catch (e) {
        console.warn("[KreaPhoton] widget callback", name, e);
    }
    return 1;
}

async function applyPreset(node, preset) {
    const q = params({ preset, profile: widgetValue(node, "precision_profile") ?? "auto",
        negative: inputLinked(node, "negative") });
    let data;
    try {
        const r = await fetch(`/kreaphoton/preset_values?${q}`);
        data = await r.json();
        if (!r.ok) throw new Error(data.error || r.statusText);
    } catch (e) {
        return `Could not load ${preset}: ${e.message ?? e}`;
    }
    let n = 0;
    for (const [k, v] of Object.entries(data.sampler)) n += setWidget(node, k, v);
    node.properties.kp_reference = { preset, values: data.sampler };
    const parts = [`${preset}: ${n} sampler value${n === 1 ? "" : "s"} changed`];
    const sched = upstream(node, "sigmas");
    if (cls(sched) === "KreaPhotonScheduler") {
        let m = 0;
        for (const [k, v] of Object.entries(data.scheduler)) m += setWidget(sched, k, v);
        parts.push(`Scheduler ${m} changed`);
        if (data.scheduler_needs_latent && !inputLinked(sched, "latent")) {
            parts.push("connect the latent to the Scheduler to match the Sampler's resolution-aware alpha");
        }
        sched.setDirtyCanvas?.(true, true);
    } else {
        parts.push("no KreaPhoton Scheduler on sigmas: set the schedule by hand");
    }
    if (data.note) parts.push(data.note);
    node.graph?.setDirtyCanvas?.(true, true);
    return parts.join("; ");
}

app.registerExtension({
    name: "KreaPhoton.Sections",
    settings: [
        {
            id: SETTING_SECTIONS,
            category: ["KreaPhoton", "Nodes", "Advanced sections"],
            name: "KreaPhoton: sections on Sampler (Advanced)",
            tooltip: "Group the 43 widgets of Sampler (Advanced) into collapsible sections, one open at a time. "
                + "Off shows the flat list; changed dots and dimming stay.",
            type: "boolean",
            defaultValue: true,
        },
    ],
    nodeCreated(node) {
        const c = cls(node);
        if (c === "KreaPhotonSamplerAdvanced") attachSections(node, "advanced");
        else if (c === "KreaPhotonSampler") attachSections(node, "sampler");
        else if (c === "KreaPhotonScheduler") attachSections(node, "scheduler");
    },
});
