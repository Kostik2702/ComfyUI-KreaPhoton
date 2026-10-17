// KreaPhoton node colours (node-ui-concept P1, 2026-10-17).
// A node takes the colour of the part of the sigma trajectory it can change: composition blue
// (can move the layout, from sigma 1.0), identity green (from 0.85 down), texture amber (below
// ~0.65: cannot move layout or identity). Nodes outside the trajectory (graphite in the concept)
// keep the theme's default colour. LoRA Phase follows its `phase` widget live.
//
// Both color (title) and bgcolor (body) are set: the ComfyUI frontend (1.53, checked live
// 2026-10-17) paints the whole node from bgcolor, a title-only colour is not visible. Body tints
// are muted so widget text keeps its contrast.
//
// Colours are saved into workflows by LiteGraph, so the extension only colours nodes that
// have no colour yet and marks them with properties.kp_palette; a colour the user picked is
// never touched. The "KreaPhoton: node colours" setting turns the palette on and off.
import { app } from "../../scripts/app.js";

const SETTING_ID = "KreaPhoton.NodeColors";
const MARK = "kp_palette";
const PALETTE_VERSION = 2;

// title tints from docs/design/node-ui-concept.md section 3; body = the same hue, muted
const TITLE = {
    dark: { composition: "#1f3f66", identity: "#14503c", texture: "#6e2f14" },
    light: { composition: "#c9dcf3", identity: "#c4e6d7", texture: "#f2d2c3" },
};
const BODY = {
    dark: { composition: "#1b2b40", identity: "#17342b", texture: "#3b2519" },
    light: { composition: "#e6eefa", identity: "#e3f3ec", texture: "#faece5" },
};

// node type -> family; LoRA Phase is resolved from its phase widget
export const FAMILY = {
    KreaPhotonEmptyLatent: "neutral",
    KreaPhotonEncode: "neutral",
    KreaPhotonLoraPhase: "phase",
    KreaPhotonCompositionDrafts: "composition",
    KreaPhotonSampler: "composition",
    KreaPhotonSamplerAdvanced: "composition",
    KreaPhotonScheduler: "composition",
    KreaPhotonUpscaleV2: "texture",
    KreaPhotonUpscale: "texture",
    KreaPhotonFaceDetailer: "texture",
    KreaPhotonSaveImage: "neutral",
};

// LoRA Phase `phase` value -> family (composition+identity starts at sigma 1.0: it can move the layout)
const PHASE_FAMILY = {
    composition: "composition",
    "composition+identity": "composition",
    identity: "identity",
    texture: "texture",
    all: "neutral",
};

function enabled(override) {
    if (typeof override === "boolean") return override;   // the setting's onChange passes the new value
    try {
        return app.extensionManager?.setting?.get(SETTING_ID) ?? true;
    } catch {
        return true;
    }
}

function themeName() {
    try {
        const id = String(app.extensionManager?.setting?.get("Comfy.ColorPalette") ?? "dark");
        return id.toLowerCase().includes("light") ? "light" : "dark";
    } catch {
        return "dark";
    }
}

function familyOf(node) {
    const fam = FAMILY[node.comfyClass ?? node.type];
    if (fam !== "phase") return fam;
    const w = node.widgets?.find((x) => x.name === "phase");
    return PHASE_FAMILY[w?.value] ?? "neutral";
}

function ours(node) {
    return node.properties?.[MARK] !== undefined;
}

function unpaint(node) {
    if (!ours(node)) return;
    // assignment, not delete: color / bgcolor are prototype accessors in the frontend's LiteGraph,
    // `delete` on the instance leaves the colour in place (checked live 2026-10-17)
    node.color = undefined;
    node.bgcolor = undefined;
    delete node.properties[MARK];
    node.setDirtyCanvas?.(true, true);
}

function paint(node, on) {
    const fam = familyOf(node);
    if (!fam) return;
    if (!enabled(on)) return unpaint(node);
    // a colour that is not ours belongs to the user
    if ((node.color || node.bgcolor) && !ours(node)) return;
    if (fam === "neutral") return unpaint(node);      // outside the trajectory: theme default
    const theme = themeName();
    node.color = TITLE[theme][fam];
    node.bgcolor = BODY[theme][fam];
    node.properties = node.properties ?? {};
    node.properties[MARK] = PALETTE_VERSION;
    node.setDirtyCanvas?.(true, true);
}

function repaintAll(on) {
    for (const node of app.graph?._nodes ?? []) {
        if (FAMILY[node.comfyClass ?? node.type]) paint(node, on);
    }
}

// a palette switch re-tints our nodes at once (the old settings API emits "<id>.change")
try {
    app.ui?.settings?.addEventListener?.("Comfy.ColorPalette.change", () => setTimeout(() => repaintAll(), 50));
} catch {
    /* older / newer frontends without the event: the next graph load re-tints */
}

function followPhase(node) {
    const w = node.widgets?.find((x) => x.name === "phase");
    if (!w || w.__kpFollow) return;
    const cb = w.callback;
    w.callback = function () {
        const r = cb ? cb.apply(this, arguments) : undefined;
        paint(node);
        return r;
    };
    w.__kpFollow = true;
}

app.registerExtension({
    name: "KreaPhoton.NodeColors",
    settings: [
        {
            id: SETTING_ID,
            category: ["KreaPhoton", "Nodes", "Colours"],
            name: "KreaPhoton: node colours by trajectory phase",
            tooltip: "Blue can change the layout, green the identity, amber only texture; graphite is outside "
                + "the sampling trajectory. Colours you set yourself are never changed.",
            type: "boolean",
            defaultValue: true,
            onChange: (value) => repaintAll(value),
        },
    ],
    // after a workflow is configured (saved colours and properties restored) re-tint our nodes for
    // the current theme; loadedGraphNode alone was not enough on frontend 1.53 (checked live)
    afterConfigureGraph() {
        repaintAll();
    },
    nodeCreated(node) {
        if (!FAMILY[node.comfyClass ?? node.type]) return;
        if (FAMILY[node.comfyClass ?? node.type] === "phase") followPhase(node);
        // deferred: when a workflow loads, nodeCreated runs BEFORE configure() restores the saved
        // colour and properties; painting now would be overwritten without the mark and the next
        // save would turn our colour into a "user" colour
        setTimeout(() => paint(node), 0);
    },
    loadedGraphNode(node) {
        if (!FAMILY[node.comfyClass ?? node.type]) return;
        if (FAMILY[node.comfyClass ?? node.type] === "phase") followPhase(node);
        paint(node);
    },
});
