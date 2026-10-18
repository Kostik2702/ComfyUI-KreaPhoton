# -*- coding: utf-8 -*-
"""
Plain-assert test for the optional <think> block of KreaPhoton Encode (2026-09-29):
kreaphoton/think.py (template, plan prompt, plan cleaning, composition of the block)
and the node contract - `off` keeps the exact stock tokenize call (bit-exact), `manual`
puts the user's text in the assistant think turn, `auto` asks the loaded Qwen for a
layout plan (greedy, no empty think block) and appends the manual text.

A fake clip records calls - no model, no generation.

Run: <embedded python> tests/test_think.py
"""
import importlib
import importlib.util
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KREAPHOTON_DIR = os.path.join(REPO_ROOT, "kreaphoton")
COMFYUI_ROOT = r"E:\CUI portable\ComfyUI-torch2.9-cu130-cp313-v1.2\ComfyUI"


def _load():
    if COMFYUI_ROOT not in sys.path:
        sys.path.insert(0, COMFYUI_ROOT)
    if "kreaphoton" not in sys.modules:
        spec = importlib.util.spec_from_file_location("kreaphoton", os.path.join(KREAPHOTON_DIR, "__init__.py"),
                                                      submodule_search_locations=[KREAPHOTON_DIR])
        pkg = importlib.util.module_from_spec(spec)
        sys.modules["kreaphoton"] = pkg
        spec.loader.exec_module(pkg)
    return importlib.import_module("kreaphoton.think"), importlib.import_module("kreaphoton.nodes")


class FakeClip:
    """Records tokenize / generate / decode calls; tokenize returns its own arguments."""

    def __init__(self, generated="Two people stand side by side.", ids=(1, 2, 3)):
        self.generated = generated
        self.ids = list(ids)
        self.tokenize_calls = []
        self.generate_calls = []

    def tokenize(self, text, return_word_ids=False, **kwargs):
        self.tokenize_calls.append((text, dict(kwargs)))
        return {"text": text, "kwargs": dict(kwargs)}

    def encode_from_tokens_scheduled(self, tokens):
        return [["COND", tokens]]

    def generate(self, tokens, **kwargs):
        self.generate_calls.append((tokens, dict(kwargs)))
        return self.ids

    def decode(self, ids, skip_special_tokens=True):
        return self.generated


def main():
    th, nodes = _load()
    print("=" * 78)
    print("test_think: kreaphoton.think + KreaPhoton Encode think block")
    print("=" * 78)

    stock = th.stock_template()
    # --- (1) the stock template is comfy's KREA2_TEMPLATE; the fallback copy has not drifted ---
    print("[1] stock template == comfy KREA2_TEMPLATE, one {} slot, fallback identical")
    from comfy.text_encoders.krea2 import KREA2_TEMPLATE
    assert stock == KREA2_TEMPLATE, "stock_template must be comfy's own template"
    assert th.FALLBACK_KREA2_TEMPLATE == KREA2_TEMPLATE, "fallback copy drifted from comfy"
    assert stock.count("{}") == 1 and stock.startswith("<|im_start|>system")

    # --- (2) think template: stock + assistant think turn, braces in the content escaped ---
    print("[2] think_template = stock + <think>..</think>, content braces escaped")
    t = th.think_template("Layout: A {left} and B {right}.")
    filled = t.format("PROMPT")
    assert filled == KREA2_TEMPLATE.format("PROMPT") + "<think>\nLayout: A {left} and B {right}.\n</think>\n\n", filled
    assert "<|im_start|>system" in filled and filled.count("<|im_start|>") == 3, "system turn must stay (prefix strip)"

    # --- (3) raw-template detection mirrors comfy (qwen3vl.py: text.startswith('<|im_start|>')) ---
    print("[3] is_raw_template")
    assert th.is_raw_template("<|im_start|>system\nx")
    assert not th.is_raw_template(" <|im_start|>system") and not th.is_raw_template("a photo")

    # --- (4) plan cleaning: reasoning stripped, whitespace collapsed, role-word junk -> empty ---
    print("[4] clean_plan")
    assert th.clean_plan("  Two people.\n\nThe man   holds it. ") == "Two people. The man holds it."
    assert th.clean_plan("<think>hmm</think>\nOne person, left third.") == "One person, left third."
    for junk in ("user", " assistant\n", "", "   ", "system"):
        assert th.clean_plan(junk) == "", junk
    long = "word " * 400
    assert len(th.clean_plan(long)) <= th.PLAN_MAX_CHARS

    # --- (5) compose: off / manual / auto (+ manual appended) ---
    print("[5] compose_think")
    assert th.compose_think("off", "x", "plan") == ""
    assert th.compose_think("manual", "  keep  ", "plan") == "keep"
    assert th.compose_think("manual", "   ", "plan") == ""
    assert th.compose_think("auto", "", "Plan.") == "Plan."
    assert th.compose_think("auto", "Extra.", "Plan.") == "Plan. Extra."
    assert th.compose_think("auto", "Extra.", "") == "Extra."

    # --- (6) generate_plan: greedy, no empty think block, the plan system prompt, the prompt text ---
    print("[6] generate_plan calls tokenize(thinking=True, system_prompt=PLAN) + greedy generate")
    fc = FakeClip(generated="A woman and a man stand.")
    plan, dropped = th.generate_plan(fc, "A woman and a man.")
    assert plan == "A woman and a man stand." and dropped == []
    (text, kw), = fc.tokenize_calls
    assert text == "A woman and a man." and kw.get("thinking") is True
    assert kw.get("system_prompt") == th.PLAN_SYSTEM_PROMPT and "llama_template" not in kw
    (_, gkw), = fc.generate_calls
    assert gkw.get("do_sample") is False and gkw.get("mtp") is False
    assert gkw.get("max_length") == th.PLAN_MAX_TOKENS

    # --- (6b) generate_plan filters against the prompt; a generation that hit the token cap
    #          loses its unfinished last sentence ---
    print("[6b] generate_plan: filtered plan, truncation detected from the token count")
    fc = FakeClip(generated="Two cats sleep on a sofa.  \nNo other objects are present.  \nThe sofa is",
                  ids=range(th.PLAN_MAX_TOKENS))
    plan, dropped = th.generate_plan(fc, "Two cats sleeping on a sofa.")
    assert plan == "Two cats sleep on a sofa.", plan
    assert dropped == ["No other objects are present.", "The sofa is"], dropped

    # --- (13) filter: negations and talk about the prompt are dropped (real Qwen outputs,
    #          research/think_spike/probe_v2.log) ---
    print("[13] filter_plan drops negations and meta sentences")
    assert th.filter_plan("The beach is empty. There are no people. There are no boats.",
                          "An empty beach at dawn with no people and no boats.") == (
        "The beach is empty.", ["There are no people.", "There are no boats."])
    assert th.filter_plan("Three coworkers are in a bright modern office kitchen. No specific positions, "
                          "directions, or interactions are described.",
                          "Three coworkers having a coffee break in a bright modern office kitchen.") == (
        "Three coworkers are in a bright modern office kitchen.",
        ["No specific positions, directions, or interactions are described."])
    assert th.filter_plan("The cat isn't moving. A cat sits. The dog doesn’t bark.", "A cat sits on a step.") == (
        "A cat sits.", ["The cat isn't moving.", "The dog doesn’t bark."])
    assert th.filter_plan("A still life. The layout is unspecified.", "Moody still life, soft light.") == (
        "A still life.", ["The layout is unspecified."])

    # --- (14) filter: a frame position, direction or shot word absent from the prompt means the
    #          model chose it - the sentence is dropped; words the prompt uses are kept ---
    print("[14] filter_plan drops layout the prompt does not state, keeps layout it does")
    assert th.filter_plan("A family of four walks along a sandy beach. They are positioned in the foreground. "
                          "The sunset is behind them.", "A family of four walking along a sandy beach at sunset.") == (
        "A family of four walks along a sandy beach.",
        ["They are positioned in the foreground.", "The sunset is behind them."])
    assert th.filter_plan("A cyclist rests on the left edge of a mountain road. The cyclist faces toward the right. "
                          "A wide valley view extends across the background.",
                          "A cyclist resting on the left edge of a mountain road with a wide valley view.") == (
        "A cyclist rests on the left edge of a mountain road.",
        ["The cyclist faces toward the right.", "A wide valley view extends across the background."])
    assert th.filter_plan("Two friends are chatting. The shot is medium, at eye level.",
                          "Two friends chatting in a sunny city park in the afternoon, candid photo.") == (
        "Two friends are chatting.", ["The shot is medium, at eye level."])
    long_prompt = ("A tall woman in a long green coat stands on the left third of the frame on a wet cobblestone "
                   "street at night, her back to the camera, looking at a lit bakery window on the right; a cyclist "
                   "passes in the background, shot from a low angle with a wide lens.")
    kept = ("A tall woman stands on the left third of the frame, her back to the camera. A cyclist passes in the "
            "background. The shot is taken from a low angle with a wide lens.")
    assert th.filter_plan(kept, long_prompt) == (kept, [])
    assert th.filter_plan("The vase is in the center of the table.", "A vase centered on a table.") == (
        "The vase is in the center of the table.", [])

    # --- (15) filter: a count the prompt does not give is dropped; digits == words; "one" is free ---
    print("[15] filter_plan drops counts the prompt does not give")
    assert th.filter_plan("Two people dance. A woman and a man are dancing.", "A woman and a man dancing in a kitchen.") == (
        "A woman and a man are dancing.", ["Two people dance."])
    assert th.filter_plan("Four people walk on a beach.", "A family of 4 on a beach.") == ("Four people walk on a beach.", [])
    assert th.filter_plan("Three friends sit at a table. One friend laughs.", "Three friends at a table.") == (
        "Three friends sit at a table. One friend laughs.", [])
    assert th.filter_plan("Four friends sit.", "Three friends at a table.") == ("", ["Four friends sit."])
    assert th.filter_plan("Fifteen birds fly.", "Birds over a lake.") == ("", ["Fifteen birds fly."])
    assert th.filter_plan("Fifteen birds fly.", "15 birds over a lake.") == ("Fifteen birds fly.", [])

    # --- (16) filter: a refusal discards the whole plan ---
    print("[16] filter_plan: a refusal -> no plan")
    refusal = "I'm sorry, but I can't help with that request."
    assert th.filter_plan(refusal, "a photo") == ("", [refusal])
    refusal = "As an AI, I cannot create images. A photo of a cat."
    assert th.filter_plan(refusal, "a photo of a cat") == ("", [refusal])

    # --- (17) filter: the unfinished last sentence goes only when generation hit the cap ---
    print("[17] filter_plan: truncated -> unfinished last sentence dropped; untruncated kept as is")
    assert th.filter_plan("Two cats sleep on a sofa. The sofa is", "Two cats sleeping on a sofa.", truncated=True) == (
        "Two cats sleep on a sofa.", ["The sofa is"])
    assert th.filter_plan("Two cats sleep on a sofa", "Two cats sleeping on a sofa.") == ("Two cats sleep on a sofa", [])
    assert th.filter_plan("", "a photo") == ("", [])

    enc = nodes.KreaPhotonEncode()

    # --- (7) node: off (default) keeps the exact stock call - bit-exact with v1.8.0 ---
    print("[7] Encode off: stock tokenize call, no kwargs, empty think output")
    fc = FakeClip()
    out = enc.encode(fc, "a photo", "off")
    assert fc.tokenize_calls == [("a photo", {})], fc.tokenize_calls
    assert fc.generate_calls == [] and len(out) == 2 and out[1] == ""
    fc = FakeClip()
    enc.encode(fc, "a photo", "natural", think="off", think_text="ignored")
    assert fc.tokenize_calls == [(nodes.STYLE_DIRECTIVES["natural"] + "a photo", {})]

    # --- (8) node: manual ---
    print("[8] Encode manual: llama_template with the think block; empty manual == off")
    fc = FakeClip()
    out = enc.encode(fc, "a photo", "off", think="manual", think_text="  Woman left, man right.  ")
    (text, kw), = fc.tokenize_calls
    assert text == "a photo" and kw == {"llama_template": th.think_template("Woman left, man right.")}
    assert out[1] == "Woman left, man right." and fc.generate_calls == []
    fc = FakeClip()
    out = enc.encode(fc, "a photo", "off", think="manual", think_text="   ")
    assert fc.tokenize_calls == [("a photo", {})] and out[1] == ""

    # --- (9) node: auto (+ manual appended), plan generated from the prompt text only ---
    print("[9] Encode auto: plan from Qwen, manual text appended, directive not sent to the planner")
    fc = FakeClip(generated="Exactly three people around the table.")
    out = enc.encode(fc, "Three friends in a cafe.", "cinematic", think="auto", think_text="All faces visible.")
    plan_call, enc_call = fc.tokenize_calls
    assert plan_call[0] == "Three friends in a cafe.", "planner sees the prompt, not the style directive"
    want = "Exactly three people around the table. All faces visible."
    assert enc_call == (nodes.STYLE_DIRECTIVES["cinematic"] + "Three friends in a cafe.",
                        {"llama_template": th.think_template(want)}), enc_call
    assert out[1] == want and len(fc.generate_calls) == 1

    # --- (10) auto with an empty/junk generation and no manual text -> the stock path ---
    print("[10] Encode auto, junk generation ('user') -> falls back to the stock call")
    fc = FakeClip(generated="user")
    out = enc.encode(fc, "a photo", "off", think="auto")
    assert fc.tokenize_calls[-1] == ("a photo", {}) and out[1] == ""

    # --- (10b) auto whose every sentence is filtered out, no manual text -> the stock path;
    #           with manual text only the manual text is used ---
    print("[10b] Encode auto, every plan sentence filtered -> stock call / manual text only")
    fc = FakeClip(generated="There are no people. The shot is medium.")
    out = enc.encode(fc, "An empty beach.", "off", think="auto")
    assert fc.tokenize_calls[-1] == ("An empty beach.", {}) and out[1] == "", fc.tokenize_calls
    fc = FakeClip(generated="There are no people.")
    out = enc.encode(fc, "An empty beach.", "off", think="auto", think_text="Wide empty sand.")
    assert fc.tokenize_calls[-1] == ("An empty beach.", {"llama_template": th.think_template("Wide empty sand.")})
    assert out[1] == "Wide empty sand."

    # --- (11) a raw template on the input wins: think ignored, no generation ---
    print("[11] raw <|im_start|> prompt: think ignored")
    raw = KREA2_TEMPLATE.format("a photo")
    fc = FakeClip()
    out = enc.encode(fc, raw, "off", think="auto", think_text="x")
    assert fc.tokenize_calls == [(raw, {})] and fc.generate_calls == [] and out[1] == ""

    # --- (12) INPUT_TYPES append-only, outputs: conditioning first, think second ---
    print("[12] INPUT_TYPES append-only, RETURN_TYPES (CONDITIONING, STRING)")
    it = nodes.KreaPhotonEncode.INPUT_TYPES()
    assert list(it["required"]) == ["clip", "text", "style"]
    assert list(it["optional"]) == ["custom_directive", "think", "think_text"]
    assert list(it["optional"]["think"][0]) == ["off", "manual", "auto"]
    assert it["optional"]["think"][1]["default"] == "off" and it["optional"]["think_text"][1]["default"] == ""
    assert nodes.KreaPhotonEncode.RETURN_TYPES == ("CONDITIONING", "STRING")
    assert nodes.KreaPhotonEncode.RETURN_NAMES == ("CONDITIONING", "think")

    print("\ntest_think: ALL ASSERTS PASSED")


if __name__ == "__main__":
    main()
