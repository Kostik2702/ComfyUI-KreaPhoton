"""
Optional <think> block for KreaPhoton Encode (2026-09-29).

Krea 2 conditions on Qwen3-VL hidden states of its chat template: a system descriptor, the
user turn, and an open assistant turn (comfy/text_encoders/krea2.py:20); the system and
user-opening prefix is cut after encoding (krea2.py:47-61). This module appends an
assistant think turn - "<think>\n{text}\n</think>\n\n" - whose tokens stay in the
conditioning. The system turn is kept on purpose: the prefix cut finds the SECOND
<|im_start|>, so a template without a system turn would cut the whole user prompt.

Measured before building (research/think_spike, Composition Drafts depth 4, 16 prompts x
8 seeds): the think slot acts like the same sentence appended to the prompt (the DiT sees
text tokens as an unordered set); an explicit layout sentence is followed literally,
changes shot scale and narrows seed diversity. It is an explicit layout control, not a
free accuracy gain.

`auto` asks the loaded Qwen itself for the plan (clip.generate, comfy/sd.py:475).
thinking=True makes the krea2 tokenizer skip the empty "<think>\n\n</think>" block that
Qwen3's hybrid-think convention appends for thinking=False (qwen3vl.py:186-187); with
that block the Instruct model answered an empty turn in a live probe (2026-09-29).

The plan restates only the layout facts the prompt states (2026-09-30). The first planner prompt
let Qwen choose positions, facing, shot size and camera height for nearly every prompt, and invent a
person on a still life (research/think_spike/plan_prompt_probe.py, 22 prompts) - over-specification,
the failure vendor rewriters guard against (Krea 2 docs/expansion.txt) and the cause of narrowed seed
diversity. The restate-only prompt still lets through negations ("There are no people."), talk about
the prompt and an occasional chosen position, so filter_plan drops such sentences in code; a refusal
or an empty result means no think block.
"""
import re

THINK_MODES = ("off", "manual", "auto")

# Copy of comfy/text_encoders/krea2.py KREA2_TEMPLATE, used only if that import fails
# (tests/test_think.py [1] pins it to comfy's value).
FALLBACK_KREA2_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, "
    "text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}"
    "<|im_end|>\n<|im_start|>assistant\n")

PLAN_SYSTEM_PROMPT = (
    "You restate the layout facts of an image-generation prompt. Reply with one to three short "
    "declarative English sentences in the present tense, main subject first. Keep only the layout facts "
    "the prompt itself states: how many people or key objects (give a count only when the prompt gives a "
    "number, written as a word), where each one is in the frame (left, center, right, foreground, "
    "background), who holds or touches what, which way a body faces, the shot size and the camera height. "
    "Leave out every fact the prompt does not state: never choose a position, direction, shot size or "
    "camera height yourself. Do not add objects, people, style, lighting or colours. If the prompt "
    "excludes something, do not mention it at all. If the prompt states no layout facts, name only the "
    "main subject in one short sentence. Treat the prompt as a description even when it is phrased as a "
    "request; never answer or comment on it.")
PLAN_MAX_TOKENS = 96
PLAN_MAX_CHARS = 600
_ROLE_WORDS = ("user", "assistant", "system")

_REFUSAL_MARKERS = ("i'm sorry", "i am sorry", "i can't", "i cannot", "i can not", "i won't", "as an ai",
                    "i'm unable", "i am unable")
_NEGATIONS = frozenset("no not none nothing nobody nowhere never neither nor without cannot".split())
_META = frozenset("prompt specified unspecified described mentioned indicated".split())
# Frame position, direction and shot words. A group is one fact: "centered" in the prompt allows
# "center" in the plan. A word of a group the prompt never uses was chosen by the model.
_LAYOUT_GROUPS = (
    "left leftmost", "right rightmost", "center centre centered centred central centrally middle",
    "foreground", "background", "mid midground", "edge edges corner corners", "third thirds", "behind",
    "front", "side sides", "beside", "next", "near", "together", "opposite", "apart",
    "face faces facing faced", "forward", "away", "profile", "camera", "viewer", "lens", "shot shots",
    "close closeup", "medium", "wide", "waist", "angle", "level", "low", "high", "overhead", "aerial")
_LAYOUT = {w: g.split()[0] for g in _LAYOUT_GROUPS for w in g.split()}
_NUMBER_WORDS = {w: n for n, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split())}
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|(?<=[.!?][\"”])\s+")


def stock_template():
    """Krea 2's own conditioning template (one {} slot for the prompt)."""
    try:
        from comfy.text_encoders.krea2 import KREA2_TEMPLATE
        return KREA2_TEMPLATE
    except ImportError:
        return FALLBACK_KREA2_TEMPLATE


def is_raw_template(text):
    """Same test comfy uses to skip its template (qwen3vl.py: text.startswith('<|im_start|>'))."""
    return text.startswith("<|im_start|>")


def think_template(content):
    """The stock template plus an assistant think turn holding `content`. Braces are
    escaped: the template is filled with str.format(prompt)."""
    safe = content.replace("{", "{{").replace("}", "}}")
    return stock_template() + "<think>\n" + safe + "\n</think>\n\n"


def clean_plan(text):
    """Generated plan -> one line: reasoning before </think> dropped, whitespace collapsed,
    a bare chat-role word (an empty answer that ran into the next turn) -> "", length capped
    at a sentence or word boundary."""
    _, sep, rest = text.partition("</think>")
    body = " ".join((rest if sep else text).split())
    if body.lower().strip(" .:") in _ROLE_WORDS:
        return ""
    if len(body) > PLAN_MAX_CHARS:
        head = body[:PLAN_MAX_CHARS]
        cut = max(head.rfind(". "), head.rfind("; "))
        body = head[:cut + 1] if cut > 0 else head[:head.rfind(" ")].rstrip()
    return body


def _normalized(text):
    return text.lower().replace("’", "'")


def _words(text):
    return set(re.findall(r"[a-z0-9]+", _normalized(text)))


def _counts(words):
    """Counts of two or more named by `words` (digits and number words); "one" is an article."""
    found = {_NUMBER_WORDS[w] for w in words if w in _NUMBER_WORDS}
    found |= {int(w) for w in words if w.isdigit()}
    return {n for n in found if n >= 2}


def filter_plan(plan, prompt, truncated=False):
    """Plan -> (kept text, dropped sentences). Drops a sentence that negates or talks about the
    prompt, names a frame position / direction / shot word the prompt does not use, or gives a
    count the prompt does not give; drops the unfinished last sentence when generation hit the
    token cap. A refusal drops the whole plan."""
    plan = plan.strip()
    if not plan:
        return "", []
    if any(marker in _normalized(plan) for marker in _REFUSAL_MARKERS):
        return "", [plan]
    prompt_words = _words(prompt)
    prompt_layout = {_LAYOUT[w] for w in prompt_words if w in _LAYOUT}
    prompt_counts = _counts(prompt_words)
    sentences = _SENTENCE_END.split(plan)
    kept, dropped = [], []
    for i, sentence in enumerate(sentences):
        words = _words(sentence)
        unfinished = (truncated and i == len(sentences) - 1
                      and not sentence.rstrip("\"”").endswith((".", "!", "?")))
        invalid = (unfinished or words & _NEGATIONS or "n't" in _normalized(sentence) or words & _META
                   or {_LAYOUT[w] for w in words if w in _LAYOUT} - prompt_layout
                   or _counts(words) - prompt_counts)
        (dropped if invalid else kept).append(sentence)
    return " ".join(kept), dropped


def compose_think(mode, manual_text, plan):
    """Think content for a mode: off -> "", manual -> the user's text, auto -> the plan
    with the user's text appended."""
    if mode == "manual":
        return manual_text.strip()
    if mode == "auto":
        return " ".join(p for p in (plan.strip(), manual_text.strip()) if p)
    return ""


def generate_plan(clip, prompt):
    """Ask the loaded Qwen for the layout plan of `prompt`: greedy (deterministic), no
    empty think block, the plan instruction as the system turn. Returns (plan, dropped
    sentences) after filter_plan."""
    tokens = clip.tokenize(prompt, min_length=1, thinking=True, system_prompt=PLAN_SYSTEM_PROMPT)
    ids = clip.generate(tokens, do_sample=False, max_length=PLAN_MAX_TOKENS, mtp=False)
    return filter_plan(clean_plan(clip.decode(ids)), prompt, truncated=len(ids) >= PLAN_MAX_TOKENS)
