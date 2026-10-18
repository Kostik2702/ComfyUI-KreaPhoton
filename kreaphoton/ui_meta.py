"""Node-library metadata shared by every node module (node-ui-concept P0).

Categories are numbered to name the fixed order the pack is used in: setup -> preview ->
sampling -> finishing -> output. The library itself lists sub-categories in node REGISTRATION
order (frontend 1.53.6, checked live 2026-10-17), so NODE_CLASS_MAPPINGS keeps the pipeline
order too. Category and display name are not stored in saved workflows, so changing them never
breaks a workflow.
"""

ROOT = "KreaPhoton"
CAT_SETUP = ROOT + "/1 Setup"
CAT_PREVIEW = ROOT + "/2 Preview"
CAT_SAMPLING = ROOT + "/3 Sampling"
CAT_FINISHING = ROOT + "/4 Finishing"
CAT_OUTPUT = ROOT + "/5 Output"
CATEGORIES = (CAT_SETUP, CAT_PREVIEW, CAT_SAMPLING, CAT_FINISHING, CAT_OUTPUT)
