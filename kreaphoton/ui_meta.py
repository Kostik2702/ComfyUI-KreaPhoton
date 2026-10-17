"""Node-library metadata shared by every node module (node-ui-concept P0).

Categories are numbered because the library sorts them alphabetically and the pack is used in
a fixed order: setup -> preview -> sampling -> finishing -> output. Category and display name
are not stored in saved workflows, so changing them never breaks a workflow.
"""

ROOT = "KreaPhoton"
CAT_SETUP = ROOT + "/1 Setup"
CAT_PREVIEW = ROOT + "/2 Preview"
CAT_SAMPLING = ROOT + "/3 Sampling"
CAT_FINISHING = ROOT + "/4 Finishing"
CAT_OUTPUT = ROOT + "/5 Output"
CATEGORIES = (CAT_SETUP, CAT_PREVIEW, CAT_SAMPLING, CAT_FINISHING, CAT_OUTPUT)
