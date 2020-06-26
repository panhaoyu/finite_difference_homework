import os as _os

FIGURE_PATH = _os.path.join(_os.path.expanduser('~'), 'Desktop', 'Finite difference homework')
not _os.path.exists(FIGURE_PATH) and _os.makedirs(FIGURE_PATH)
