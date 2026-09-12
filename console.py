"""
Make stdout safe for the characters this project prints.

Scripts print arrows, box-drawing separators and Greek letters. On Windows the
default console encoding is cp1252, which cannot represent them, so every
documented entry point died with UnicodeEncodeError before producing output —
on its own status messages, not on anything that mattered.

Reconfiguring to UTF-8 with replacement keeps the output readable everywhere
and means a console that still cannot render a glyph degrades to '?' instead of
taking the process down.
"""

import sys


def init() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


init()
