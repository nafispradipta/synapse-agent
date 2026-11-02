#!/usr/bin/env python3
"""
Synapse Memory Engine - Model Context Protocol (MCP) Server Launcher.
Delegates to synapse.mcp for global and packaged CLI compatibility.
"""

import sys
from pathlib import Path

# Ensure package root is in sys.path when running uninstalled from source
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from synapse.mcp import main

if __name__ == "__main__":
    main()
