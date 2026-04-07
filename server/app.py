"""
Server entrypoint — re-exports FastAPI app from root app.py.
This structure satisfies multi-mode deployment requirements.
"""

import sys
import os

# Add parent directory to path to import root app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app, main

__all__ = ["app", "main"]

if __name__ == "__main__":
    main()
