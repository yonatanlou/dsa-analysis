#!/usr/bin/env python3
"""Entry point script for DSA data sampling pipeline."""

import sys
from pathlib import Path

# Add src directory to Python path
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

from dsa_pipeline.cli import main

if __name__ == '__main__':
    main()