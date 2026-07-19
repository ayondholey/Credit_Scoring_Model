#!/usr/bin/env python3
"""Main entry point for Credit Scoring Model."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.logging_config import setup_logging
from src.pipeline.run_pipeline import main

if __name__ == "__main__":
    setup_logging()
    main()