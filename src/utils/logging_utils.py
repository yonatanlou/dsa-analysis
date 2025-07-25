"""Logging utilities for DSA pipeline."""

import logging
import sys


def setup_logging(verbose: bool = False) -> None:
    """Set up basic logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('dsa_sampler.log')
        ]
    )