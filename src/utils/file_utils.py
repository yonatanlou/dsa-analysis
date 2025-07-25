"""File management utilities for DSA pipeline."""

import logging
import shutil
from pathlib import Path


def cleanup_files(*paths: Path) -> None:
    """Clean up files and directories."""
    logger = logging.getLogger(__name__)
    
    for path in paths:
        if not path.exists():
            continue
            
        try:
            if path.is_file():
                path.unlink()
                logger.debug(f"Removed file: {path}")
            elif path.is_dir():
                shutil.rmtree(path)
                logger.debug(f"Removed directory: {path}")
        except Exception as e:
            logger.warning(f"Failed to remove {path}: {e}")