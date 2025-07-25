"""Data download functionality for DSA pipeline."""

import logging
import subprocess
from pathlib import Path
from typing import List


def download_data(start_date: str, end_date: str, data_dir: Path) -> List[Path]:
    """Download DSA data for date range using shantay."""
    logger = logging.getLogger(__name__)
    
    # Create data directory
    data_dir.mkdir(exist_ok=True)
    
    # Create unique staging directory to avoid lock conflicts
    staging_dir = data_dir.parent / f"staging_{start_date}_{end_date}"
    staging_dir.mkdir(exist_ok=True)
    
    # Clean up any existing lock file
    lock_file = staging_dir / "staging.lock"
    if lock_file.exists():
        logger.warning(f"Removing existing lock file: {lock_file}")
        lock_file.unlink()
    
    # Build shantay command with separate staging directory
    cmd = [
        'uvx', 'shantay', 'download',
        '--first', start_date,
        '--last', end_date,
        '--archive', str(data_dir),
        '--staging', str(staging_dir)
    ]
    
    logger.info(f"Downloading data: {start_date} to {end_date}")
    logger.debug(f"Command: {' '.join(cmd)}")
    
    try:
        # Run shantay download
        result = subprocess.run(
            cmd, 
            check=True, 
            capture_output=False, 
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.stdout:
            logger.debug(f"Download output: {result.stdout}")
        
        # Find downloaded zip files
        zip_files = list(data_dir.glob('**/sor-global-*.zip'))
        logger.info(f"Downloaded {len(zip_files)} files")
        
        return zip_files
        
    except subprocess.TimeoutExpired:
        logger.error("Download timeout exceeded (1 hour)")
        raise
    except subprocess.CalledProcessError as e:
        logger.error(f"Download failed: {e.stderr}")
        raise
    except Exception as e:
        logger.error(f"Download error: {e}")
        raise