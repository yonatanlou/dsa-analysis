"""File extraction functionality for DSA pipeline."""

import logging
import zipfile
from pathlib import Path
from typing import List

import tqdm


def extract_files(zip_files: List[Path], temp_dir: Path) -> List[Path]:
    """Extract nested zip files and return list of CSV files."""
    logger = logging.getLogger(__name__)
    csv_files = []
    
    for zip_file in zip_files:
        logger.info(f"Extracting {zip_file.name}")
        
        # Create extraction directory for this zip (use just filename, not full path)
        extract_dir = temp_dir / zip_file.name.replace('.zip', '')
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Extract main zip file
            with zipfile.ZipFile(zip_file, 'r') as zf:
                zf.extractall(extract_dir)
            
            # Find and extract nested CSV zip files
            nested_zips = list(extract_dir.glob('*.csv.zip'))
            logger.debug(f"Found {len(nested_zips)} nested zip files")
            
            for nested_zip in tqdm.tqdm(nested_zips, desc="Extracting nested csv's"):
                try:
                    with zipfile.ZipFile(nested_zip, 'r') as nzf:
                        # Extract to same directory
                        nzf.extractall(extract_dir)
                        
                    # Remove the nested zip after extraction
                    nested_zip.unlink()
                    
                except zipfile.BadZipFile:
                    logger.warning(f"Skipping corrupted zip: {nested_zip}")
                    continue
            
            # Collect CSV files
            csv_files.extend(list(extract_dir.glob('*.csv')))
            
        except zipfile.BadZipFile:
            logger.error(f"Corrupted zip file: {zip_file}")
            raise
        except Exception as e:
            logger.error(f"Extraction failed for {zip_file}: {e}")
            raise
    
    logger.info(f"Extracted {len(csv_files)} CSV files")
    return csv_files