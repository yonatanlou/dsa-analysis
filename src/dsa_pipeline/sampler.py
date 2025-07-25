"""Data sampling functionality for DSA pipeline."""

import logging
import random
from pathlib import Path
from typing import List

import pandas as pd
import tqdm


def sample_data(csv_files: List[Path], sample_ratio: float, output_dir: Path, chunk_name: str) -> Path:
    """Sample data from CSV files and save to parquet."""
    logger = logging.getLogger(__name__)
    
    if not csv_files:
        raise ValueError("No CSV files provided for sampling")
    
    logger.info(f"Sampling {len(csv_files)} CSV files with ratio {sample_ratio:.1%}")
    
    # Collect samples from all CSV files
    samples = []
    total_rows = 0
    sampled_rows = 0
    
    # Limit to sample_ratio of files for performance
    csv_files_sampled = random.sample(csv_files, int(len(csv_files) * sample_ratio))
    logger.info(f"Sampling {len(csv_files_sampled)} out of {len(csv_files)} CSV files")
    
    for i, csv_file in tqdm.tqdm(enumerate(csv_files_sampled), total=len(csv_files_sampled), desc="Sampling CSV files"):
        logger.debug(f"Processing file {i+1}/{len(csv_files_sampled)}: {csv_file.name}")
        
        try:
            # Read CSV with optimized dtypes for memory efficiency
            df = pd.read_csv(
                csv_file,
                dtype={
                    'platform_name': 'category',
                    'category': 'category', 
                    'decision_ground': 'category',
                    'content_type': 'category',
                    'source_type': 'category',
                    'automated_detection': 'category',
                    'automated_decision': 'category'
                },
                low_memory=False
            )
            
            file_rows = len(df)
            total_rows += file_rows
            
            if file_rows == 0:
                logger.warning(f"Empty CSV file: {csv_file.name}")
                continue
            
            samples.append(df)
            sampled_rows += len(df)
            
        except Exception as e:
            logger.error(f"Failed to process {csv_file.name}: {e}")
            continue
    
    if not samples:
        raise ValueError("No valid samples collected from CSV files")
    
    # Combine all samples
    logger.info(f"Combining {len(samples)} sample DataFrames")
    combined_df = pd.concat(samples, ignore_index=True)
    
    # Save to parquet with compression, fallback to CSV
    output_file = output_dir / f"sample_{chunk_name}.parquet"
    logger.info(f"Saving {len(combined_df):,} sampled rows to {output_file}")
    
    try:
        combined_df.to_parquet(output_file, compression='snappy', index=False)
    except ImportError:
        # Fallback to CSV if parquet support not available
        logger.warning("Parquet support not available, saving as CSV")
        output_file = output_dir / f"sample_{chunk_name}.csv"
        combined_df.to_csv(output_file, index=False)
    
    # Log sampling statistics
    actual_ratio = sampled_rows / total_rows if total_rows > 0 else 0
    logger.info(f"Sampling complete: {sampled_rows:,}/{total_rows:,} rows ({actual_ratio:.1%})")
    
    return output_file