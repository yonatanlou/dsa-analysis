"""Command-line interface for DSA pipeline."""

import argparse
import logging
import sys
from pathlib import Path
from time import time

from utils.logging_utils import setup_logging
from utils.file_utils import cleanup_files
from dsa_pipeline.date_utils import generate_date_chunks
from dsa_pipeline.downloader import download_data
from dsa_pipeline.extractor import extract_files
from dsa_pipeline.sampler import sample_data


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description='DSA Data Sampling Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --start 2025-01-01 --end 2025-01-05 --ratio 0.1
  %(prog)s --start 2025-01-01 --end 2025-01-10 --ratio 0.05 --verbose
        """
    )
    
    parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--ratio', type=float, default=0.1, 
                       help='Sampling ratio (default: 0.1 = 10%%)')
    parser.add_argument('--output-dir', default=None, 
                       help='Output directory for samples (default: data/{start}_{end}_{ratio})')
    parser.add_argument('--temp-dir', default='temp', 
                       help='Temporary directory for extraction (default: temp)')
    parser.add_argument('--verbose', action='store_true', 
                       help='Enable verbose logging')
    
    return parser


def run_pipeline(args: argparse.Namespace) -> None:
    """Run the DSA sampling pipeline."""
    logger = logging.getLogger(__name__)
    start = time.now()
    try:
        # Validate arguments
        if not 0 < args.ratio <= 1:
            raise ValueError("Sample ratio must be between 0 and 1")
        
        # Generate date chunks
        date_chunks = generate_date_chunks(args.start, args.end)
        logger.info(f"Processing {len(date_chunks)} date chunks with {args.ratio:.1%} sampling")
        
        # Create output directories
        output_dir = Path(args.output_dir)
        temp_dir = Path(args.temp_dir)
        print(output_dir)
        output_dir.mkdir(exist_ok=True)
        temp_dir.mkdir(exist_ok=True)
        
        logger.info(f"Output directory: {output_dir.absolute()}")
        logger.info(f"Temporary directory: {temp_dir.absolute()}")
        
        # Process each date chunk
        for i, (chunk_start, chunk_end) in enumerate(date_chunks, 1):
            logger.info(f"Processing chunk {i}/{len(date_chunks)}: {chunk_start} to {chunk_end}")
            
            chunk_data_dir = temp_dir / f"data_{chunk_start}_{chunk_end}"
            chunk_extract_dir = temp_dir / f"extract_{chunk_start}_{chunk_end}"
            
            try:
                # Download data for this chunk
                zip_files = download_data(chunk_start, chunk_end, chunk_data_dir)
                
                # Extract all zip files
                csv_files = extract_files(zip_files, chunk_extract_dir)
                
                # Sample the data
                chunk_name = f"{chunk_start}_{chunk_end}"
                sample_file = sample_data(csv_files, args.ratio, output_dir, chunk_name)
                
                logger.info(f"Completed processing chunk {chunk_start}-{chunk_end}: {sample_file}")
                
            except Exception as e:
                logger.error(f"Failed to process chunk {chunk_start}-{chunk_end}: {e}")
                raise
            finally:
                # Cleanup temporary files for this chunk
                staging_dir = temp_dir / f"staging_{chunk_start}_{chunk_end}"
                cleanup_files(chunk_data_dir, chunk_extract_dir, staging_dir)
        
        logger.info("Pipeline completed successfully")
        end = time.now()
        logger.info(f"Total time taken: {end - start:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


def main() -> None:
    """Main entry point for CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Set dynamic default output directory if not explicitly provided
    if args.output_dir is None:  # Check if default sentinel was used
        args.output_dir = f'data/{args.start}_{args.end}_{args.ratio:.2f}'
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Run pipeline
    run_pipeline(args)