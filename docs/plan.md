# DSA Data Sampling Pipeline - Simple Implementation Plan

## Overview
A simple 200-250 line script that downloads DSA data in 2-day chunks, extracts nested zip files, samples the data, and cleans up. Focus on core functionality with minimal complexity.

## Architecture
Single Python script (`dsa_sampler.py`) with 4 main functions:
1. `download_data(start_date, end_date)` - Download using shantay
2. `extract_files(zip_path)` - Extract nested zip files  
3. `sample_data(csv_files, ratio)` - Read CSVs and create sample
4. `main()` - Orchestrate the pipeline with cleanup

## Implementation Steps

### Step 1: Basic Script Structure (50 lines)
```
Create dsa_sampler.py with:
- Argument parsing (start_date, end_date, sample_ratio)
- Basic logging setup
- Main function skeleton
- Date handling utilities (2-day chunks)
- Simple error handling
```

### Step 2: Download and Extract Functions (80 lines)
```
Implement core data acquisition:
- download_data(): Execute shantay subprocess calls
- extract_files(): Handle nested zip extraction with zipfile
- Temporary directory management
- Basic progress logging
- File cleanup after processing
```

### Step 3: Data Sampling Function (70 lines)
```
Add data processing capability:
- sample_data(): Read CSV files with pandas
- Random sampling with configurable ratio
- Combine samples from multiple CSV files
- Save to single output parquet file
- Memory-efficient processing (chunking if needed)
```

### Step 4: Integration and Polish (50 lines)
```
Complete the pipeline:
- Wire all functions together in main()
- Add comprehensive error handling
- Implement cleanup of temporary files
- Add summary statistics and logging
- Command-line interface polish
```

## File Structure
```
dsa_sampler.py          # Single main script (~200-250 lines)
samples/                # Output directory for sampled data
temp/                   # Temporary extraction directory
```

## Expected Usage
```bash
uv run python dsa_sampler.py --start 2025-01-01 --end 2025-01-10 --ratio 0.1
```

## Key Design Decisions
- Single script for simplicity
- Process 2 days at a time to manage memory
- Use pandas for CSV reading (already in dependencies)
- Save samples as parquet for efficiency
- Clean up after each 2-day batch
- Basic error handling without complex recovery