# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains ad hoc analysis of the DSA (Digital Services Act) dataset. The project analyzes content moderation decisions from various platforms across the EU, including decision types, content categories, platform information, and enforcement actions.

## Development Environment

### Python Version Management
- Uses `uv` for Python version management and dependency handling
- Python 3.12+ required (specified in pyproject.toml)
- Always use `uv run` or `uv shell` to ensure correct Python version

### Dependencies
Core data science stack managed via `uv`:
- pandas, numpy for data manipulation
- matplotlib, seaborn for visualization  
- scikit-learn for ML analysis
- jupyter/ipykernel for notebook analysis
- pytest for testing
- ruff for linting
- pre-commit for code quality

## Common Commands

### Environment Setup
```bash
# Install dependencies
uv sync

# Run Python scripts
uv run python main.py

# Start Jupyter for notebook analysis
uv run jupyter lab

# Install new packages
uv add package-name
uv add --dev package-name  # for dev dependencies
```

### Data Management
```bash
# Download DSA data using shantay
uvx shantay download --first 2025-01-01 --last 2025-01-10 --archive data

# Extract compressed data archives (done in data/2025/01/)
unzip sor-global-YYYY-MM-DD-full.zip
unzip sor-global-YYYY-MM-DD-full-XXXXX.csv.zip
```

### Code Quality
```bash
# Lint code
uv run ruff check .
uv run ruff format .

# Run tests
uv run pytest

# Pre-commit hooks (if configured)
uv run pre-commit run --all-files
```

## Data Architecture

### Data Structure
- **Raw Data**: `data/YYYY/MM/` - Downloaded ZIP archives from shantay
- **Staging**: `dsa-db-staging/` - Additional data staging area
- **Analysis**: `analysis/` - Jupyter notebooks for exploratory analysis
- **Processed**: `data/processed/` - Cleaned datasets (gitignored)

### DSA Dataset Schema
The DSA dataset contains 37 columns including:
- `uuid`: Unique identifier for each content moderation decision
- `platform_name`: Platform that made the decision (Pinterest, BlaBlaCar, Google Shopping, etc.)
- `decision_visibility`: Action taken (content removed, disabled, limited distribution)
- `category`: Content category (pornography, scope of platform service, etc.)
- `decision_ground`: Legal/policy basis for decision
- `territorial_scope`: EU countries where decision applies
- `automated_decision`: Level of automation (fully, partially, not automated)
- `created_at`, `content_date`, `application_date`: Temporal information

### Data Processing Workflow
1. **Download**: Use shantay to get compressed archives
2. **Extract**: Unzip main archive, then individual CSV.zip files (460+ files total)
3. **Load**: Use pandas to read CSV files with appropriate dtype handling
4. **Combine**: Concatenate multiple CSV files into single DataFrame
5. **Analyze**: Exploratory analysis in Jupyter notebooks
6. **Export**: Save processed results to parquet/CSV for sharing

## Working with Large Datasets

### Memory Management
- DSA dataset is large (1GB+ of CSV files)
- Use `low_memory=False` or specify dtypes when reading CSVs
- Consider chunked processing for very large analyses
- Save intermediate results in parquet format for faster loading

### File Organization
- Multiple compressed archives contain 10 CSV files each
- Pattern: `sor-global-YYYY-MM-DD-full-XXXXX-YYYYY.csv`
- Use glob patterns to find and process multiple files systematically

## Jupyter Notebook Conventions

### Analysis Notebooks
- Located in `analysis/` directory
- Use descriptive names: `0.1-first_analysis.ipynb`
- Standard imports: pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
- Include data loading, exploration, and visualization sections
- Save processed datasets for reuse

### Cell Organization
1. **Imports and Setup**: Standard data science libraries
2. **Data Loading**: File discovery and DataFrame creation
3. **Data Overview**: Shape, dtypes, missing values, basic stats
4. **Exploratory Analysis**: Platform distribution, decision types, categories
5. **Visualizations**: Charts and plots for key insights
6. **Export**: Save processed data and figures

## Testing and Validation

When working with DSA data:
- Verify data integrity after extraction (check file counts, shapes)
- Validate date ranges and platform coverage
- Check for duplicate UUIDs across files
- Monitor memory usage during large dataset operations
- Test analysis code on sample datasets before full processing