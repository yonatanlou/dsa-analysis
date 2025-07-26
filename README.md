# DSA Analysis
An easy to use tool to analyze the DSA dataset locally.
The [shantay](https://github.com/apparebit/shantay/tree/boss) package is great for downloading the data in batches, but the data is huge.

This repo have two main sections:
- **Download DSA data** - simply by: 
`uv run python scripts/run_pipeline.py --start 2025-01-01 --end 2025-01-07 --ratio 0.04`
ratio of 0.04 will give you a ~100M per day, so make sure you are not bloating your machine with data. 

- **Analysis** - jupyter notebooks in analysis/notebooks.
