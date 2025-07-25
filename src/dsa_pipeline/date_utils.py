"""Date handling utilities for DSA pipeline."""

from datetime import datetime, timedelta
from typing import List, Tuple


def parse_date(date_str: str) -> datetime:
    """Parse date string in YYYY-MM-DD format."""
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        raise ValueError(f"Invalid date format '{date_str}'. Use YYYY-MM-DD.")


def generate_date_chunks(start_date: str, end_date: str) -> List[Tuple[str, str]]:
    """Generate 1-day date chunks between start and end dates."""
    start = parse_date(start_date)
    end = parse_date(end_date)
    
    if start > end:
        raise ValueError("Start date must be before or equal to end date.")
    
    chunks = []
    current = start
    
    while current <= end:
        chunks.append((
            current.strftime('%Y-%m-%d'),
            current.strftime('%Y-%m-%d')
        ))
        current += timedelta(days=1)
    
    return chunks