import re
import numpy as np

def clean_total_sqft(x):
    x = str(x).strip()

    # Handle ranges like "2100-2850"
    if '-' in x:
        try:
            nums = list(map(float, x.split('-')))
            return (nums[0] + nums[1]) / 2
        except:
            return np.nan

    # Remove non-numeric characters except one decimal point
    cleaned = re.sub(r"[^\d.]", "", x)
    if cleaned.count('.') > 1:
        parts = cleaned.split('.')
        cleaned = parts[0] + '.' + ''.join(parts[1:])
    cleaned = cleaned.rstrip('.')

    try:
        return float(cleaned)
    except:
        return np.nan
