# Boolean Type Handling Fix

## Issue

NumPy's percentile/quantile functions don't support boolean arithmetic, causing a `TypeError` when calculating statistics for boolean columns.

### Error Message
```
TypeError: numpy boolean subtract, the `-` operator, is not supported, 
use the bitwise_xor, the `^` operator, or the logical_xor function instead.
```

### Root Cause
- `pd.api.types.is_numeric_dtype()` returns `True` for boolean types
- Boolean columns were being processed as numeric
- NumPy's `quantile()` function tried to perform subtraction on booleans
- This fails because boolean subtraction is not supported in NumPy

---

## Solution

### 1. **Early Boolean Detection**

Added explicit boolean check **before** numeric check in `_infer_semantic_type_and_stats()`:

```python
# Boolean type - treat as categorical
if pd.api.types.is_bool_dtype(dtype):
    stats = _compute_categorical_stats(series, col_name, config, alerts)
    return "Categorical", stats

# Numeric type (excluding boolean)
if pd.api.types.is_numeric_dtype(dtype):
    stats = _compute_numeric_stats(series, col_name, config, alerts)
    return "Numeric", stats
```

**Why:** Boolean columns are now treated as categorical (True/False values), which is semantically correct.

### 2. **Safety Check in Numeric Stats**

Added fallback conversion in `_compute_numeric_stats()`:

```python
def _compute_numeric_stats(series: pd.Series, col_name: str, config: dict, alerts: list) -> dict:
    """Compute statistics for numeric columns."""
    # Convert boolean to int if needed (safety check)
    if pd.api.types.is_bool_dtype(series.dtype):
        series = series.astype(int)
    
    stats = {
        "min": float(series.min()),
        "max": float(series.max()),
        # ... rest of stats
    }
```

**Why:** Defense-in-depth approach. If a boolean somehow reaches this function, it's safely converted to int (0/1).

---

## Behavior

### Before Fix
```python
# Boolean column: [True, False, True, True, False]
# Result: TypeError in quantile calculation
```

### After Fix
```python
# Boolean column: [True, False, True, True, False]
# Semantic Type: "Categorical"
# Stats: {
#   "unique_values_count": 2,
#   "top_values": [
#     {"value": "True", "count": 3},
#     {"value": "False", "count": 2}
#   ],
#   "entropy": 0.971
# }
```

---

## Type Handling Priority

The type inference now follows this order:

1. **Boolean** → Categorical
2. **Numeric** (int, float) → Numeric
3. **Object** → Temporal / Categorical / Free Text
4. **Datetime** → Temporal
5. **Default** → Categorical

---

## Examples

### Boolean Column
```python
df = pd.DataFrame({
    'is_active': [True, False, True, True, False]
})
```

**Result:**
- Semantic Type: `"Categorical"`
- Stats: Unique values, top values, entropy
- No quantile calculations attempted

### Integer Column
```python
df = pd.DataFrame({
    'age': [25, 30, 35, 40, 45]
})
```

**Result:**
- Semantic Type: `"Numeric"`
- Stats: min, max, mean, median, std_dev, p25, p75, outliers

### Mixed Boolean/Numeric
```python
df = pd.DataFrame({
    'is_active': [True, False, True],
    'age': [25, 30, 35]
})
```

**Result:**
- `is_active`: Categorical stats
- `age`: Numeric stats

---

## Testing

### Test with Boolean Column

```python
import pandas as pd

# Create test data
df = pd.DataFrame({
    'customer_id': [1, 2, 3, 4, 5],
    'is_premium': [True, False, True, True, False],
    'is_verified': [True, True, False, True, False]
})

# Save and test
df.to_csv('test_boolean.csv', index=False)
```

```bash
curl -X POST "http://localhost:8000/unified-profiler" \
  -F "file=@test_boolean.csv"
```

**Expected:**
- `customer_id`: Numeric type
- `is_premium`: Categorical type (not Numeric)
- `is_verified`: Categorical type (not Numeric)
- No NumPy errors

---

## Benefits

✅ **No More Errors** - Boolean columns handled gracefully  
✅ **Semantically Correct** - Booleans treated as categorical  
✅ **Defense in Depth** - Multiple safety checks  
✅ **Proper Statistics** - Categorical stats (counts, entropy) for booleans  
✅ **Backward Compatible** - Numeric columns still work correctly  

---

## Related Data Types

| Pandas Type | Detected As | Stats Function |
|-------------|-------------|----------------|
| `bool` | Categorical | `_compute_categorical_stats` |
| `int64`, `float64` | Numeric | `_compute_numeric_stats` |
| `object` (dates) | Temporal | `_compute_temporal_stats` |
| `object` (low cardinality) | Categorical | `_compute_categorical_stats` |
| `object` (high cardinality) | Free Text | `_compute_text_stats` |
| `datetime64` | Temporal | `_compute_temporal_stats` |

---

**Status:** ✅ Fixed  
**Date:** October 22, 2025  
**Affected Agent:** Unified Profiler  
**Files Modified:** `app/agents/unified_profiler.py`
