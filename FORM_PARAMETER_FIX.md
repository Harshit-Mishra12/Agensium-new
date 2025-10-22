# Form Parameter Handling Fix

## Issue

When optional Form parameters are not provided in a FastAPI request, they are sent as **empty strings (`""`)** instead of `None`. This causes parsing errors for numeric types (int, float).

### Error Example
```json
{
  "detail": [
    {
      "type": "float_parsing",
      "loc": ["body", "outlier_iqr_multiplier"],
      "msg": "Input should be a valid number, unable to parse string as a number",
      "input": ""
    }
  ]
}
```

---

## Solution

### Changed Parameter Types to `Optional[str]`

Instead of:
```python
outlier_iqr_multiplier: Optional[float] = Form(None)
```

Use:
```python
outlier_iqr_multiplier: Optional[str] = Form(None)
```

### Added `parse_param` Helper Function

```python
def parse_param(value, param_type):
    """Convert form values, handling empty strings properly."""
    if value is None or value == "":
        return None
    try:
        return param_type(value)
    except (ValueError, TypeError):
        return None
```

### Parse Parameters Before Use

```python
user_params = {
    'null_alert_threshold': parse_param(null_alert_threshold, float),
    'categorical_threshold': parse_param(categorical_threshold, int),
    'outlier_iqr_multiplier': parse_param(outlier_iqr_multiplier, float),
    # ... etc
}
```

---

## Behavior

| Input | Before | After |
|-------|--------|-------|
| Empty field | ❌ Error: "unable to parse string" | ✅ `None` → uses config.json |
| Valid value | ✅ Works | ✅ Works |
| Invalid value | ❌ Error | ✅ `None` → uses config.json |
| Not provided | ✅ `None` | ✅ `None` |

---

## Applied To

✅ **Unified Profiler** (`/unified-profiler`)
- All 6 optional parameters

✅ **ReadinessRater** (`/rate-readiness`)
- All 5 optional parameters

---

## Testing

### Empty Fields (Should Use Defaults)
```bash
curl -X POST "http://localhost:8000/unified-profiler" \
  -F "file=@data.csv" \
  -F "null_alert_threshold=" \
  -F "categorical_threshold="
```

**Expected:** No error, uses values from `config.json`

### Valid Values
```bash
curl -X POST "http://localhost:8000/unified-profiler" \
  -F "file=@data.csv" \
  -F "null_alert_threshold=15.0" \
  -F "categorical_threshold=30"
```

**Expected:** Uses provided values (15.0, 30)

### Mixed (Some Empty, Some Provided)
```bash
curl -X POST "http://localhost:8000/unified-profiler" \
  -F "file=@data.csv" \
  -F "null_alert_threshold=15.0" \
  -F "categorical_threshold="
```

**Expected:** Uses 15.0 for null_alert_threshold, config.json for categorical_threshold

---

## Benefits

✅ **Robust Error Handling** - No parsing errors on empty fields  
✅ **Graceful Degradation** - Invalid values fall back to defaults  
✅ **User-Friendly** - Works with partial form submissions  
✅ **Backward Compatible** - Existing valid requests still work  

---

**Status:** ✅ Fixed  
**Date:** October 22, 2025  
**Affected Endpoints:** `/unified-profiler`, `/rate-readiness`
