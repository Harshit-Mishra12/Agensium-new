# Unified Profiler - Configuration Logic Summary

## Smart Configuration Loading

The endpoint uses **intelligent configuration loading** to optimize performance:

```
┌─────────────────────────────────────────────────────────────┐
│                    User Request                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Check: Are ALL 6 threshold parameters provided?           │
└─────────────────────────────────────────────────────────────┘
                            ↓
                ┌───────────┴───────────┐
                │                       │
               YES                     NO
                │                       │
                ↓                       ↓
    ┌──────────────────────┐  ┌──────────────────────┐
    │  Use user values     │  │  Load config.json    │
    │  directly            │  │  (defaults)          │
    │                      │  │                      │
    │  ✓ Fast (no I/O)    │  │  ✓ Fallback support  │
    │  ✗ config.json NOT  │  │  ✓ Partial override  │
    │    loaded            │  │                      │
    └──────────────────────┘  └──────────────────────┘
                │                       │
                │                       ↓
                │           ┌──────────────────────┐
                │           │  Override with user  │
                │           │  values where present│
                │           └──────────────────────┘
                │                       │
                └───────────┬───────────┘
                            ↓
                ┌──────────────────────┐
                │  Final Config Dict   │
                └──────────────────────┘
                            ↓
                ┌──────────────────────┐
                │  profile_dataset()   │
                └──────────────────────┘
```

## Three Usage Scenarios

### Scenario 1: All Defaults
```python
# User provides: Nothing (only file)
# Behavior: Load all 6 values from config.json
# config.json loaded: ✓ YES

POST /unified-profiler
  file=data.csv
```

**Result:**
```python
config = {
    'null_alert_threshold': 20.0,      # from config.json
    'categorical_threshold': 50,        # from config.json
    'categorical_ratio_threshold': 0.5, # from config.json
    'top_n_values': 10,                 # from config.json
    'outlier_iqr_multiplier': 1.5,      # from config.json
    'outlier_alert_threshold': 0.05     # from config.json
}
```

---

### Scenario 2: Partial Override
```python
# User provides: 2 parameters
# Behavior: Load config.json, override 2 values
# config.json loaded: ✓ YES

POST /unified-profiler
  file=data.csv
  null_alert_threshold=15.0
  categorical_threshold=30
```

**Result:**
```python
config = {
    'null_alert_threshold': 15.0,      # USER OVERRIDE
    'categorical_threshold': 30,        # USER OVERRIDE
    'categorical_ratio_threshold': 0.5, # from config.json
    'top_n_values': 10,                 # from config.json
    'outlier_iqr_multiplier': 1.5,      # from config.json
    'outlier_alert_threshold': 0.05     # from config.json
}
```

---

### Scenario 3: Full Override (Optimized)
```python
# User provides: All 6 parameters
# Behavior: Use user values directly (skip config.json)
# config.json loaded: ✗ NO (performance optimization)

POST /unified-profiler
  file=data.csv
  null_alert_threshold=15.0
  categorical_threshold=30
  categorical_ratio_threshold=0.4
  top_n_values=5
  outlier_iqr_multiplier=2.0
  outlier_alert_threshold=0.03
```

**Result:**
```python
config = {
    'null_alert_threshold': 15.0,      # USER VALUE
    'categorical_threshold': 30,        # USER VALUE
    'categorical_ratio_threshold': 0.4, # USER VALUE
    'top_n_values': 5,                  # USER VALUE
    'outlier_iqr_multiplier': 2.0,      # USER VALUE
    'outlier_alert_threshold': 0.03     # USER VALUE
}
```

---

## Code Implementation

```python
# Collect user parameters
user_params = {
    'null_alert_threshold': null_alert_threshold,
    'categorical_threshold': categorical_threshold,
    'categorical_ratio_threshold': categorical_ratio_threshold,
    'top_n_values': top_n_values,
    'outlier_iqr_multiplier': outlier_iqr_multiplier,
    'outlier_alert_threshold': outlier_alert_threshold
}

# Smart loading logic
if all(value is not None for value in user_params.values()):
    # Scenario 3: All provided - use directly (optimized)
    config = user_params
else:
    # Scenario 1 or 2: Load defaults, then override
    config = load_from_config_json()
    for key, value in user_params.items():
        if value is not None:
            config[key] = value
```

## Benefits

✓ **Performance**: Skips file I/O when all parameters provided  
✓ **Flexibility**: Supports full, partial, or no customization  
✓ **Fallback**: Always has defaults from config.json  
✓ **Clear Logic**: Easy to understand and maintain  
✓ **User-Friendly**: UI can send any combination of parameters
