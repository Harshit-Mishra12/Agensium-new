"""
Metric Definitions and Glossary for All Agents
Provides clear UX-friendly definitions, weightings, and tooltips for all metrics.
"""

# ============================================================================
# GLOSSARY - Common Terms Used Across All Agents
# ============================================================================

GLOSSARY = {
    "completeness": {
        "term": "Completeness",
        "definition": "The percentage of non-null values in your dataset. Higher completeness means fewer missing values.",
        "formula": "100 - (null_cells / total_cells × 100)",
        "good_range": "≥ 95%",
        "tooltip": "Measures how much of your data is actually filled in vs. empty"
    },
    "uniqueness": {
        "term": "Uniqueness",
        "definition": "The percentage of distinct values in a column. High uniqueness indicates diverse data.",
        "formula": "unique_values / total_values × 100",
        "good_range": "Varies by field type",
        "tooltip": "Shows how many different values exist - important for IDs and keys"
    },
    "consistency": {
        "term": "Consistency",
        "definition": "Measures data uniformity and absence of duplicates. Higher scores mean cleaner data.",
        "formula": "100 - (duplicate_rows / total_rows × 100)",
        "good_range": "≥ 95%",
        "tooltip": "Checks if your data follows consistent patterns and has no duplicate records"
    },
    "schema_stability": {
        "term": "Schema Stability",
        "definition": "Tracks whether your data structure (columns, types) remains unchanged over time.",
        "formula": "Based on column additions, removals, and type changes",
        "good_range": "No changes",
        "tooltip": "Ensures your database structure isn't unexpectedly changing"
    },
    "type_consistency": {
        "term": "Type Consistency",
        "definition": "Verifies that data types (text, numbers, dates) remain consistent across datasets.",
        "formula": "Percentage of columns with matching data types",
        "good_range": "100%",
        "tooltip": "Makes sure numbers stay numbers and text stays text"
    },
    "null_density": {
        "term": "Null Density",
        "definition": "The concentration of missing values in your data. Lower is better.",
        "formula": "null_values / total_values × 100",
        "good_range": "≤ 5%",
        "tooltip": "Percentage of your data that's missing or empty"
    },
    "distribution_shift": {
        "term": "Distribution Shift",
        "definition": "Measures how much your data patterns have changed compared to a baseline.",
        "formula": "Statistical tests (KS-test, PSI, JS-divergence)",
        "good_range": "PSI < 0.1",
        "tooltip": "Detects if your data is behaving differently than expected"
    },
    "governance_coverage": {
        "term": "Governance Coverage",
        "definition": "Percentage of data that has proper security, privacy, and compliance controls.",
        "formula": "fields_with_governance / total_fields × 100",
        "good_range": "100% for sensitive data",
        "tooltip": "Ensures your data has proper security labels and controls"
    },
    "pii_exposure": {
        "term": "PII Exposure",
        "definition": "Personally Identifiable Information detected in your data (emails, SSN, phone numbers).",
        "formula": "Count of fields containing PII patterns",
        "good_range": "0 (or properly protected)",
        "tooltip": "Identifies sensitive personal information that needs protection"
    },
    "data_quality_score": {
        "term": "Data Quality Score",
        "definition": "Overall health metric combining completeness, accuracy, and consistency.",
        "formula": "Weighted average of quality dimensions",
        "good_range": "≥ 85",
        "tooltip": "Single number summarizing your overall data health"
    }
}

# ============================================================================
# UNIFIED PROFILER - Metric Definitions
# ============================================================================

UNIFIED_PROFILER_METRICS = {
    "agent_name": "UnifiedProfiler",
    "description": "Comprehensive data profiling combining schema scanning and field statistics",
    "metrics": {
        "completeness": {
            "name": "Completeness",
            "definition": "Percentage of non-null values across all fields",
            "calculation": "100 - (total_null_values / total_cells × 100)",
            "weight": "N/A (descriptive metric)",
            "threshold": {
                "good": "≥ 95%",
                "warning": "80-95%",
                "critical": "< 80%"
            },
            "tooltip": GLOSSARY["completeness"]["tooltip"]
        },
        "uniqueness": {
            "name": "Uniqueness",
            "definition": "Ratio of distinct values to total values per field",
            "calculation": "unique_count / total_count × 100",
            "weight": "N/A (descriptive metric)",
            "threshold": {
                "varies": "Context-dependent (IDs should be 100%, categories lower)"
            },
            "tooltip": GLOSSARY["uniqueness"]["tooltip"]
        },
        "data_type": {
            "name": "Data Type Detection",
            "definition": "Automatically inferred data type for each field",
            "calculation": "Pattern analysis and type inference",
            "weight": "N/A (descriptive metric)",
            "tooltip": "Identifies if field contains text, numbers, dates, or booleans"
        },
        "outliers": {
            "name": "Outlier Detection",
            "definition": "Values that fall outside the normal range (IQR method)",
            "calculation": "Values < Q1 - 1.5×IQR or > Q3 + 1.5×IQR",
            "weight": "N/A (descriptive metric)",
            "threshold": {
                "alert": "> 5% outliers"
            },
            "tooltip": "Finds unusual values that might be errors or special cases"
        }
    },
    "output_format": "Detailed field-level statistics with alerts and routing suggestions"
}

# ============================================================================
# DRIFT DETECTOR - Metric Definitions
# ============================================================================

DRIFT_DETECTOR_METRICS = {
    "agent_name": "DriftDetector",
    "description": "Detects changes in data structure and statistical distributions over time",
    "score_formula": "Drift Score = 40% schema stability + 30% type consistency + 30% distribution similarity",
    "components": {
        "schema_stability": {
            "name": "Schema Stability",
            "definition": "Tracks structural changes (new/dropped columns, tables)",
            "calculation": "Comparison of baseline vs current schema",
            "weight": "40%",
            "weight_value": 0.40,
            "threshold": {
                "stable": "No changes",
                "warning": "New columns added",
                "critical": "Columns dropped or renamed"
            },
            "tooltip": GLOSSARY["schema_stability"]["tooltip"]
        },
        "type_consistency": {
            "name": "Type Consistency",
            "definition": "Ensures data types haven't changed between baseline and current",
            "calculation": "Percentage of fields with matching types",
            "weight": "30%",
            "weight_value": 0.30,
            "threshold": {
                "consistent": "100% match",
                "warning": "< 100% match"
            },
            "tooltip": GLOSSARY["type_consistency"]["tooltip"]
        },
        "distribution_similarity": {
            "name": "Distribution Similarity",
            "definition": "Statistical comparison of data patterns using PSI, KS-test, and JS-divergence",
            "calculation": "PSI (Population Stability Index) for categorical, KS-test for numerical",
            "weight": "30%",
            "weight_value": 0.30,
            "threshold": {
                "stable": "PSI < 0.1",
                "warning": "PSI 0.1-0.25",
                "critical": "PSI > 0.25"
            },
            "tooltip": GLOSSARY["distribution_shift"]["tooltip"]
        }
    },
    "statistical_tests": {
        "PSI": {
            "name": "Population Stability Index",
            "definition": "Measures shift in categorical distributions",
            "formula": "Σ(current% - baseline%) × ln(current% / baseline%)",
            "interpretation": "< 0.1 = stable, 0.1-0.25 = moderate drift, > 0.25 = significant drift"
        },
        "KS_test": {
            "name": "Kolmogorov-Smirnov Test",
            "definition": "Compares numerical distributions",
            "formula": "Maximum distance between cumulative distributions",
            "interpretation": "p-value < 0.05 indicates significant drift"
        },
        "JS_divergence": {
            "name": "Jensen-Shannon Divergence",
            "definition": "Symmetric measure of distribution similarity",
            "formula": "Symmetrized KL-divergence",
            "interpretation": "0 = identical, 1 = completely different"
        }
    },
    "output_format": "Drift alerts with severity levels and detailed change reports"
}

# ============================================================================
# READINESS RATER - Metric Definitions
# ============================================================================

READINESS_RATER_METRICS = {
    "agent_name": "ReadinessRater",
    "description": "Evaluates if data is ready for production use or needs cleaning",
    "score_formula": "Readiness Score = 40% completeness + 40% consistency + 20% schema health",
    "components": {
        "completeness": {
            "name": "Completeness Score",
            "definition": "Percentage of non-null values in the dataset",
            "calculation": "100 - (null_cells / total_cells × 100)",
            "weight": "40%",
            "weight_value": 0.40,
            "threshold": {
                "excellent": "≥ 95%",
                "good": "85-95%",
                "needs_work": "< 85%"
            },
            "tooltip": GLOSSARY["completeness"]["tooltip"]
        },
        "consistency": {
            "name": "Consistency Score",
            "definition": "Measures data uniformity and absence of duplicates",
            "calculation": "100 - (duplicate_rows / total_rows × 100)",
            "weight": "40%",
            "weight_value": 0.40,
            "threshold": {
                "excellent": "≥ 95%",
                "good": "85-95%",
                "needs_work": "< 85%"
            },
            "tooltip": GLOSSARY["consistency"]["tooltip"]
        },
        "schema_health": {
            "name": "Schema Health Score",
            "definition": "Evaluates data structure quality (no single-value columns, no mixed types)",
            "calculation": "100 - penalties for structural issues",
            "weight": "20%",
            "weight_value": 0.20,
            "penalties": {
                "single_value_column": "-10 points per column",
                "mixed_type_column": "-5 points per column"
            },
            "threshold": {
                "excellent": "100",
                "good": "≥ 80",
                "needs_work": "< 80"
            },
            "tooltip": "Checks for structural problems like columns with only one value or mixed data types"
        }
    },
    "overall_thresholds": {
        "ready": {
            "score": "≥ 85",
            "status": "Ready",
            "action": "Proceed to master data tool"
        },
        "needs_review": {
            "score": "70-84",
            "status": "Needs Review",
            "action": "Run Clean My Data tool"
        },
        "not_ready": {
            "score": "< 70",
            "status": "Not Ready",
            "action": "Significant data quality work required"
        }
    },
    "output_format": "Overall readiness score with component breakdown and actionable recommendations"
}

# ============================================================================
# RISK SCORER - Metric Definitions
# ============================================================================

RISK_SCORER_METRICS = {
    "agent_name": "RiskScorer",
    "description": "Identifies security, privacy, and compliance risks in your data",
    "score_formula": "Risk Score = 40% PII exposure + 30% sensitive fields + 30% governance gaps",
    "components": {
        "pii_exposure": {
            "name": "PII Exposure Score",
            "definition": "Detects Personally Identifiable Information (email, SSN, phone, credit cards)",
            "calculation": "Percentage of fields containing PII patterns",
            "weight": "40%",
            "weight_value": 0.40,
            "patterns_detected": [
                "Email addresses",
                "Social Security Numbers (SSN)",
                "Phone numbers",
                "Credit card numbers",
                "IP addresses",
                "Passport numbers"
            ],
            "threshold": {
                "high_risk": "≥ 75 (multiple PII types found)",
                "medium_risk": "50-74 (some PII found)",
                "low_risk": "< 50 (minimal PII)"
            },
            "tooltip": GLOSSARY["pii_exposure"]["tooltip"]
        },
        "sensitive_fields": {
            "name": "Sensitive Field Detection",
            "definition": "Identifies fields with sensitive information based on naming patterns",
            "calculation": "Count of fields matching sensitive patterns",
            "weight": "30%",
            "weight_value": 0.30,
            "field_patterns": [
                "SSN, social_security, tax_id",
                "Password, secret, token, api_key",
                "Salary, income, wage",
                "Medical, health, diagnosis",
                "Religion, ethnicity, political affiliation"
            ],
            "threshold": {
                "high_risk": "Multiple sensitive fields without protection",
                "medium_risk": "Some sensitive fields detected",
                "low_risk": "No sensitive fields or properly protected"
            },
            "tooltip": "Finds fields that contain sensitive information based on their names"
        },
        "governance_coverage": {
            "name": "Governance Coverage Score",
            "definition": "Checks for data governance indicators (GDPR, consent, encryption flags)",
            "calculation": "Percentage of sensitive data with governance controls",
            "weight": "30%",
            "weight_value": 0.30,
            "governance_indicators": [
                "GDPR compliance flags",
                "Consent/opt-in fields",
                "Privacy classification",
                "Encryption status",
                "Data retention policies"
            ],
            "threshold": {
                "good": "100% of sensitive data has governance",
                "warning": "Partial governance coverage",
                "critical": "No governance controls found"
            },
            "tooltip": GLOSSARY["governance_coverage"]["tooltip"]
        }
    },
    "risk_levels": {
        "high_risk": {
            "score": "≥ 75",
            "severity": "High",
            "action": "Immediate action required - implement data protection measures"
        },
        "medium_risk": {
            "score": "50-74",
            "severity": "Medium",
            "action": "Review and enhance data security controls"
        },
        "low_risk": {
            "score": "< 50",
            "severity": "Low",
            "action": "Maintain current security practices"
        }
    },
    "output_format": "Risk score with detailed findings, PII locations, and compliance recommendations"
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_metric_definition(agent_name: str) -> dict:
    """
    Returns the complete metric definition for a specific agent.
    
    Args:
        agent_name: Name of the agent (UnifiedProfiler, DriftDetector, ReadinessRater, RiskScorer)
    
    Returns:
        dict: Complete metric definition with formulas, weights, and tooltips
    """
    metrics_map = {
        "UnifiedProfiler": UNIFIED_PROFILER_METRICS,
        "DriftDetector": DRIFT_DETECTOR_METRICS,
        "ReadinessRater": READINESS_RATER_METRICS,
        "RiskScorer": RISK_SCORER_METRICS
    }
    return metrics_map.get(agent_name, {})

def get_glossary_term(term: str) -> dict:
    """
    Returns the definition for a specific glossary term.
    
    Args:
        term: The term to look up (e.g., 'completeness', 'uniqueness')
    
    Returns:
        dict: Term definition with formula, range, and tooltip
    """
    return GLOSSARY.get(term.lower(), {})

def format_metric_display(agent_name: str) -> dict:
    """
    Formats metric information for UX display with clear formulas and weightings.
    
    Args:
        agent_name: Name of the agent
    
    Returns:
        dict: Formatted metric information ready for UI display
    """
    metric_def = get_metric_definition(agent_name)
    
    if not metric_def:
        return {}
    
    display = {
        "agent": metric_def.get("agent_name"),
        "description": metric_def.get("description"),
        "formula": metric_def.get("score_formula", "N/A"),
        "components": []
    }
    
    # Format components with weights
    components = metric_def.get("components", {})
    for key, component in components.items():
        display["components"].append({
            "name": component.get("name"),
            "weight": component.get("weight"),
            "definition": component.get("definition"),
            "tooltip": component.get("tooltip")
        })
    
    return display

def get_all_metrics_summary() -> dict:
    """
    Returns a summary of all agent metrics for documentation or UI display.
    
    Returns:
        dict: Complete summary of all agents and their metrics
    """
    return {
        "UnifiedProfiler": format_metric_display("UnifiedProfiler"),
        "DriftDetector": format_metric_display("DriftDetector"),
        "ReadinessRater": format_metric_display("ReadinessRater"),
        "RiskScorer": format_metric_display("RiskScorer"),
        "glossary": GLOSSARY
    }
