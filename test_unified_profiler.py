"""
Quick test script for the Unified Profiler endpoint.
Run this after starting your FastAPI server.
"""

import requests
import json

# Configuration
BASE_URL = "http://localhost:8000"  # Adjust if your server runs on a different port
ENDPOINT = "/unified-profiler"

def test_unified_profiler(file_path: str, custom_thresholds: dict = None):
    """
    Test the unified profiler endpoint with a CSV or Excel file.
    
    Args:
        file_path: Path to the file to profile
        custom_thresholds: Optional dict of threshold parameters to override defaults
                          Example: {
                              'null_alert_threshold': 15.0,
                              'categorical_threshold': 30,
                              'outlier_iqr_multiplier': 2.0
                          }
    """
    url = f"{BASE_URL}{ENDPOINT}"
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (file_path.split('\\')[-1], f)}
            
            # Add optional threshold parameters as form data
            data = custom_thresholds if custom_thresholds else {}
            
            response = requests.post(url, files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Success!")
            print(json.dumps(result, indent=2))
        else:
            print(f"✗ Error {response.status_code}")
            print(response.text)
            
    except FileNotFoundError:
        print(f"✗ File not found: {file_path}")
    except requests.exceptions.ConnectionError:
        print(f"✗ Could not connect to {BASE_URL}. Is the server running?")
    except Exception as e:
        print(f"✗ Unexpected error: {str(e)}")


if __name__ == "__main__":
    # Example usage - replace with your actual file path
    test_file = "sample_data.csv"
    
    print("=" * 70)
    print("Test 1: All Defaults (config.json loaded)")
    print("=" * 70)
    test_unified_profiler(test_file)
    
    print("\n" + "=" * 70)
    print("Test 2: Partial Override (config.json loaded + user overrides)")
    print("=" * 70)
    partial_params = {
        'null_alert_threshold': 15.0,        # Override (default: 20.0)
        'categorical_threshold': 30          # Override (default: 50)
        # Other params loaded from config.json
    }
    test_unified_profiler(test_file, partial_params)
    
    print("\n" + "=" * 70)
    print("Test 3: Full Override (config.json NOT loaded - optimized)")
    print("=" * 70)
    full_params = {
        'null_alert_threshold': 15.0,
        'categorical_threshold': 30,
        'categorical_ratio_threshold': 0.4,
        'top_n_values': 5,
        'outlier_iqr_multiplier': 2.0,
        'outlier_alert_threshold': 0.03
    }
    test_unified_profiler(test_file, full_params)
