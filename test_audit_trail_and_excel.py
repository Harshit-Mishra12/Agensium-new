"""
Test script for Unified Profiler with enhanced audit trail and Excel export.
"""

import requests
import json
import base64

BASE_URL = "http://localhost:8000"
ENDPOINT = "/unified-profiler"


def test_audit_trail_and_excel(file_path: str, custom_thresholds: dict = None):
    """
    Test the unified profiler with focus on audit trail and Excel export.
    
    Args:
        file_path: Path to the CSV/Excel file
        custom_thresholds: Optional custom threshold parameters
    """
    url = f"{BASE_URL}{ENDPOINT}"
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (file_path.split('\\')[-1], f)}
            data = custom_thresholds if custom_thresholds else {}
            
            print(f"📤 Sending request to {url}")
            print(f"   File: {file_path}")
            if custom_thresholds:
                print(f"   Custom thresholds: {custom_thresholds}")
            print()
            
            response = requests.post(url, files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            
            print("✓ SUCCESS!")
            print("=" * 70)
            
            # Display Audit Trail
            print("\n📋 AUDIT TRAIL")
            print("-" * 70)
            audit = result['audit']
            
            print(f"Agent Name:      {audit['agent_name']}")
            print(f"Timestamp:       {audit['timestamp']}")
            print(f"Version:         {audit['agent_version']}")
            print(f"Compute Time:    {audit['compute_time_seconds']} seconds")
            print()
            
            print(f"Fields Scanned:  {len(audit['fields_scanned'])} columns")
            for field in audit['fields_scanned'][:5]:  # Show first 5
                print(f"  • {field}")
            if len(audit['fields_scanned']) > 5:
                print(f"  ... and {len(audit['fields_scanned']) - 5} more")
            print()
            
            # Display Findings
            print(f"Findings:        {len(audit['findings'])} issues detected")
            for finding in audit['findings'][:3]:  # Show first 3
                print(f"  [{finding['severity'].upper()}] {finding['field']}")
                print(f"    Category: {finding['category']}")
                print(f"    Issue: {finding['issue']}")
                print()
            if len(audit['findings']) > 3:
                print(f"  ... and {len(audit['findings']) - 3} more findings")
            print()
            
            # Display Actions
            print("Actions Performed:")
            for action in audit['actions']:
                print(f"  ✓ {action}")
            print()
            
            # Display Scores
            print("Scores:")
            scores = audit['scores']
            print(f"  Columns Profiled:    {scores['total_columns_profiled']}")
            print(f"  Sheets Analyzed:     {scores['total_sheets_analyzed']}")
            print(f"  Total Alerts:        {scores['total_alerts_generated']}")
            print(f"    • Critical:        {scores['critical_alerts']}")
            print(f"    • Warning:         {scores['warning_alerts']}")
            print(f"    • Info:            {scores['info_alerts']}")
            print()
            
            # Display Overrides
            if audit['overrides']:
                print("User Overrides:")
                for key, value in audit['overrides'].items():
                    print(f"  {key}: {value}")
                print()
            else:
                print("User Overrides: None (using defaults)")
                print()
            
            # Handle Excel Export
            print("=" * 70)
            print("\n📊 EXCEL EXPORT")
            print("-" * 70)
            excel_export = result.get('excel_export', {})
            
            if excel_export.get('download_ready'):
                print(f"✓ Excel file generated successfully!")
                print(f"  Filename:     {excel_export['filename']}")
                print(f"  Size:         {excel_export['size_bytes']:,} bytes")
                print(f"  Format:       {excel_export['format']}")
                print(f"  Sheets:       {len([s for s in excel_export['sheets_included'] if s])}")
                print()
                print("  Sheets included:")
                for sheet in excel_export['sheets_included']:
                    if sheet:
                        print(f"    • {sheet}")
                print()
                
                # Save Excel file
                excel_bytes = base64.b64decode(excel_export['base64_data'])
                output_filename = excel_export['filename']
                
                with open(output_filename, 'wb') as f:
                    f.write(excel_bytes)
                
                print(f"✓ Excel file saved: {output_filename}")
                print()
            else:
                print(f"✗ Excel generation failed")
                if 'error' in excel_export:
                    print(f"  Error: {excel_export['error']}")
                print()
            
            # Display routing decision
            print("=" * 70)
            print("\n🔀 ROUTING DECISION")
            print("-" * 70)
            for sheet_name, sheet_result in result['results'].items():
                routing = sheet_result.get('routing', {})
                print(f"Sheet: {sheet_name}")
                print(f"  Status:     {routing.get('status')}")
                print(f"  Reason:     {routing.get('reason')}")
                print(f"  Suggestion: {routing.get('suggestion')}")
                print(f"  Next Agent: {routing.get('suggested_agent_endpoint')}")
                print()
            
        else:
            print(f"✗ ERROR {response.status_code}")
            print(response.text)
            
    except FileNotFoundError:
        print(f"✗ File not found: {file_path}")
    except requests.exceptions.ConnectionError:
        print(f"✗ Could not connect to {BASE_URL}. Is the server running?")
    except Exception as e:
        print(f"✗ Unexpected error: {str(e)}")


if __name__ == "__main__":
    import sys
    
    # Example usage
    test_file = "sample_data.csv"
    
    print("=" * 70)
    print("UNIFIED PROFILER - AUDIT TRAIL & EXCEL EXPORT TEST")
    print("=" * 70)
    print()
    
    # Test 1: Default configuration
    print("TEST 1: Default Configuration")
    print("=" * 70)
    test_audit_trail_and_excel(test_file)
    
    print("\n\n")
    
    # Test 2: With custom thresholds
    print("TEST 2: Custom Thresholds (User Overrides)")
    print("=" * 70)
    custom_params = {
        'null_alert_threshold': 15.0,
        'categorical_threshold': 30,
        'outlier_iqr_multiplier': 2.0
    }
    test_audit_trail_and_excel(test_file, custom_params)
