"""
Test script for enhanced ReadinessRater with audit trail and Excel export.
"""

import requests
import json
import base64

BASE_URL = "http://localhost:8000"
ENDPOINT = "/rate-readiness"


def test_readiness_rater(file_path: str, custom_params: dict = None):
    """
    Test the ReadinessRater with focus on audit trail and Excel export.
    
    Args:
        file_path: Path to the CSV/Excel file
        custom_params: Optional custom threshold parameters
    """
    url = f"{BASE_URL}{ENDPOINT}"
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (file_path.split('\\')[-1], f)}
            data = custom_params if custom_params else {}
            
            print(f"📤 Sending request to {url}")
            print(f"   File: {file_path}")
            if custom_params:
                print(f"   Custom parameters: {custom_params}")
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
            for field in audit['fields_scanned'][:5]:
                print(f"  • {field}")
            if len(audit['fields_scanned']) > 5:
                print(f"  ... and {len(audit['fields_scanned']) - 5} more")
            print()
            
            # Display Scores
            print("📊 READINESS SCORES")
            print("-" * 70)
            scores = audit['scores']
            print(f"Average Readiness Score:     {scores['average_readiness_score']}")
            print(f"Average Completeness Score:  {scores['average_completeness_score']}")
            print(f"Average Consistency Score:   {scores['average_consistency_score']}")
            print(f"Average Schema Health Score: {scores['average_schema_health_score']}")
            print()
            print(f"Sheets Analyzed:  {scores['total_sheets_analyzed']}")
            print(f"Rows Analyzed:    {scores['total_rows_analyzed']}")
            print(f"Total Alerts:     {scores['total_alerts_generated']}")
            print(f"  • Critical:     {scores['critical_alerts']}")
            print(f"  • Warning:      {scores['warning_alerts']}")
            print(f"  • Info:         {scores['info_alerts']}")
            print()
            
            # Display Findings
            print(f"🔍 FINDINGS ({len(audit['findings'])} total)")
            print("-" * 70)
            for finding in audit['findings'][:3]:
                print(f"[{finding['severity'].upper()}] {finding.get('sheet', 'N/A')}")
                print(f"  Category: {finding['category']}")
                print(f"  Issue: {finding['issue']}")
                if 'readiness_score' in finding:
                    print(f"  Readiness Score: {finding['readiness_score']}")
                print()
            if len(audit['findings']) > 3:
                print(f"... and {len(audit['findings']) - 3} more findings")
            print()
            
            # Display Actions
            print("⚙️  ACTIONS PERFORMED")
            print("-" * 70)
            for action in audit['actions']:
                print(f"  ✓ {action}")
            print()
            
            # Display Overrides
            if audit['overrides']:
                print("🔧 USER OVERRIDES")
                print("-" * 70)
                for key, value in audit['overrides'].items():
                    print(f"  {key}: {value}")
                print()
            else:
                print("🔧 USER OVERRIDES: None (using defaults)")
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
                data = sheet_result.get('data', {})
                scores = data.get('readiness_score', {})
                
                print(f"Sheet: {sheet_name}")
                print(f"  Status:           {routing.get('status')}")
                print(f"  Readiness Score:  {scores.get('overall', 0)}")
                print(f"  Reason:           {routing.get('reason')}")
                print(f"  Suggestion:       {routing.get('suggestion')}")
                print(f"  Next Agent:       {routing.get('suggested_agent_endpoint')}")
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
    print("READINESS RATER - ENHANCED AUDIT TRAIL & EXCEL EXPORT TEST")
    print("=" * 70)
    print()
    
    # Test 1: Default configuration
    print("TEST 1: Default Configuration")
    print("=" * 70)
    test_readiness_rater(test_file)
    
    print("\n\n")
    
    # Test 2: With custom thresholds
    print("TEST 2: Custom Thresholds (Stricter Standards)")
    print("=" * 70)
    custom_params = {
        'ready_threshold': 90,
        'needs_review_threshold': 75,
        'completeness_weight': 0.5,
        'consistency_weight': 0.3,
        'schema_health_weight': 0.2
    }
    test_readiness_rater(test_file, custom_params)
