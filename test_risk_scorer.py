"""
Test script for RiskScorer agent with PII detection and risk assessment.
"""

import requests
import json
import base64

BASE_URL = "http://localhost:8000"
ENDPOINT = "/score-risk"


def test_risk_scorer(file_path: str, custom_params: dict = None):
    """
    Test the RiskScorer agent.
    
    Args:
        file_path: Path to the CSV/Excel file
        custom_params: Optional custom parameters
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
            
            # Display Risk Scores
            print("🚨 RISK ASSESSMENT")
            print("-" * 70)
            scores = audit['scores']
            print(f"Average Risk Score:      {scores['average_risk_score']}")
            print(f"PII Fields Detected:     {scores['pii_fields_detected']}")
            print(f"Sensitive Fields:        {scores['sensitive_fields_detected']}")
            print(f"Governance Gaps:         {scores['governance_gaps']}")
            print()
            print(f"Risk Breakdown:")
            print(f"  • Critical:            {scores['critical_risks']}")
            print(f"  • High:                {scores['high_risks']}")
            print(f"  • Medium:              {scores['medium_risks']}")
            print(f"  • Low:                 {scores['low_risks']}")
            print()
            
            # Display Fields Scanned
            print(f"Fields Scanned:  {len(audit['fields_scanned'])} columns")
            for field in audit['fields_scanned'][:5]:
                print(f"  • {field}")
            if len(audit['fields_scanned']) > 5:
                print(f"  ... and {len(audit['fields_scanned']) - 5} more")
            print()
            
            # Display Findings
            print(f"🔍 FINDINGS ({len(audit['findings'])} total)")
            print("-" * 70)
            for finding in audit['findings'][:5]:
                print(f"[{finding['severity'].upper()}] {finding.get('field', 'N/A')}")
                print(f"  Category: {finding['category']}")
                print(f"  Issue: {finding['issue']}")
                if 'risk_score' in finding:
                    print(f"  Risk Score: {finding['risk_score']}")
                print()
            if len(audit['findings']) > 5:
                print(f"... and {len(audit['findings']) - 5} more findings")
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
            
            # Handle Excel Export
            print("=" * 70)
            print("\n📊 EXCEL EXPORT")
            print("-" * 70)
            excel_export = result.get('excel_export', {})
            
            if excel_export.get('download_ready'):
                print(f"✓ Excel file generated successfully!")
                print(f"  Filename:     {excel_export['filename']}")
                print(f"  Size:         {excel_export['size_bytes']:,} bytes")
                print(f"  Sheets:       {len([s for s in excel_export['sheets_included'] if s])}")
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
                
                print(f"Sheet: {sheet_name}")
                print(f"  Status:           {routing.get('status')}")
                print(f"  Risk Score:       {data.get('overall_risk_score', 0)}")
                print(f"  PII Fields:       {len(data.get('pii_fields', []))}")
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
    
    test_file = "sample_data.csv"
    
    print("=" * 70)
    print("RISK SCORER - PII DETECTION & RISK ASSESSMENT TEST")
    print("=" * 70)
    print()
    
    # Test 1: Default configuration
    print("TEST 1: Default Configuration")
    print("=" * 70)
    test_risk_scorer(test_file)
    
    print("\n\n")
    
    # Test 2: Stricter thresholds
    print("TEST 2: Stricter Risk Thresholds")
    print("=" * 70)
    custom_params = {
        'high_risk_threshold': 60,
        'pii_sample_size': 200
    }
    test_risk_scorer(test_file, custom_params)
