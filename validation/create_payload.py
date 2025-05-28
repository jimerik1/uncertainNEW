#!/usr/bin/env python3
"""
Script to generate API payload from reference_well.txt and offset_well.txt files
for testing the anti-collision API in Postman
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any


def parse_trajectory_file(file_path: str) -> List[Dict[str, float]]:
    """
    Parse trajectory data from tab-separated text file
    
    Args:
        file_path: Path to the trajectory file
        
    Returns:
        List of survey point dictionaries
    """
    survey_points = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Skip header line
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
            
        # Split by tab
        parts = line.split('\t')
        if len(parts) != 6:
            print(f"Warning: Skipping line with {len(parts)} parts: {line}")
            continue
        
        try:
            # Convert European decimal format (comma) to float
            md = float(parts[0].replace(',', '.'))
            inc = float(parts[1].replace(',', '.'))
            azi = float(parts[2].replace(',', '.'))
            tvd = float(parts[3].replace(',', '.'))
            ew = float(parts[4].replace(',', '.'))  # East-West (East coordinate)
            ns = float(parts[5].replace(',', '.'))  # North-South (North coordinate)
            
            survey_point = {
                "depth": md,      # Measured Depth
                "inc": inc,       # Inclination
                "az": azi,        # Azimuth
                "tvd": tvd,       # True Vertical Depth
                "north": ns,      # North coordinate
                "east": ew        # East coordinate
            }
            
            survey_points.append(survey_point)
            
        except ValueError as e:
            print(f"Warning: Could not parse line: {line}")
            print(f"Error: {e}")
            continue
    
    return survey_points


def create_api_payload(reference_file: str, offset_file: str) -> Dict[str, Any]:
    """
    Create API payload from trajectory files
    
    Args:
        reference_file: Path to reference well file
        offset_file: Path to offset well file
        
    Returns:
        Dictionary containing the API payload
    """
    print(f"Reading reference well data from: {reference_file}")
    reference_points = parse_trajectory_file(reference_file)
    print(f"Loaded {len(reference_points)} survey points for reference well")
    
    print(f"Reading offset well data from: {offset_file}")
    offset_points = parse_trajectory_file(offset_file)
    print(f"Loaded {len(offset_points)} survey points for offset well")
    
    # Create the API payload
    payload = {
        "reference_well": {
            "well_id": "REF-VALIDATION-001",
            "survey_points": reference_points
        },
        "offset_wells": [
            {
                "well_id": "OFFSET-VALIDATION-001", 
                "survey_points": offset_points
            }
        ]
    }
    
    return payload


def save_payload_to_file(payload: Dict[str, Any], output_file: str):
    """Save payload to JSON file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"API payload saved to: {output_file}")


def print_payload_summary(payload: Dict[str, Any]):
    """Print summary of the payload"""
    print("\n" + "="*60)
    print("PAYLOAD SUMMARY")
    print("="*60)
    
    ref_well = payload["reference_well"]
    print(f"Reference Well ID: {ref_well['well_id']}")
    print(f"Reference Well Survey Points: {len(ref_well['survey_points'])}")
    
    # Show first and last survey points for reference well
    if ref_well['survey_points']:
        first_point = ref_well['survey_points'][0]
        last_point = ref_well['survey_points'][-1]
        print(f"  First point: MD={first_point['depth']}, Inc={first_point['inc']}, TVD={first_point['tvd']}")
        print(f"  Last point:  MD={last_point['depth']}, Inc={last_point['inc']}, TVD={last_point['tvd']}")
    
    print(f"\nOffset Wells: {len(payload['offset_wells'])}")
    for i, offset_well in enumerate(payload['offset_wells']):
        print(f"  Well {i+1} ID: {offset_well['well_id']}")
        print(f"  Survey Points: {len(offset_well['survey_points'])}")
        
        if offset_well['survey_points']:
            first_point = offset_well['survey_points'][0]
            last_point = offset_well['survey_points'][-1]
            print(f"    First point: MD={first_point['depth']}, Inc={first_point['inc']}, TVD={first_point['tvd']}")
            print(f"    Last point:  MD={last_point['depth']}, Inc={last_point['inc']}, TVD={last_point['tvd']}")
    
    print(f"\nTotal payload size: ~{len(json.dumps(payload)) / 1024:.1f} KB")


def main():
    """Main function"""
    print("Anti-Collision API Payload Generator")
    print("="*60)
    
    # File paths - adjust these if your files are in different locations
    reference_file = "reference_well.txt"
    offset_file = "offset_well.txt"
    output_file = "postman_payload.json"
    
    # Check if files exist
    for file_path in [reference_file, offset_file]:
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            print("Make sure the text files are in the same directory as this script.")
            return
    
    try:
        # Generate payload
        payload = create_api_payload(reference_file, offset_file)
        
        # Save to file
        save_payload_to_file(payload, output_file)
        
        # Print summary
        print_payload_summary(payload)
        
        print("\n" + "="*60)
        print("✅ SUCCESS!")
        print("="*60)
        print(f"Your Postman payload is ready in: {output_file}")
        print("\nTo use in Postman:")
        print("1. Open Postman")
        print("2. Create a new POST request")
        print("3. Set URL to: http://localhost:5000/calculate-minimum-distance")
        print("4. Set Headers: Content-Type: application/json")
        print("5. Copy the contents of postman_payload.json to the Body (raw JSON)")
        print("6. Send the request!")
        
        # Also print a sample for quick copy-paste
        print(f"\nPayload preview (first 500 characters):")
        payload_str = json.dumps(payload, indent=2)
        print(payload_str[:500] + "..." if len(payload_str) > 500 else payload_str)
        
    except Exception as e:
        print(f"❌ Error generating payload: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()