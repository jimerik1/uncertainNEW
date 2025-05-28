"""
Utilities for converting between different data formats
"""
import numpy as np
from typing import Dict, List, Any
from models.input_schemas import WellTrajectory, SurveyPoint


def convert_numpy_data_to_api_format(numpy_data_path: str) -> List[WellTrajectory]:
    """
    Convert numpy data format (from original scripts) to API format
    
    Args:
        numpy_data_path: Path to Data.npy file from original pipeline
        
    Returns:
        List of WellTrajectory objects
    """
    # Load the numpy data
    data_w = np.load(numpy_data_path, allow_pickle=True).item()
    
    wells = []
    
    for well_idx, well_data in data_w.items():
        survey_points = []
        
        for point_idx in sorted(well_data.keys()):
            point_data = well_data[point_idx]
            
            survey_point = SurveyPoint(
                depth=float(point_data['depth']),
                inc=float(point_data['inc']),
                az=float(point_data['az']),
                tvd=float(point_data['TVD']),
                north=float(point_data['North']),
                east=float(point_data['East'])
            )
            survey_points.append(survey_point)
        
        well = WellTrajectory(
            well_id=f"WELL-{well_idx:03d}",
            survey_points=survey_points
        )
        wells.append(well)
    
    return wells


def convert_excel_trajectory_to_api_format(trajectory_data: List[Dict[str, float]], 
                                          well_id: str) -> WellTrajectory:
    """
    Convert trajectory data from Excel format to API format
    
    Args:
        trajectory_data: List of dictionaries with keys: MD, Inc, Azi, TVD, EW, NS
        well_id: Identifier for the well
        
    Returns:
        WellTrajectory object
    """
    survey_points = []
    
    for point in trajectory_data:
        survey_point = SurveyPoint(
            depth=float(point['MD']),
            inc=float(point['Inc']),
            az=float(point['Azi']),
            tvd=float(point['TVD']),
            north=float(point['NS']),  # North-South coordinate
            east=float(point['EW'])   # East-West coordinate
        )
        survey_points.append(survey_point)
    
    return WellTrajectory(
        well_id=well_id,
        survey_points=survey_points
    )


def create_test_data_from_original_format() -> Dict[str, Any]:
    """
    Create test API request from sample data in original format
    
    Returns:
        Dictionary suitable for API request
    """
    # Sample trajectory data (similar to what was in the Excel file)
    ref_trajectory = [
        {'MD': 0, 'Inc': 0, 'Azi': 0, 'TVD': 0, 'EW': -7.3229, 'NS': 1.5106},
        {'MD': 197.4, 'Inc': 0, 'Azi': 0, 'TVD': 197.4, 'EW': -7.3229, 'NS': 1.5106},
        {'MD': 210, 'Inc': 0.05, 'Azi': 107.49, 'TVD': 210, 'EW': -7.3176, 'NS': 1.5089},
        {'MD': 220, 'Inc': 0.04, 'Azi': 15.83, 'TVD': 220, 'EW': -7.3125, 'NS': 1.511},
        {'MD': 230, 'Inc': 0.06, 'Azi': 323.73, 'TVD': 230, 'EW': -7.3146, 'NS': 1.5186},
        {'MD': 240, 'Inc': 0.09, 'Azi': 325.05, 'TVD': 240, 'EW': -7.3222, 'NS': 1.5292},
        {'MD': 250, 'Inc': 0.13, 'Azi': 329.67, 'TVD': 250, 'EW': -7.3325, 'NS': 1.5455},
        {'MD': 260, 'Inc': 0.19, 'Azi': 337.87, 'TVD': 259.9999, 'EW': -7.3444, 'NS': 1.5706}
    ]
    
    offset_trajectory = [
        {'MD': 0, 'Inc': 0, 'Azi': 0, 'TVD': 0, 'EW': 50.0, 'NS': 1.5106},
        {'MD': 197.4, 'Inc': 0, 'Azi': 0, 'TVD': 197.4, 'EW': 50.0, 'NS': 1.5106},
        {'MD': 210, 'Inc': 0.03, 'Azi': 120.0, 'TVD': 210, 'EW': 50.1, 'NS': 1.51},
        {'MD': 220, 'Inc': 0.05, 'Azi': 125.0, 'TVD': 220, 'EW': 50.2, 'NS': 1.52},
        {'MD': 230, 'Inc': 0.08, 'Azi': 130.0, 'TVD': 230, 'EW': 50.3, 'NS': 1.53},
        {'MD': 240, 'Inc': 0.12, 'Azi': 135.0, 'TVD': 240, 'EW': 50.4, 'NS': 1.54},
        {'MD': 250, 'Inc': 0.15, 'Azi': 140.0, 'TVD': 250, 'EW': 50.5, 'NS': 1.55},
        {'MD': 260, 'Inc': 0.18, 'Azi': 145.0, 'TVD': 259.9999, 'EW': 50.6, 'NS': 1.56}
    ]
    
    # Convert to API format
    reference_well = convert_excel_trajectory_to_api_format(ref_trajectory, "REF-EQUINOR-001")
    offset_well = convert_excel_trajectory_to_api_format(offset_trajectory, "OFFSET-EQUINOR-001")
    
    return {
        "reference_well": reference_well.dict(),
        "offset_wells": [offset_well.dict()]
    }


def save_api_request_to_file(request_data: Dict[str, Any], filename: str):
    """Save API request data to JSON file for testing"""
    import json
    
    with open(filename, 'w') as f:
        json.dump(request_data, f, indent=2)
    
    print(f"API request data saved to {filename}")


if __name__ == "__main__":
    # Create and save test data
    test_data = create_test_data_from_original_format()
    save_api_request_to_file(test_data, "sample_api_request.json")
    
    print("Sample data created!")
    print(f"Reference well has {len(test_data['reference_well']['survey_points'])} survey points")
    print(f"Offset well has {len(test_data['offset_wells'][0]['survey_points'])} survey points")