"""
Test suite for minimum distance calculation API
"""
import json
import requests
import numpy as np
from typing import List, Dict, Any

# Test data - simple well trajectories for validation
def create_test_trajectory(well_id: str, start_coords: tuple, end_coords: tuple, num_points: int = 10) -> Dict[str, Any]:
    """Create a simple straight-line test trajectory"""
    start_n, start_e, start_tvd = start_coords
    end_n, end_e, end_tvd = end_coords
    
    survey_points = []
    for i in range(num_points):
        t = i / (num_points - 1)
        
        # Linear interpolation
        north = start_n + t * (end_n - start_n)
        east = start_e + t * (end_e - start_e)
        tvd = start_tvd + t * (end_tvd - start_tvd)
        depth = tvd  # Assume vertical well for simplicity
        
        survey_points.append({
            "depth": depth,
            "inc": 0.0,  # Vertical well
            "az": 0.0,
            "tvd": tvd,
            "north": north,
            "east": east
        })
    
    return {
        "well_id": well_id,
        "survey_points": survey_points
    }


def create_parallel_wells_test() -> Dict[str, Any]:
    """Create test case with two parallel vertical wells"""
    reference_well = create_test_trajectory(
        "REF-001", 
        start_coords=(0, 0, 0),
        end_coords=(0, 0, 1000),
        num_points=11
    )
    
    offset_well = create_test_trajectory(
        "OFFSET-001",
        start_coords=(100, 0, 0),  # 100m east offset
        end_coords=(100, 0, 1000),
        num_points=11
    )
    
    return {
        "reference_well": reference_well,
        "offset_wells": [offset_well]
    }


def create_intersecting_wells_test() -> Dict[str, Any]:
    """Create test case with intersecting wells"""
    reference_well = create_test_trajectory(
        "REF-002",
        start_coords=(0, 0, 0),
        end_coords=(0, 0, 1000),
        num_points=11
    )
    
    # Create an intersecting well
    offset_well = create_test_trajectory(
        "OFFSET-002", 
        start_coords=(-50, 0, 500),  # Starts 50m west at 500m depth
        end_coords=(50, 0, 500),     # Ends 50m east at same depth (horizontal)
        num_points=11
    )
    
    return {
        "reference_well": reference_well,
        "offset_wells": [offset_well]
    }


def test_api_endpoint(base_url: str = "http://localhost:5000"):
    """Test the minimum distance API endpoint"""
    
    print("Testing Anti-Collision Distance API")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n1. Testing health check...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✓ Health check passed")
            print(f"  Response: {response.json()}")
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return
    except requests.RequestException as e:
        print(f"✗ Health check failed - API not reachable: {e}")
        return
    
    # Test 2: Parallel wells
    print("\n2. Testing parallel wells...")
    test_data = create_parallel_wells_test()
    
    try:
        response = requests.post(
            f"{base_url}/calculate-minimum-distance",
            json=test_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Parallel wells calculation successful")
            print(f"  Calculation time: {result['calculation_time']:.3f}s")
            
            # Check first result
            if result['results']:
                first_result = result['results'][0]
                print(f"  Reference well: {first_result['reference_well_id']}")
                print(f"  Offset well: {first_result['offset_well_id']}")
                
                # Check a few distance values
                distances = first_result['distances']
                if '0' in distances:
                    dist_0 = distances['0']['sht_dist']
                    print(f"  Distance at station 0: {dist_0:.2f}m (expected: ~100m)")
                    
                    if abs(dist_0 - 100.0) < 5.0:  # Allow 5m tolerance
                        print("✓ Distance calculation appears correct")
                    else:
                        print("✗ Distance calculation may be incorrect")
        else:
            print(f"✗ Parallel wells test failed: {response.status_code}")
            print(f"  Response: {response.text}")
    
    except requests.RequestException as e:
        print(f"✗ Parallel wells test failed: {e}")
    
    # Test 3: Intersecting wells  
    print("\n3. Testing intersecting wells...")
    test_data = create_intersecting_wells_test()
    
    try:
        response = requests.post(
            f"{base_url}/calculate-minimum-distance",
            json=test_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Intersecting wells calculation successful")
            print(f"  Calculation time: {result['calculation_time']:.3f}s")
            
            # Find minimum distance
            if result['results']:
                distances = result['results'][0]['distances']
                min_dist = min(d['sht_dist'] for d in distances.values())
                print(f"  Minimum distance found: {min_dist:.2f}m")
                
                if min_dist < 10.0:  # Should be very close to 0 for intersecting wells
                    print("✓ Intersection detected correctly")
                else:
                    print("? Intersection may not be detected (this could be expected depending on survey density)")
        else:
            print(f"✗ Intersecting wells test failed: {response.status_code}")
            print(f"  Response: {response.text}")
    
    except requests.RequestException as e:
        print(f"✗ Intersecting wells test failed: {e}")
        
    # Test 4: Batch processing
    print("\n4. Testing batch processing...")
    batch_data = {
        "calculations": [
            create_parallel_wells_test(),
            create_intersecting_wells_test()
        ]
    }
    
    try:
        response = requests.post(
            f"{base_url}/calculate-minimum-distance/batch",
            json=batch_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Batch processing successful")
            print(f"  Total calculation time: {result['total_calculation_time']:.3f}s")
            print(f"  Processed {result['processed_count']} calculations")
        else:
            print(f"✗ Batch processing failed: {response.status_code}")
            print(f"  Response: {response.text}")
    
    except requests.RequestException as e:
        print(f"✗ Batch processing failed: {e}")
    
    print("\n" + "=" * 50)
    print("Testing completed!")


def test_local_service():
    """Test the distance service directly without API"""
    print("Testing Distance Service Locally")
    print("=" * 50)
    
    from services.distance_service import DistanceCalculationService
    from models.input_schemas import WellTrajectory, SurveyPoint
    
    service = DistanceCalculationService()
    
    # Create test wells
    ref_points = [
        SurveyPoint(depth=i*100, inc=0, az=0, tvd=i*100, north=0, east=0)
        for i in range(11)
    ]
    ref_well = WellTrajectory(well_id="REF-001", survey_points=ref_points)
    
    offset_points = [
        SurveyPoint(depth=i*100, inc=0, az=0, tvd=i*100, north=100, east=0)
        for i in range(11)
    ]
    offset_well = WellTrajectory(well_id="OFFSET-001", survey_points=offset_points)
    
    # Test calculation
    try:
        results, calc_time = service.calculate_minimum_distances(ref_well, [offset_well])
        print(f"✓ Local calculation successful in {calc_time:.3f}s")
        
        if results:
            distances = results[0].distances
            first_dist = distances[0].sht_dist
            print(f"  First distance: {first_dist:.2f}m (expected: ~100m)")
            
            if abs(first_dist - 100.0) < 5.0:
                print("✓ Distance calculation correct")
            else:
                print("✗ Distance calculation incorrect")
    
    except Exception as e:
        print(f"✗ Local calculation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Choose test mode:")
    print("1. Test local service only")
    print("2. Test API endpoints (requires running Flask app)")
    print("3. Both")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice in ['1', '3']:
        test_local_service()
    
    if choice in ['2', '3']:
        if choice == '3':
            print("\n" + "=" * 50)
        test_api_endpoint()