"""
Validation script that compares API results with original script results
"""
import os
import sys
import subprocess
import numpy as np
import pandas as pd
import requests
import json
import time
from typing import Dict, List, Any, Tuple
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.input_schemas import WellTrajectory, SurveyPoint
from utils.data_conversion import convert_excel_trajectory_to_api_format


class ValidationComparison:
    def __init__(self, api_base_url: str = "http://localhost:5059"):
        self.api_base_url = api_base_url
        self.project_root = project_root
        self.tores_reference_dir = self.project_root / "tores_reference"
        self.data_dir = self.project_root / "data"
        
        # Ensure data directory exists
        self.data_dir.mkdir(exist_ok=True)
    
    def run_original_script(self) -> Tuple[bool, float]:
        """
        Run the original Eq_min_distance_v3.py script and time its execution
        
        Returns:
            Tuple of (success, execution_time_seconds)
        """
        print("Running original Eq_min_distance_v3.py script...")
        
        original_script = self.tores_reference_dir / "Eq_min_distance_v3.py"
        
        if not original_script.exists():
            print(f"❌ Original script not found at {original_script}")
            return False, 0.0
        
        try:
            # Change to the tores_reference directory to run the script
            original_cwd = os.getcwd()
            os.chdir(self.tores_reference_dir)
            
            # Time the original script execution
            start_time = time.time()
            result = subprocess.run([sys.executable, "Eq_min_distance_v3.py"], 
                                  capture_output=True, text=True, timeout=300)
            execution_time = time.time() - start_time
            
            # Change back to original directory
            os.chdir(original_cwd)
            
            if result.returncode == 0:
                print(f"✅ Original script completed successfully in {execution_time:.3f}s")
                if result.stdout:
                    print(f"Script output: {result.stdout}")
                return True, execution_time
            else:
                print(f"❌ Original script failed with return code {result.returncode}")
                if result.stderr:
                    print(f"Error output: {result.stderr}")
                return False, execution_time
                
        except subprocess.TimeoutExpired:
            print("❌ Original script timed out after 5 minutes")
            return False, 300.0  # Return timeout duration
        except Exception as e:
            print(f"❌ Error running original script: {e}")
            return False, 0.0
    
    def load_original_results(self) -> Dict[str, Any]:
        """
        Load the results from the original script (.npy files)
        
        Returns:
            Dictionary containing the original results
        """
        print("Loading original script results...")
        
        # Look for the Sht_dist.npy file in tores_reference directory
        sht_dist_file = self.tores_reference_dir / "Sht_dist.npy"
        data_file = self.tores_reference_dir / "Data.npy"
        
        if not sht_dist_file.exists():
            print(f"❌ Results file not found at {sht_dist_file}")
            return {}
        
        if not data_file.exists():
            print(f"❌ Data file not found at {data_file}")
            return {}
        
        try:
            # Load the shortest distance results
            sht_dist = np.load(sht_dist_file, allow_pickle=True).item()
            data_w = np.load(data_file, allow_pickle=True).item()
            
            print(f"✅ Loaded original results for {len(data_w)} wells")
            print(f"   Distance calculations between {len(sht_dist)} well pairs")
            
            return {
                'shortest_distances': sht_dist,
                'well_data': data_w
            }
            
        except Exception as e:
            print(f"❌ Error loading original results: {e}")
            return {}
    
    def load_excel_data(self) -> List[WellTrajectory]:
        """
        Load trajectory data from Excel file and convert to API format
        
        Returns:
            List of WellTrajectory objects
        """
        print("Loading Excel data...")
        
        # Look for Excel file in tores_reference directory
        excel_files = list(self.tores_reference_dir.glob("*.xlsx"))
        if not excel_files:
            excel_files = list(self.tores_reference_dir.glob("*.xls"))
        
        if not excel_files:
            print("❌ No Excel files found in tores_reference directory")
            return []
        
        excel_file = excel_files[0]  # Use the first Excel file found
        print(f"Using Excel file: {excel_file.name}")
        
        try:
            # Read Excel file - assuming it has sheets for different wells
            xl_file = pd.ExcelFile(excel_file)
            wells = []
            
            # Process each sheet as a well
            for sheet_idx, sheet_name in enumerate(xl_file.sheet_names):
                if sheet_name.startswith('Sheet') or str(sheet_idx) in sheet_name:
                    # Skip if it looks like a default sheet name, check if it has data
                    df = pd.read_excel(xl_file, sheet_name=sheet_name)
                    if len(df) < 2:  # Need at least 2 points for a trajectory
                        continue
                
                df = pd.read_excel(xl_file, sheet_name=sheet_name)
                
                # Expected columns: MD, Inc, Azi, TVD, NS, EW (or similar)
                required_cols = ['MD', 'Inc', 'Azi', 'TVD']
                if not all(col in df.columns for col in required_cols):
                    print(f"⚠️  Skipping sheet {sheet_name} - missing required columns")
                    continue
                
                # Handle different naming conventions for coordinates
                north_col = None
                east_col = None
                for col in df.columns:
                    if col.upper() in ['NS', 'NORTH', 'N']:
                        north_col = col
                    elif col.upper() in ['EW', 'EAST', 'E']:
                        east_col = col
                
                if north_col is None or east_col is None:
                    print(f"⚠️  Skipping sheet {sheet_name} - missing coordinate columns")
                    continue
                
                # Convert to trajectory format
                trajectory_data = []
                for _, row in df.iterrows():
                    trajectory_data.append({
                        'MD': row['MD'],
                        'Inc': row['Inc'],
                        'Azi': row['Azi'],
                        'TVD': row['TVD'],
                        'NS': row[north_col],
                        'EW': row[east_col]
                    })
                
                well = convert_excel_trajectory_to_api_format(
                    trajectory_data, 
                    f"WELL-{sheet_idx:02d}"
                )
                wells.append(well)
                print(f"✅ Loaded well {well.well_id} with {len(well.survey_points)} survey points")
            
            return wells
            
        except Exception as e:
            print(f"❌ Error loading Excel data: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def call_api(self, reference_well: WellTrajectory, offset_wells: List[WellTrajectory]) -> Tuple[Dict[str, Any], float]:
        """
        Call the API with the trajectory data and measure execution time
        
        Args:
            reference_well: Reference well trajectory
            offset_wells: List of offset well trajectories
            
        Returns:
            Tuple of (API response as dictionary, total_request_time_seconds)
        """
        print(f"Calling API for reference well {reference_well.well_id} vs {len(offset_wells)} offset wells...")
        
        request_data = {
            "reference_well": reference_well.model_dump(),  # Fixed: Use model_dump instead of dict()
            "offset_wells": [well.model_dump() for well in offset_wells]  # Fixed: Use model_dump instead of dict()
        }
        
        try:
            # Time the entire API request (including network overhead)
            start_time = time.time()
            response = requests.post(
                f"{self.api_base_url}/calculate-minimum-distance",
                json=request_data,
                headers={'Content-Type': 'application/json'},
                timeout=300  # 5 minute timeout
            )
            total_request_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                api_calc_time = result.get('calculation_time', 0.0)
                print(f"✅ API call successful in {total_request_time:.3f}s total")
                print(f"   (API calculation time: {api_calc_time:.3f}s, network overhead: {total_request_time - api_calc_time:.3f}s)")
                return result, total_request_time
            else:
                print(f"❌ API call failed with status {response.status_code}")
                print(f"Response: {response.text}")
                return {}, total_request_time
                
        except requests.RequestException as e:
            print(f"❌ API call failed: {e}")
            return {}, 0.0
    
    def compare_results(self, original_results: Dict[str, Any], api_results: Dict[str, Any]) -> bool:
        """
        Compare the results from original script and API
        
        Args:
            original_results: Results from original script
            api_results: Results from API
            
        Returns:
            True if results match within tolerance, False otherwise
        """
        print("\n" + "="*60)
        print("COMPARING RESULTS")
        print("="*60)
        
        if not original_results or not api_results:
            print("❌ Cannot compare - missing results")
            return False
        
        original_distances = original_results.get('shortest_distances', {})
        api_response_results = api_results.get('results', [])
        
        if not original_distances or not api_response_results:
            print("❌ Cannot compare - empty results")
            return False
        
        tolerance = 1e-6  # Tolerance for floating point comparison
        differences_found = False
        
        # Compare each well pair
        for api_result in api_response_results:
            ref_well_id = api_result['reference_well_id']
            offset_well_id = api_result['offset_well_id']
            api_distances = api_result['distances']
            
            # Find corresponding original results
            # Fixed: Extract numeric part from well IDs properly
            def extract_well_index(well_id: str) -> int:
                """Extract numeric index from well ID like 'WELL-00' -> 0"""
                try:
                    parts = well_id.split('-')
                    if len(parts) >= 2:
                        last_part = parts[-1]
                        if last_part.isdigit():
                            return int(last_part)
                    return 0
                except:
                    return 0
            
            ref_well_idx = extract_well_index(ref_well_id)
            offset_well_idx = extract_well_index(offset_well_id)
            
            # Original results might use different indexing
            # Check if we need to adjust indices
            if ref_well_idx not in original_distances:
                # Try alternative indexing schemes
                for possible_ref_idx in range(len(original_distances)):
                    if possible_ref_idx in original_distances:
                        ref_well_idx = possible_ref_idx
                        break
            
            if ref_well_idx not in original_distances:
                print(f"⚠️  No original results for reference well index {ref_well_idx}")
                continue
            
            # Find offset well in original results
            original_ref_data = original_distances[ref_well_idx]
            found_offset_idx = None
            
            # Try the extracted index first
            if offset_well_idx in original_ref_data:
                found_offset_idx = offset_well_idx
            else:
                # Try other indices
                for possible_offset_idx in original_ref_data.keys():
                    if isinstance(possible_offset_idx, int):
                        found_offset_idx = possible_offset_idx
                        break
            
            if found_offset_idx is None:
                print(f"⚠️  No original results for offset well index {offset_well_idx}")
                continue
            
            original_well_distances = original_ref_data[found_offset_idx]
            
            print(f"\nComparing {ref_well_id} vs {offset_well_id}:")
            print(f"  Using original indices: ref={ref_well_idx}, offset={found_offset_idx}")
            
            # Compare each survey station
            station_matches = 0
            station_total = 0
            
            for station_str, api_dist_data in api_distances.items():
                station_idx = int(station_str)
                station_total += 1
                
                if station_idx not in original_well_distances:
                    print(f"  ⚠️  Station {station_idx}: Not in original results")
                    continue
                
                original_dist_data = original_well_distances[station_idx]
                
                # Compare shortest distance (handle both old and new field names)
                orig_dist = original_dist_data['sht_dist']
                
                # API now uses 'shortest_distance' instead of 'sht_dist'
                api_dist = api_dist_data.get('shortest_distance', api_dist_data.get('sht_dist', 0))
                dist_diff = abs(orig_dist - api_dist)
                
                if dist_diff > tolerance:
                    print(f"  ❌ Station {station_idx}: Distance mismatch")
                    print(f"     Original: {orig_dist:.6f}, API: {api_dist:.6f}, Diff: {dist_diff:.6f}")
                    differences_found = True
                else:
                    print(f"  ✅ Station {station_idx}: Distance match ({orig_dist:.6f})")
                    station_matches += 1
                
                # Compare other fields with field name mapping (old -> new)
                field_mapping = {
                    'ind1': 'segment_start_index',
                    'ind2': 'segment_end_index', 
                    't': 'interpolation_factor'
                }
                
                for old_field, new_field in field_mapping.items():
                    if old_field in original_dist_data:
                        orig_val = original_dist_data[old_field]
                        # Try new field name first, then fall back to old field name
                        api_val = api_dist_data.get(new_field, api_dist_data.get(old_field))
                        
                        if api_val is not None and isinstance(orig_val, (int, float)) and isinstance(api_val, (int, float)):
                            if abs(orig_val - api_val) > tolerance:
                                print(f"     ⚠️  {old_field} ({new_field}): {orig_val} vs {api_val}")
                            # else:
                            #     print(f"     ✅  {old_field} ({new_field}): match ({orig_val})")
                
                # Optional: Compare coordinate arrays if available
                if 'Cl_coord' in original_dist_data and 'closest_point_coordinates' in api_dist_data:
                    orig_coords = np.array(original_dist_data['Cl_coord'])
                    api_coords = np.array(api_dist_data['closest_point_coordinates'])
                    coord_diff = np.linalg.norm(orig_coords - api_coords)
                    if coord_diff > tolerance:
                        print(f"     ⚠️  Coordinates differ by {coord_diff:.6f}")
                
                # Optional: Compare direction vectors if available  
                if 'D_vec' in original_dist_data and 'direction_vector' in api_dist_data:
                    orig_dir = np.array(original_dist_data['D_vec'])
                    api_dir = np.array(api_dist_data['direction_vector'])
                    dir_diff = np.linalg.norm(orig_dir - api_dir)
                    if dir_diff > tolerance:
                        print(f"     ⚠️  Direction vectors differ by {dir_diff:.6f}")
            
            print(f"  Summary: {station_matches}/{station_total} stations matched")
        
        if differences_found:
            print(f"\n❌ VALIDATION FAILED - Differences found beyond tolerance ({tolerance})")
            return False
        else:
            print(f"\n✅ VALIDATION PASSED - All results match within tolerance ({tolerance})")
            return True
    
    def wait_for_api(self, max_wait_time: int = 60) -> bool:
        """
        Wait for the API to be ready
        
        Args:
            max_wait_time: Maximum time to wait in seconds
            
        Returns:
            True if API is ready, False if timeout
        """
        print(f"Waiting for API at {self.api_base_url} to be ready...")
        
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            try:
                response = requests.get(f"{self.api_base_url}/health", timeout=5)
                if response.status_code == 200:
                    print("✅ API is ready")
                    return True
            except requests.RequestException:
                pass
            
            time.sleep(2)
        
        print(f"❌ API not ready after {max_wait_time} seconds")
        return False
    
    def compare_performance(self, original_time: float, api_total_time: float, api_calc_time: float):
        """
        Compare performance between original script and API
        
        Args:
            original_time: Execution time of original script (seconds)
            api_total_time: Total API request time including network (seconds)  
            api_calc_time: Pure API calculation time (seconds)
        """
        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON")
        print("="*60)
        
        print(f"Original Script Execution Time:    {original_time:.3f} seconds")
        print(f"API Total Request Time:            {api_total_time:.3f} seconds")
        print(f"API Pure Calculation Time:         {api_calc_time:.3f} seconds")
        print(f"API Network Overhead:              {api_total_time - api_calc_time:.3f} seconds")
        
        print("\nPerformance Analysis:")
        print("-" * 30)
        
        # Compare pure calculation times
        if original_time > 0 and api_calc_time > 0:
            if api_calc_time < original_time:
                speedup = original_time / api_calc_time
                improvement = ((original_time - api_calc_time) / original_time) * 100
                print(f"🚀 API is {speedup:.2f}x FASTER than original script")
                print(f"   Performance improvement: {improvement:.1f}%")
            elif api_calc_time > original_time:
                slowdown = api_calc_time / original_time
                degradation = ((api_calc_time - original_time) / original_time) * 100
                print(f"🐌 API is {slowdown:.2f}x SLOWER than original script")
                print(f"   Performance degradation: {degradation:.1f}%")
            else:
                print("⚖️  API and original script have similar performance")
        
        # Compare total times (including network)
        if original_time > 0 and api_total_time > 0:
            print(f"\nIncluding Network Overhead:")
            if api_total_time < original_time:
                speedup = original_time / api_total_time
                improvement = ((original_time - api_total_time) / original_time) * 100
                print(f"🌐 API total time is {speedup:.2f}x faster ({improvement:.1f}% improvement)")
            elif api_total_time > original_time:
                slowdown = api_total_time / original_time
                degradation = ((api_total_time - original_time) / original_time) * 100
                print(f"🌐 API total time is {slowdown:.2f}x slower ({degradation:.1f}% degradation)")
            else:
                print("🌐 API total time is similar to original script")
        
        # Performance recommendations
        print(f"\nRecommendations:")
        print("-" * 15)
        
        network_overhead_pct = ((api_total_time - api_calc_time) / api_total_time) * 100 if api_total_time > 0 else 0
        
        if network_overhead_pct > 50:
            print("⚠️  High network overhead detected. Consider:")
            print("   - Running API locally or on faster network")
            print("   - Batch processing for multiple calculations")
        
        if api_calc_time > original_time:
            print("⚠️  API calculation is slower than original. Consider:")
            print("   - Code optimization")
            print("   - Algorithm improvements") 
            print("   - Profiling bottlenecks")
        elif api_calc_time < original_time:
            print("✅ API calculation is faster! Benefits:")
            print("   - Better performance")
            print("   - Scalable via web requests")
            print("   - Can handle concurrent requests")
        
        if original_time < 1.0 and api_calc_time < 1.0:
            print("ℹ️  Both implementations are very fast (<1s)")
            print("   Performance differences may not be significant for this dataset")
    
    def run_full_validation(self) -> bool:
        """
        Run the complete validation process with performance benchmarking
        
        Returns:
            True if validation passes, False otherwise
        """
        print("STARTING FULL VALIDATION WITH PERFORMANCE BENCHMARKING")
        print("="*70)
        
        # Step 1: Wait for API to be ready
        if not self.wait_for_api():
            return False
        
        # Step 2: Run original script and time it
        original_success, original_time = self.run_original_script()
        if not original_success:
            return False
        
        # Step 3: Load original results
        original_results = self.load_original_results()
        if not original_results:
            return False
        
        # Step 4: Load Excel data and convert to API format
        wells = self.load_excel_data()
        if len(wells) < 2:
            print("❌ Need at least 2 wells for comparison")
            return False
        
        # Step 5: Call API with the same data and time it
        reference_well = wells[0]
        offset_wells = wells[1:]
        
        api_results, api_total_time = self.call_api(reference_well, offset_wells)
        if not api_results:
            return False
        
        api_calc_time = api_results.get('calculation_time', 0.0)
        
        # Step 6: Compare results for correctness
        validation_success = self.compare_results(original_results, api_results)
        
        # Step 7: Compare performance (always show, regardless of validation result)
        self.compare_performance(original_time, api_total_time, api_calc_time)
        
        return validation_success


def main():
    """Main function to run validation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate API against original script')
    parser.add_argument('--api-url', default='http://localhost:5059', 
                       help='Base URL for the API (default: http://localhost:5059)')
    parser.add_argument('--wait-time', type=int, default=60,
                       help='Maximum time to wait for API (default: 60 seconds)')
    
    args = parser.parse_args()
    
    validator = ValidationComparison(api_base_url=args.api_url)
    
    try:
        success = validator.run_full_validation()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()