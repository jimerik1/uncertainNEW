"""
Validation script that compares NEV covariance API results with original script results
"""
import os
import sys
import subprocess
import numpy as np
import pandas as pd
import requests
import json
import time
import csv
from typing import Dict, List, Any, Tuple
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.nev_schemas import (
    NEVCovarianceRequest, WellErrorModel, IPMModel, IPMSensor,
    GlobalConstants, ModelConstants, TieOn, SurveyLeg,
    SiteUncertainty, WellUncertainty
)
from models.input_schemas import WellTrajectory, SurveyPoint


class NEVValidationComparison:
    def __init__(self, api_base_url: str = "http://localhost:5059"):
        self.api_base_url = api_base_url
        self.project_root = project_root
        self.tores_reference_dir = self.project_root / "tores_reference"
        self.data_dir = self.project_root / "data"
        
        # Ensure data directory exists
        self.data_dir.mkdir(exist_ok=True)
    
    def run_original_nev_scripts(self) -> Tuple[bool, float]:
        """
        Run the original NEV calculation scripts and time their execution
        
        Returns:
            Tuple of (success, execution_time_seconds)
        """
        print("Running original NEV calculation scripts...")
        
        errvec_script = self.tores_reference_dir / "Eq_ErrVec_calc.py"
        nev_script = self.tores_reference_dir / "Eq_NEV_calc_newest_v2.py"
        
        if not errvec_script.exists() or not nev_script.exists():
            print(f"❌ Original scripts not found")
            return False, 0.0
        
        try:
            # Change to the tores_reference directory to run the scripts
            original_cwd = os.getcwd()
            os.chdir(self.tores_reference_dir)
            
            # Time the original script execution
            start_time = time.time()
            
            # First run error vector calculation
            print("  Running Eq_ErrVec_calc.py...")
            result1 = subprocess.run([sys.executable, "Eq_ErrVec_calc.py"], 
                                   capture_output=True, text=True, timeout=300)
            
            if result1.returncode != 0:
                print(f"❌ Error vector calculation failed: {result1.stderr}")
                return False, 0.0
            
            # Then run NEV calculation
            print("  Running Eq_NEV_calc_newest_v2.py...")
            result2 = subprocess.run([sys.executable, "Eq_NEV_calc_newest_v2.py"], 
                                   capture_output=True, text=True, timeout=300)
            
            execution_time = time.time() - start_time
            
            # Change back to original directory
            os.chdir(original_cwd)
            
            if result2.returncode == 0:
                print(f"✅ Original NEV scripts completed successfully in {execution_time:.3f}s")
                return True, execution_time
            else:
                print(f"❌ NEV calculation failed: {result2.stderr}")
                return False, execution_time
                
        except subprocess.TimeoutExpired:
            print("❌ Original scripts timed out after 5 minutes")
            return False, 300.0
        except Exception as e:
            print(f"❌ Error running original scripts: {e}")
            return False, 0.0
    
    def load_original_nev_results(self) -> Dict[str, Any]:
        """
        Load the NEV covariance results from the original scripts (.npy files)
        
        Returns:
            Dictionary containing the original NEV results
        """
        print("Loading original NEV script results...")
        
        # Look for NEV result files in tores_reference directory
        nev_total_file = self.tores_reference_dir / "NEVtotal.npy"
        nev_gw_file = self.tores_reference_dir / "NEVcov_GW.npy"
        nev_sr_file = self.tores_reference_dir / "NEVcov_SR.npy"
        data_file = self.tores_reference_dir / "Data.npy"
        
        required_files = [nev_total_file, data_file]
        for file_path in required_files:
            if not file_path.exists():
                print(f"❌ Required file not found: {file_path}")
                return {}
        
        try:
            # Load the NEV results
            nev_total = np.load(nev_total_file, allow_pickle=True).item()
            data_w = np.load(data_file, allow_pickle=True).item()
            
            # Load optional files if they exist
            nev_gw = {}
            nev_sr = {}
            if nev_gw_file.exists():
                nev_gw = np.load(nev_gw_file, allow_pickle=True).item()
            if nev_sr_file.exists():
                nev_sr = np.load(nev_sr_file, allow_pickle=True).item()
            
            print(f"✅ Loaded original NEV results for {len(data_w)} wells")
            
            return {
                'nev_total': nev_total,
                'nev_gw': nev_gw,
                'nev_sr': nev_sr,
                'well_data': data_w
            }
            
        except Exception as e:
            print(f"❌ Error loading original NEV results: {e}")
            return {}
    
    def parse_ipm_file(self, ipm_file_path: Path) -> List[IPMSensor]:
        """
        Parse an IPM CSV file and convert to IPMSensor objects
        
        Args:
            ipm_file_path: Path to IPM CSV file
            
        Returns:
            List of IPMSensor objects
        """
        sensors = []
        
        try:
            with open(ipm_file_path, 'r', encoding='utf-8') as f:
                # Skip the header line
                next(f)
                reader = csv.reader(f, delimiter=';')
                
                for row in reader:
                    if len(row) < 16:  # Ensure we have enough columns
                        continue
                    
                    try:
                        sensor = IPMSensor(
                            code=row[1].strip(),
                            term_description=row[2].strip() if row[2] else None,
                            wt_fn=row[3].strip() if row[3] else None,
                            sensor_type=row[4].strip() if row[4] else None,
                            magnitude=float(row[5]) if row[5] else 0.0,
                            units=row[6].strip() if row[6] else None,
                            mode=row[7].strip() if row[7] else 'S',
                            wt_fn_comment=row[8].strip() if row[8] else None,
                            depth_formula=row[9].strip() if row[9] else '0',
                            inclination_formula=row[10].strip() if row[10] else '0',
                            azimuth_formula=row[11].strip() if row[11] else '0',
                            propagation=row[12].strip() if row[12] else None,
                            type_field=row[13].strip() if row[13] else None,
                            initialized=row[14].strip().lower() == 'true' if row[14] else False,
                            h=float(row[15]) if row[15] and row[15] != 'False' else 0.0,
                            inc0=float(row[16]) if len(row) > 16 and row[16] else 0.0,
                            inc1=float(row[17]) if len(row) > 17 and row[17] else 180.0,
                            d_init=float(row[18]) if len(row) > 18 and row[18] else 0.0
                        )
                        sensors.append(sensor)
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Could not parse sensor row: {row[:5]}... Error: {e}")
                        continue
            
            print(f"✅ Parsed {len(sensors)} sensors from {ipm_file_path.name}")
            return sensors
            
        except Exception as e:
            print(f"❌ Error parsing IPM file {ipm_file_path}: {e}")
            return []
    
    def load_well_trajectories(self) -> List[WellTrajectory]:
        """
        Load well trajectory data from Data.npy file
        
        Returns:
            List of WellTrajectory objects
        """
        print("Loading well trajectories...")
        
        data_file = self.tores_reference_dir / "Data.npy"
        if not data_file.exists():
            print(f"❌ Data file not found: {data_file}")
            return []
        
        try:
            data_w = np.load(data_file, allow_pickle=True).item()
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
                    well_id=f"WELL-{well_idx:02d}",
                    survey_points=survey_points
                )
                wells.append(well)
            
            print(f"✅ Loaded {len(wells)} well trajectories")
            return wells
            
        except Exception as e:
            print(f"❌ Error loading well trajectories: {e}")
            return []
    
    def get_available_ipm_files(self) -> Dict[str, str]:
        """
        Map the IPM filenames used in original scripts to actual available files
        
        Returns:
            Dictionary mapping original names to actual filenames
        """
        # Files used in the original script mapped to actual filenames
        file_mapping = {
            "a_depth.csv": "a_depth.csv",
            "b_continuous_gyro_GD.csv": "b_continuous_gyro_GD.csv",
            "b_Magn_IFR_mag_corr_REDQC_MSA_dual_incl.csv": "b_Magn_IFR_mag_corr_RedQC_MSA_dual_incl.csv",  # Note case difference
            "b_Magn_IFR_non_mag_REDQC.csv": "b_Magn_IFR_non_mag_REDQC.csv",
            "b_MWD_gyro_GD_GWD90.csv": "b_MWD_gyro_GD_GWD90.csv",
            "b_Magn_IFR_non_mag_dual_incl.csv": "b_Magn_IFR_non_mag_dual_incl.csv",
            "b_Magn_IFR_non_mag_dual_incl_2.csv": "b_Magn_IFR_non_mag_dual_incl_2.csv",
            "b_Magn_IFR_mag_corr_dual_incl.csv": "b_Magn_IFR_mag_corr_dual_incl.csv",
            "b_Magn_interpolated_azimuth.csv": "b_Magn_interpolated_azimuth.csv",
            "b_Magn_IFR_mag_corr_REDQC.csv": "b_Magn_IFR_mag_corr_REDQC.csv"
        }
        
        # Verify files exist and return only available ones
        available_files = {}
        for original_name, actual_name in file_mapping.items():
            file_path = self.tores_reference_dir / actual_name
            if file_path.exists():
                available_files[original_name] = actual_name
            else:
                print(f"⚠️  File not found: {actual_name}")
        
        return available_files
    
    def load_td_values(self) -> Tuple[float, float]:
        """
        Load TD (Total Depth) values from the well data
        
        Returns:
            Tuple of (TD_0, TD_1) for wells 0 and 1
        """
        try:
            data_file = self.tores_reference_dir / "Data.npy"
            data_w = np.load(data_file, allow_pickle=True).item()
            
            TD_0 = data_w[0][len(data_w[0])-1]['depth'] if 0 in data_w else 3681.0
            TD_1 = data_w[1][len(data_w[1])-1]['depth'] if 1 in data_w else 2719.0
            
            return TD_0, TD_1
        except Exception as e:
            print(f"⚠️  Could not load TD values: {e}, using defaults")
            return 3681.0, 2719.0  # Default values from the original script
    
    def create_api_payload(self) -> NEVCovarianceRequest:
        """
        Create API payload from original data files
        
        Returns:
            NEVCovarianceRequest object
        """
        print("Creating API payload from original data...")
        
        # Load well trajectories
        trajectories = self.load_well_trajectories()
        if not trajectories:
            raise ValueError("Could not load well trajectories")
        
        # Get available IPM files
        available_files = self.get_available_ipm_files()
        
        # Load TD values
        TD_0, TD_1 = self.load_td_values()
        
        # Define global constants (from original scripts)
        global_const_0 = GlobalConstants(
            g_field=9.82,
            b_field=51137.759223,
            dip=72.058675,
            declination=0.447619,
            grid_convergence=-0.440182
        )
        
        global_const_1 = GlobalConstants(
            g_field=9.82, 
            b_field=50752.755273,
            dip=72.113858,
            declination=-0.499753,
            grid_convergence=-0.44
        )
        
        # Define model constants
        model_const = ModelConstants(
            earth_rotation=15.041,
            latitude=59.16524656,
            running_speed=2880.0,  # 3600*0.8
            inc_init=180.0,
            noise_reduction_factor=1.0,
            min_distance=0.0,
            cant_angle=20.0,
            inc_init_constraint=15.0,
            az_init_constraint=45.0 - 0.440182,  # Adjusted for grid convergence
            depth_init=1000.0,
            survey_start=0
        )
        
        model_const_2 = ModelConstants(
            earth_rotation=15.041,
            latitude=59.16524656,
            running_speed=2880.0,
            inc_init=180.0,
            noise_reduction_factor=1.0,
            min_distance=0.0,
            cant_angle=20.0,
            inc_init_constraint=15.0,
            az_init_constraint=45.0 - 0.440182,
            depth_init=1921.0,  # Different for well 1
            survey_start=0
        )
        
        wells = []
        
        # Well 0 configuration (based on original script)
        if len(trajectories) > 0:
            # IPM files for well 0 with correct mapping
            ipm_files_0_config = [
                ("a_depth.csv", (0, 198), model_const),
                ("b_continuous_gyro_GD.csv", (198, 2261), model_const),
                ("b_Magn_IFR_mag_corr_REDQC_MSA_dual_incl.csv", (2261, 2280), model_const),
                ("b_Magn_IFR_mag_corr_REDQC_MSA_dual_incl.csv", (2280, 2613), model_const),
                ("b_Magn_IFR_non_mag_REDQC.csv", (2613, 2733), model_const),
                ("b_MWD_gyro_GD_GWD90.csv", (2733, 3051), model_const),
                ("b_Magn_IFR_non_mag_dual_incl_2.csv", (3051, TD_0), model_const)
            ]
            
            ipm_models_0 = []
            model_constants_0 = []
            tie_ons_0 = []
            survey_legs_0 = []
            
            for i, (ipm_file, (start_depth, end_depth), model_const_item) in enumerate(ipm_files_0_config):
                if ipm_file in available_files:
                    actual_filename = available_files[ipm_file]
                    ipm_path = self.tores_reference_dir / actual_filename
                    
                    sensors = self.parse_ipm_file(ipm_path)
                    if sensors:
                        ipm_model = IPMModel(
                            name=f"model_{i}",
                            sensors=sensors
                        )
                        ipm_models_0.append(ipm_model)
                        model_constants_0.append(model_const_item)
                        tie_ons_0.append(TieOn(start_depth=start_depth, end_depth=end_depth))
                        survey_legs_0.append([SurveyLeg(start_depth=start_depth, end_depth=end_depth)])
                else:
                    print(f"⚠️  Skipping unavailable file: {ipm_file}")
            
            if ipm_models_0:
                well_0 = WellErrorModel(
                    well_id=trajectories[0].well_id,
                    trajectory=trajectories[0],
                    global_constants=global_const_0,
                    ipm_models=ipm_models_0,
                    constants=model_constants_0,
                    tie_ons=tie_ons_0,
                    survey_legs=survey_legs_0,
                    mudline_depth=197.4
                )
                wells.append(well_0)
        
        # Well 1 configuration (based on original script)
        if len(trajectories) > 1:
            ipm_files_1_config = [
                ("a_depth.csv", (0, 198), model_const),
                ("b_continuous_gyro_GD.csv", (198, 1902), model_const),
                ("b_Magn_IFR_mag_corr_dual_incl.csv", (1902, 1921), model_const),
                ("b_Magn_interpolated_azimuth.csv", (1921, 2013), model_const_2),
                ("b_Magn_IFR_mag_corr_REDQC.csv", (2013, 2225), model_const),
                ("b_Magn_IFR_mag_corr_REDQC.csv", (2225, 2476), model_const),
                ("b_Magn_IFR_non_mag_dual_incl.csv", (2476, 2719), model_const),
                ("b_Magn_IFR_non_mag_dual_incl.csv", (2719, TD_1), model_const)
            ]
            
            ipm_models_1 = []
            model_constants_1 = []
            tie_ons_1 = []
            survey_legs_1 = []
            
            for i, (ipm_file, (start_depth, end_depth), model_const_item) in enumerate(ipm_files_1_config):
                if ipm_file in available_files:
                    actual_filename = available_files[ipm_file]
                    ipm_path = self.tores_reference_dir / actual_filename
                    
                    sensors = self.parse_ipm_file(ipm_path)
                    if sensors:
                        ipm_model = IPMModel(
                            name=f"model_{i}",
                            sensors=sensors
                        )
                        ipm_models_1.append(ipm_model)
                        model_constants_1.append(model_const_item)
                        tie_ons_1.append(TieOn(start_depth=start_depth, end_depth=end_depth))
                        survey_legs_1.append([SurveyLeg(start_depth=start_depth, end_depth=end_depth)])
                else:
                    print(f"⚠️  Skipping unavailable file: {ipm_file}")
            
            if ipm_models_1:
                well_1 = WellErrorModel(
                    well_id=trajectories[1].well_id,
                    trajectory=trajectories[1],
                    global_constants=global_const_1,
                    ipm_models=ipm_models_1,
                    constants=model_constants_1,
                    tie_ons=tie_ons_1,
                    survey_legs=survey_legs_1,
                    mudline_depth=197.4
                )
                wells.append(well_1)
        
        # Create site and well uncertainties (from original script)
        site_uncertainty = SiteUncertainty(
            sigma_north=0.0,
            sigma_east=0.0,
            sigma_vertical=0.0
        )
        
        well_uncertainty = WellUncertainty(
            sigma_north=0.0,
            sigma_east=0.0,
            sigma_vertical=0.0
        )
        
        if not wells:
            raise ValueError("No wells could be created from available data")
        
        print(f"✅ Created API payload with {len(wells)} wells")
        return NEVCovarianceRequest(
            wells=wells,
            site_uncertainty=site_uncertainty,
            well_uncertainty=well_uncertainty
        )
    
    def call_nev_api(self, request_data: NEVCovarianceRequest) -> Tuple[Dict[str, Any], float]:
        """
        Call the NEV covariance API and measure execution time
        
        Args:
            request_data: NEV covariance request
            
        Returns:
            Tuple of (API response as dictionary, total_request_time_seconds)
        """
        print(f"Calling NEV API for {len(request_data.wells)} wells...")
        
        try:
            # Time the entire API request
            start_time = time.time()
            response = requests.post(
                f"{self.api_base_url}/calculate-nev-covariance",
                json=request_data.model_dump(),
                headers={'Content-Type': 'application/json'},
                timeout=300  # 5 minute timeout
            )
            total_request_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                api_calc_time = result.get('total_calculation_time', 0.0)
                print(f"✅ NEV API call successful in {total_request_time:.3f}s total")
                print(f"   (API calculation time: {api_calc_time:.3f}s, network overhead: {total_request_time - api_calc_time:.3f}s)")
                return result, total_request_time
            else:
                print(f"❌ NEV API call failed with status {response.status_code}")
                print(f"Response: {response.text}")
                return {}, total_request_time
                
        except requests.RequestException as e:
            print(f"❌ NEV API call failed: {e}")
            return {}, 0.0
    
    def compare_nev_results(self, original_results: Dict[str, Any], api_results: Dict[str, Any]) -> bool:
        """
        Compare the NEV covariance results from original script and API
        
        Args:
            original_results: Results from original scripts
            api_results: Results from API
            
        Returns:
            True if results match within tolerance, False otherwise
        """
        print("\n" + "="*60)
        print("COMPARING NEV COVARIANCE RESULTS")
        print("="*60)
        
        if not original_results or not api_results:
            print("❌ Cannot compare - missing results")
            return False
        
        original_nev_total = original_results.get('nev_total', {})
        api_response_results = api_results.get('results', [])
        
        if not original_nev_total or not api_response_results:
            print("❌ Cannot compare - empty results")
            return False
        
        tolerance = 1e-6  # Tolerance for floating point comparison
        differences_found = False
        
        # Compare each well's NEV matrices
        for api_well_result in api_response_results:
            well_id = api_well_result['well_id']
            api_stations = api_well_result['stations']
            
            # Extract well index from well_id (e.g., "WELL-00" -> 0)
            try:
                well_idx = int(well_id.split('-')[1])
            except (IndexError, ValueError):
                well_idx = 0
            
            if well_idx not in original_nev_total:
                print(f"⚠️  No original results for well {well_idx}")
                continue
            
            original_well_nev = original_nev_total[well_idx]
            
            print(f"\nComparing {well_id} (index {well_idx}):")
            
            station_matches = 0
            station_total = len(api_stations)
            
            for api_station in api_stations:
                station_idx = api_station['station_index']
                
                if station_idx not in original_well_nev:
                    print(f"  ⚠️  Station {station_idx}: Not in original results")
                    continue
                
                # Get original NEV matrix (3x3 numpy array)
                original_nev_matrix = original_well_nev[station_idx]
                
                # Get API NEV matrix
                api_nev = api_station['total_covariance']
                
                # Convert API result to numpy array for comparison
                api_nev_matrix = np.array([
                    [api_nev['nn'], api_nev['ne'], api_nev['nv']],
                    [api_nev['en'], api_nev['ee'], api_nev['ev']],
                    [api_nev['vn'], api_nev['ve'], api_nev['vv']]
                ])
                
                # Compare matrices element by element
                matrix_diff = np.abs(original_nev_matrix - api_nev_matrix)
                max_diff = np.max(matrix_diff)
                
                if max_diff > tolerance:
                    print(f"  ❌ Station {station_idx}: NEV matrix mismatch")
                    print(f"     Max difference: {max_diff:.2e} (tolerance: {tolerance:.2e})")
                    print(f"     Original diagonal: [{original_nev_matrix[0,0]:.2e}, {original_nev_matrix[1,1]:.2e}, {original_nev_matrix[2,2]:.2e}]")
                    print(f"     API diagonal:      [{api_nev['nn']:.2e}, {api_nev['ee']:.2e}, {api_nev['vv']:.2e}]")
                    differences_found = True
                else:
                    station_matches += 1
                    if station_idx % 10 == 0:  # Print every 10th station to avoid spam
                        print(f"  ✅ Station {station_idx}: NEV matrix match")
            
            print(f"  Summary: {station_matches}/{station_total} stations matched")
        
        if differences_found:
            print(f"\n❌ NEV VALIDATION FAILED - Differences found beyond tolerance ({tolerance:.2e})")
            return False
        else:
            print(f"\n✅ NEV VALIDATION PASSED - All results match within tolerance ({tolerance:.2e})")
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
                    health_data = response.json()
                    if '/calculate-nev-covariance' in health_data.get('endpoints', []):
                        print("✅ NEV API is ready")
                        return True
            except requests.RequestException:
                pass
            
            time.sleep(2)
        
        print(f"❌ NEV API not ready after {max_wait_time} seconds")
        return False
    
    def run_full_nev_validation(self) -> bool:
        """
        Run the complete NEV validation process with performance benchmarking
        
        Returns:
            True if validation passes, False otherwise
        """
        print("STARTING FULL NEV VALIDATION WITH PERFORMANCE BENCHMARKING")
        print("="*70)
        
        # Step 1: Wait for API to be ready
        if not self.wait_for_api():
            return False
        
        # Step 2: Run original scripts and time them
        original_success, original_time = self.run_original_nev_scripts()
        if not original_success:
            return False
        
        # Step 3: Load original results
        original_results = self.load_original_nev_results()
        if not original_results:
            return False
        
        # Step 4: Create API payload from original data
        try:
            api_request = self.create_api_payload()
        except Exception as e:
            print(f"❌ Could not create API payload: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Step 5: Call API with the same data and time it
        api_results, api_total_time = self.call_nev_api(api_request)
        if not api_results:
            return False
        
        api_calc_time = api_results.get('total_calculation_time', 0.0)
        
        # Step 6: Compare results for correctness
        validation_success = self.compare_nev_results(original_results, api_results)
        
        # Step 7: Compare performance
        self.compare_performance(original_time, api_total_time, api_calc_time)
        
        return validation_success
    
    def compare_performance(self, original_time: float, api_total_time: float, api_calc_time: float):
        """
        Compare performance between original script and API
        """
        print("\n" + "="*60)
        print("NEV PERFORMANCE COMPARISON")
        print("="*60)
        
        print(f"Original Scripts Execution Time:   {original_time:.3f} seconds")
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
                print(f"🚀 API calculation is {speedup:.2f}x FASTER than original scripts")
                print(f"   Performance improvement: {improvement:.1f}%")
            elif api_calc_time > original_time:
                slowdown = api_calc_time / original_time
                degradation = ((api_calc_time - original_time) / original_time) * 100
                print(f"🐌 API calculation is {slowdown:.2f}x SLOWER than original scripts")
                print(f"   Performance degradation: {degradation:.1f}%")
            else:
                print("⚖️  API and original scripts have similar performance")


def main():
    """Main function to run NEV validation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate NEV covariance API against original scripts')
    parser.add_argument('--api-url', default='http://localhost:5059', 
                       help='Base URL for the API (default: http://localhost:5059)')
    parser.add_argument('--wait-time', type=int, default=60,
                       help='Maximum time to wait for API (default: 60 seconds)')
    
    args = parser.parse_args()
    
    validator = NEVValidationComparison(api_base_url=args.api_url)
    
    try:
        success = validator.run_full_nev_validation()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ NEV validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ NEV validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()