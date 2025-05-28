"""
NEV covariance calculation service
"""
import numpy as np
import time
import copy
from typing import Dict, List, Any, Tuple
from models.nev_schemas import (
    NEVCovarianceRequest, NEVCovarianceResponse, NEVWellResult, 
    NEVStationResult, NEVMatrix, WellErrorModel, IPMModel, IPMSensor
)


class NEVCovarianceService:
    """Service for calculating NEV covariance matrices"""
    
    def __init__(self):
        self.r = np.pi / 180  # Degree to radian conversion
    
    def calculate_nev_covariance(self, request: NEVCovarianceRequest) -> NEVCovarianceResponse:
        """
        Main function to calculate NEV covariance matrices for all wells
        
        Args:
            request: NEV covariance calculation request
            
        Returns:
            NEV covariance response with results for all wells
        """
        start_time = time.time()
        results = []
        
        for well_model in request.wells:
            well_start_time = time.time()
            
            # Calculate error vectors for this well
            error_vectors, error_vectors_end = self._calculate_error_vectors(well_model)
            
            # Calculate NEV covariance matrices
            nev_matrices = self._calculate_nev_matrices(well_model, error_vectors, error_vectors_end)
            
            # Apply site and well uncertainties if provided
            if request.site_uncertainty or request.well_uncertainty:
                nev_matrices = self._apply_additional_uncertainties(
                    nev_matrices, request.site_uncertainty, request.well_uncertainty
                )
            
            # Convert to response format
            stations = []
            for station_idx, (depth, total_cov, sensor_contribs) in enumerate(nev_matrices):
                station_result = NEVStationResult(
                    station_index=station_idx,
                    depth=depth,
                    total_covariance=self._numpy_to_nev_matrix(total_cov),
                    sensor_contributions={
                        sensor: self._numpy_to_nev_matrix(contrib) 
                        for sensor, contrib in sensor_contribs.items()
                    }
                )
                stations.append(station_result)
            
            well_calc_time = time.time() - well_start_time
            well_result = NEVWellResult(
                well_id=well_model.well_id,
                stations=stations,
                calculation_time=well_calc_time
            )
            results.append(well_result)
        
        total_time = time.time() - start_time
        
        return NEVCovarianceResponse(
            results=results,
            total_calculation_time=total_time
        )
    
    def _calculate_error_vectors(self, well_model: WellErrorModel) -> Tuple[Dict, Dict]:
        """
        Calculate error vectors for all sensors at all survey stations
        
        Args:
            well_model: Well error model with trajectory and IPM data
            
        Returns:
            Tuple of (error_vectors, error_vectors_end) dictionaries
        """
        trajectory = well_model.trajectory.survey_points
        n_stations = len(trajectory)
        
        error_vectors = {}
        error_vectors_end = {}
        
        # Process each IPM model
        for model_idx, (ipm_model, model_const, tie_on) in enumerate(
            zip(well_model.ipm_models, well_model.model_constants, well_model.tie_ons)
        ):
            error_vectors[model_idx] = {}
            error_vectors_end[model_idx] = {}
            
            # Find station indices for this model's depth range
            start_idx, end_idx = self._find_depth_indices(
                trajectory, tie_on.start_depth, tie_on.end_depth
            )
            
            # Calculate error vectors for each station in this model's range
            for station_idx in range(start_idx, end_idx + 1):
                error_vectors[model_idx][station_idx] = {}
                error_vectors_end[model_idx][station_idx] = {}
                
                station = trajectory[station_idx]
                
                # Calculate error vector for each sensor
                for sensor in ipm_model.sensors:
                    # Skip if depth is below mudline for certain sensors
                    if (station.depth <= well_model.mudline_depth and 
                        sensor.depth_formula == '0'):
                        error_vectors[model_idx][station_idx][sensor.code] = np.zeros(3)
                        error_vectors_end[model_idx][station_idx][sensor.code] = np.zeros(3)
                        continue
                    
                    # Calculate error vector based on sensor formulas
                    err_vec = self._calculate_sensor_error_vector(
                        sensor, station, well_model.global_constants, model_const
                    )
                    
                    error_vectors[model_idx][station_idx][sensor.code] = err_vec
                    error_vectors_end[model_idx][station_idx][sensor.code] = err_vec  # Simplified for now
        
        return error_vectors, error_vectors_end
    
    def _calculate_sensor_error_vector(self, sensor: IPMSensor, station, global_const, model_const) -> np.ndarray:
        """
        Calculate error vector for a single sensor at a single station
        
        Args:
            sensor: IPM sensor definition
            station: Survey station data
            global_const: Global constants
            model_const: Model constants
            
        Returns:
            3D error vector [North, East, Vertical]
        """
        # Convert angles to radians
        inc = station.inc * self.r
        az = station.az * self.r
        
        # Set up calculation variables
        calc_vars = {
            'inc': inc,
            'az': az,
            'az_T': az,  # True azimuth
            'az_i': az,  # Initial azimuth
            'depth': station.depth,
            'tvd': station.tvd,
            'D': station.depth,
            'TVD': station.tvd,
            'G_Field': global_const.g_field,
            'B_Field': global_const.b_field,
            'Dip': global_const.dip * self.r,
            'erot': model_const.earth_rotation * self.r / 3600,  # Convert to rad/s
            'lat': model_const.latitude * self.r,
            'c_run': model_const.running_speed,
            'iinit': model_const.inc_init_constraint * self.r,
            'ainit': model_const.az_init_constraint * self.r,
            'dinit': model_const.depth_init,
            'deltad': abs(station.depth - model_const.depth_init),
            'np': np  # Make numpy available for formula evaluation
        }
        
        # Calculate error vector components
        try:
            # Evaluate formulas safely
            depth_coeff = self._safe_eval(sensor.depth_formula, calc_vars) if sensor.depth_formula != '0' else 0
            inc_coeff = self._safe_eval(sensor.inclination_formula, calc_vars) if sensor.inclination_formula != '0' else 0
            az_coeff = self._safe_eval(sensor.azimuth_formula, calc_vars) if sensor.azimuth_formula != '0' else 0
            
            # Create error vector [North, East, Vertical]
            error_vector = np.array([
                sensor.magnitude * inc_coeff,    # North component
                sensor.magnitude * az_coeff,     # East component  
                sensor.magnitude * depth_coeff   # Vertical component
            ])
            
            return error_vector
            
        except Exception as e:
            print(f"Warning: Error calculating sensor {sensor.code}: {e}")
            return np.zeros(3)
    
    def _safe_eval(self, formula: str, variables: Dict) -> float:
        """
        Safely evaluate a mathematical formula string
        
        Args:
            formula: Formula string to evaluate
            variables: Dictionary of variables available for evaluation
            
        Returns:
            Evaluated result as float
        """
        if not formula or formula == '0':
            return 0.0
        
        try:
            # Create a safe namespace for evaluation
            safe_dict = {
                '__builtins__': {},
                **variables
            }
            
            # Evaluate the formula
            result = eval(formula, safe_dict)
            return float(result) if not np.isnan(result) else 0.0
            
        except Exception as e:
            print(f"Warning: Could not evaluate formula '{formula}': {e}")
            return 0.0
    
    def _calculate_nev_matrices(self, well_model: WellErrorModel, error_vectors: Dict, error_vectors_end: Dict) -> List[Tuple]:
        """
        Calculate NEV covariance matrices from error vectors
        
        Args:
            well_model: Well error model
            error_vectors: Error vectors for each model/station/sensor
            error_vectors_end: End error vectors
            
        Returns:
            List of tuples (depth, total_covariance_matrix, sensor_contributions)
        """
        trajectory = well_model.trajectory.survey_points
        n_stations = len(trajectory)
        results = []
        
        # Calculate covariance matrix for each survey station
        for station_idx in range(n_stations):
            station = trajectory[station_idx]
            
            # Determine which model applies at this depth
            model_idx = self._find_active_model(station.depth, well_model.tie_ons)
            
            total_cov = np.zeros((3, 3))
            sensor_contributions = {}
            
            if model_idx in error_vectors and station_idx in error_vectors[model_idx]:
                # Sum contributions from all sensors
                for sensor_code, err_vec in error_vectors[model_idx][station_idx].items():
                    # Calculate covariance contribution for this sensor
                    sensor_cov = np.outer(err_vec, err_vec)
                    sensor_contributions[sensor_code] = sensor_cov
                    total_cov += sensor_cov
            
            # Add tie-on contributions for previous models
            tie_on_contrib = self._calculate_tie_on_contributions(
                station_idx, model_idx, well_model, error_vectors, error_vectors_end
            )
            total_cov += tie_on_contrib
            
            results.append((station.depth, total_cov, sensor_contributions))
        
        return results
    
    def _find_depth_indices(self, trajectory: List, start_depth: float, end_depth: float) -> Tuple[int, int]:
        """
        Find survey station indices corresponding to depth range
        
        Args:
            trajectory: List of survey points
            start_depth: Start depth
            end_depth: End depth
            
        Returns:
            Tuple of (start_index, end_index)
        """
        start_idx = 0
        end_idx = len(trajectory) - 1
        
        for i, station in enumerate(trajectory):
            if station.depth >= start_depth and start_idx == 0:
                start_idx = i
            if station.depth <= end_depth:
                end_idx = i
        
        return start_idx, end_idx
    
    def _find_active_model(self, depth: float, tie_ons: List) -> int:
        """
        Find which model is active at a given depth
        
        Args:
            depth: Depth to check
            tie_ons: List of tie-on definitions
            
        Returns:
            Model index
        """
        for i, tie_on in enumerate(tie_ons):
            if tie_on.start_depth <= depth <= tie_on.end_depth:
                return i
        return len(tie_ons) - 1  # Default to last model
    
    def _calculate_tie_on_contributions(self, station_idx: int, current_model: int, 
                                     well_model: WellErrorModel, error_vectors: Dict, 
                                     error_vectors_end: Dict) -> np.ndarray:
        """
        Calculate tie-on contributions from previous models
        
        Args:
            station_idx: Current station index
            current_model: Current model index
            well_model: Well error model
            error_vectors: Error vectors
            error_vectors_end: End error vectors
            
        Returns:
            Tie-on contribution matrix
        """
        tie_on_contrib = np.zeros((3, 3))
        
        # Add contributions from all previous models
        for prev_model in range(current_model):
            if prev_model in error_vectors:
                # Find the end station of the previous model
                tie_on = well_model.tie_ons[prev_model]
                trajectory = well_model.trajectory.survey_points
                
                end_station_idx = len(trajectory) - 1
                for i, station in enumerate(trajectory):
                    if station.depth <= tie_on.end_depth:
                        end_station_idx = i
                
                if end_station_idx in error_vectors[prev_model]:
                    # Sum error vectors from previous model
                    for sensor_code, err_vec in error_vectors[prev_model][end_station_idx].items():
                        tie_on_contrib += np.outer(err_vec, err_vec)
        
        return tie_on_contrib
    
    def _apply_additional_uncertainties(self, nev_matrices: List, site_uncertainty, well_uncertainty) -> List:
        """
        Apply site and well level uncertainties to NEV matrices
        
        Args:
            nev_matrices: List of NEV matrix tuples
            site_uncertainty: Site uncertainty definition
            well_uncertainty: Well uncertainty definition
            
        Returns:
            Updated NEV matrices with additional uncertainties
        """
        # Create site covariance matrix
        site_cov = np.zeros((3, 3))
        if site_uncertainty:
            site_cov = np.diag([
                site_uncertainty.sigma_north**2,
                site_uncertainty.sigma_east**2, 
                site_uncertainty.sigma_vertical**2
            ])
        
        # Create well covariance matrix
        well_cov = np.zeros((3, 3))
        if well_uncertainty:
            well_cov = np.diag([
                well_uncertainty.sigma_north**2,
                well_uncertainty.sigma_east**2,
                well_uncertainty.sigma_vertical**2
            ])
        
        # Add to each station's total covariance
        updated_matrices = []
        for depth, total_cov, sensor_contribs in nev_matrices:
            updated_total = total_cov + site_cov + well_cov
            updated_matrices.append((depth, updated_total, sensor_contribs))
        
        return updated_matrices
    
    def _numpy_to_nev_matrix(self, matrix: np.ndarray) -> NEVMatrix:
        """
        Convert numpy 3x3 matrix to NEVMatrix object
        
        Args:
            matrix: 3x3 numpy array
            
        Returns:
            NEVMatrix object
        """
        return NEVMatrix(
            nn=float(matrix[0, 0]),
            ne=float(matrix[0, 1]), 
            nv=float(matrix[0, 2]),
            en=float(matrix[1, 0]),
            ee=float(matrix[1, 1]),
            ev=float(matrix[1, 2]),
            vn=float(matrix[2, 0]),
            ve=float(matrix[2, 1]),
            vv=float(matrix[2, 2])
        )