"""
Distance calculation service for anti-collision analysis
"""
import numpy as np
import time
from typing import Dict, List, Any, Tuple
from models.input_schemas import WellTrajectory, DistanceResult, WellDistanceResult
from utils.mcm_calculations import mcm
from utils.optimization import golden_ratio_minimizer


class DistanceCalculationService:
    """Service for calculating minimum distances between well trajectories"""
    
    def __init__(self):
        self.r = np.pi / 180  # Degree to radian conversion
    
    def convert_wells_to_internal_format(self, wells: List[WellTrajectory]) -> Dict[int, Dict[int, Dict[str, float]]]:
        """
        Convert input well trajectories to internal calculation format
        
        Args:
            wells: List of WellTrajectory objects
            
        Returns:
            Dictionary in format expected by calculation functions
        """
        data_w = {}
        
        for well_idx, well in enumerate(wells):
            data_w[well_idx] = {}
            
            for point_idx, point in enumerate(well.survey_points):
                data_w[well_idx][point_idx] = {
                    'depth': point.depth,
                    'inc': point.inc,
                    'az': point.az,
                    'TVD': point.tvd,
                    'North': point.north,
                    'East': point.east
                }
        
        return data_w
    
    def find_interval(self, data_w: Dict[int, Dict[int, Dict[str, float]]]) -> Dict[int, Dict[int, Dict[int, List[int]]]]:
        """
        Find intervals between survey stations for minimum distance calculations
        
        Args:
            data_w: Well trajectory data in internal format
            
        Returns:
            Nested dictionary with interval information
        """
        out = {}
        N = len(data_w)
        
        for i in range(N):
            out[i] = {}
            for j in range(N):
                out[i][j] = {}
                if j == i:
                    continue
                
                for k in range(len(data_w[i])):
                    temp = 1e6
                    bool_found = True
                    
                    # Reference point coordinates
                    N_r = data_w[i][k]['North']
                    E_r = data_w[i][k]['East'] 
                    V_r = data_w[i][k]['TVD']
                    
                    # Search through offset well segments
                    for l in range(1, len(data_w[j])):
                        # Previous point
                        N_o_m1 = data_w[j][l-1]['North']
                        E_o_m1 = data_w[j][l-1]['East']
                        V_o_m1 = data_w[j][l-1]['TVD']
                        inc_m1 = self.r * data_w[j][l-1]['inc']
                        azi_m1 = self.r * data_w[j][l-1]['az']
                        
                        # Current point
                        N_o = data_w[j][l]['North']
                        E_o = data_w[j][l]['East']
                        V_o = data_w[j][l]['TVD']
                        inc = self.r * data_w[j][l]['inc']
                        azi = self.r * data_w[j][l]['az']
                        
                        # Direction vectors
                        a_m1 = np.sin(inc_m1) * np.cos(azi_m1)
                        b_m1 = np.sin(inc_m1) * np.sin(azi_m1)
                        c_m1 = np.cos(inc_m1)
                        
                        a = np.sin(inc) * np.cos(azi)
                        b = np.sin(inc) * np.sin(azi)
                        c = np.cos(inc)
                        
                        # Projection onto direction vectors
                        f_L_m1 = a_m1*(N_o_m1 - N_r) + b_m1*(E_o_m1 - E_r) + c_m1*(V_o_m1 - V_r)
                        f_L = a*(N_o - N_r) + b*(E_o - E_r) + c*(V_o - V_r)
                        
                        # Distances to segment endpoints
                        dist1 = np.sqrt((N_r - N_o_m1)**2 + (E_r - E_o_m1)**2 + (V_r - V_o_m1)**2)
                        dist2 = np.sqrt((N_r - N_o)**2 + (E_r - E_o)**2 + (V_r - V_o)**2)
                        dist = min(dist1, dist2)
                        
                        # Check if reference point projects onto this segment
                        if (f_L_m1 * f_L <= 0) and (dist <= temp):
                            bool_found = False
                            temp = dist
                            interval = [l-1, l]
                    
                    # If no suitable interval found, use last segment
                    if bool_found:
                        M = len(data_w[j]) - 1
                        interval = [M-1, M]
                    
                    out[i][j][k] = interval
        
        return out
    
    def calculate_shortest_distance(self, data_w: Dict[int, Dict[int, Dict[str, float]]], 
                                  ref_well_idx: int, intervals: Dict[int, Dict[int, Dict[int, List[int]]]]) -> Dict[int, Dict[int, Dict[str, Any]]]:
        """
        Calculate shortest distances from reference well to all other wells
        
        Args:
            data_w: Well trajectory data
            ref_well_idx: Index of reference well
            intervals: Interval data from find_interval
            
        Returns:
            Dictionary with shortest distance results
        """
        N = len(data_w[ref_well_idx])
        shortest_distance = {}
        
        for offset_well_idx in range(len(data_w)):
            if offset_well_idx == ref_well_idx:
                continue
                
            N_k = len(data_w[offset_well_idx])
            
            # End coordinates of offset well
            N_end = data_w[offset_well_idx][N_k-1]['North']
            E_end = data_w[offset_well_idx][N_k-1]['East']
            V_end = data_w[offset_well_idx][N_k-1]['TVD']
            
            shortest_distance[offset_well_idx] = {}
            
            for ref_station_idx in range(N):
                # Reference point coordinates
                nk = data_w[ref_well_idx][ref_station_idx]['North']
                ek = data_w[ref_well_idx][ref_station_idx]['East']
                vk = data_w[ref_well_idx][ref_station_idx]['TVD']
                x_n = np.array([nk, ek, vk])
                
                # Get interval for this reference station
                A = intervals[ref_well_idx][offset_well_idx][ref_station_idx][0]
                B = intervals[ref_well_idx][offset_well_idx][ref_station_idx][1]
                
                # Coordinates of interval endpoints
                NA = data_w[offset_well_idx][A]['North']
                EA = data_w[offset_well_idx][A]['East']
                VA = data_w[offset_well_idx][A]['TVD']
                
                NB = data_w[offset_well_idx][B]['North']
                EB = data_w[offset_well_idx][B]['East']
                VB = data_w[offset_well_idx][B]['TVD']
                
                # Survey data for MCM calculation
                incA = self.r * data_w[offset_well_idx][A]['inc']
                azA = self.r * data_w[offset_well_idx][A]['az']
                MA = data_w[offset_well_idx][A]['depth']
                
                incB = self.r * data_w[offset_well_idx][B]['inc']
                azB = self.r * data_w[offset_well_idx][B]['az']
                MB = data_w[offset_well_idx][B]['depth']
                
                # Define coordinate calculation along curve segment
                coord_AB = lambda s: mcm(incA, azA, MA, incB, azB, MB, s)[0]
                dN_A = lambda t: coord_AB(t)[0]
                dE_A = lambda t: coord_AB(t)[1]
                dV_A = lambda t: coord_AB(t)[2]
                
                # Distance function to minimize
                d2_ab = lambda t: ((nk - (NA + dN_A(t)))**2 + 
                                  (ek - (EA + dE_A(t)))**2 + 
                                  (vk - (VA + dV_A(t)))**2)
                
                # Find minimum distance along curve segment
                tolerance = 1e-5
                result = golden_ratio_minimizer(d2_ab, a=0, b=1, tol=tolerance)
                
                if not result['success']:
                    print(f"Warning: Optimization failed for wells {ref_well_idx}-{offset_well_idx}, station {ref_station_idx}")
                
                temp = np.sqrt(result['fun'])
                
                # Calculate distances to endpoints and well end
                dist_end = np.sqrt((nk - N_end)**2 + (ek - E_end)**2 + (vk - V_end)**2)
                dist_nA = np.sqrt((nk - NA)**2 + (ek - EA)**2 + (vk - VA)**2)
                dist_nB = np.sqrt((nk - NB)**2 + (ek - EB)**2 + (vk - VB)**2)
                
                # Determine shortest distance and optimal parameter
                if (dist_end < min(dist_nA, dist_nB)) and (dist_end < temp):
                    sht_dist = dist_end
                    t0 = 0
                    N_Cl, E_Cl, V_Cl = N_end, E_end, V_end
                elif dist_nA < temp:
                    if dist_nA < dist_nB:
                        sht_dist = dist_nA
                        t0 = 0
                    else:
                        sht_dist = dist_nB
                        t0 = 1
                    N_Cl = NA + dN_A(t0)
                    E_Cl = EA + dE_A(t0)
                    V_Cl = VA + dV_A(t0)
                elif dist_nB < temp:
                    sht_dist = dist_nB
                    t0 = 1
                    N_Cl = NA + dN_A(t0)
                    E_Cl = EA + dE_A(t0)
                    V_Cl = VA + dV_A(t0)
                else:
                    sht_dist = temp
                    t0 = float(result['x'])
                    N_Cl = NA + dN_A(t0)
                    E_Cl = EA + dE_A(t0)
                    V_Cl = VA + dV_A(t0)
                
                # Calculate closest point and direction vector
                Cl_point = np.array([N_Cl, E_Cl, V_Cl])
                D = (Cl_point - x_n)
                D_n = np.linalg.norm(D)
                if D_n > 0:
                    D = D / D_n
                else:
                    D = np.zeros(3)
                
                # Calculate MD at offset well
                MD_off = data_w[offset_well_idx][A]['depth'] + t0 * (data_w[offset_well_idx][B]['depth'] - data_w[offset_well_idx][A]['depth'])
                
                # Store results
                shortest_distance[offset_well_idx][ref_station_idx] = {
                    'sht_dist': sht_dist,
                    'C': len(data_w[offset_well_idx]) - 1 if (dist_end < min(dist_nA, dist_nB)) else (A if d2_ab(0) < d2_ab(1) else B),
                    'ind1': A,
                    'ind2': B,
                    'KC': dist_end if (dist_end < min(dist_nA, dist_nB)) else np.sqrt(d2_ab(0) if d2_ab(0) < d2_ab(1) else d2_ab(1)),
                    'segment': 'CC',
                    't': t0,
                    'Cl_coord': Cl_point,
                    'D_vec': D,
                    'MD_offset': MD_off
                }
        
        return shortest_distance
    
    def calculate_minimum_distances(self, reference_well: WellTrajectory, 
                                   offset_wells: List[WellTrajectory]) -> Tuple[List[WellDistanceResult], float]:
        """
        Main function to calculate minimum distances between reference well and all offset wells
        
        Args:
            reference_well: Reference well trajectory
            offset_wells: List of offset well trajectories
            
        Returns:
            Tuple of (results list, calculation time in seconds)
        """
        start_time = time.time()
        
        # Convert to internal format
        all_wells = [reference_well] + offset_wells
        data_w = self.convert_wells_to_internal_format(all_wells)
        
        # Find intervals
        intervals = self.find_interval(data_w)
        
        # Calculate distances (reference well is index 0)
        ref_well_idx = 0
        distances = self.calculate_shortest_distance(data_w, ref_well_idx, intervals)
        
        # Convert results to output format
        results = []
        for offset_idx, offset_well in enumerate(offset_wells):
            offset_well_idx = offset_idx + 1  # +1 because reference well is at index 0
            
            distance_results = {}
            for ref_station_idx, dist_data in distances[offset_well_idx].items():
                distance_results[ref_station_idx] = DistanceResult(
                    sht_dist=dist_data['sht_dist'],
                    ind1=dist_data['ind1'],
                    ind2=dist_data['ind2'],
                    t=dist_data['t'],
                    cl_coord=dist_data['Cl_coord'].tolist(),
                    d_vec=dist_data['D_vec'].tolist(),
                    md_offset=dist_data['MD_offset']
                )
            
            results.append(WellDistanceResult(
                reference_well_id=reference_well.well_id,
                offset_well_id=offset_well.well_id,
                distances=distance_results
            ))
        
        calculation_time = time.time() - start_time
        
        return results, calculation_time