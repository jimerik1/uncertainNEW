"""
Flask API for Anti-Collision Calculations and NEV Covariance
"""
from flask import Flask, request, jsonify
from pydantic import ValidationError
import traceback
import logging

from models.input_schemas import MinimumDistanceRequest, MinimumDistanceResponse
from models.nev_schemas import NEVCovarianceRequest, NEVCovarianceResponse
from services.distance_service import DistanceCalculationService
from services.nev_service import NEVCovarianceService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Initialize services
distance_service = DistanceCalculationService()
nev_service = NEVCovarianceService()


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'anti-collision-api',
        'version': '1.0.0',
        'endpoints': [
            '/calculate-minimum-distance',
            '/calculate-minimum-distance/batch',
            '/calculate-nev-covariance',
            '/calculate-nev-covariance/batch'
        ]
    })


@app.route('/calculate-minimum-distance', methods=['POST'])
def calculate_minimum_distance():
    """
    Calculate minimum distances between reference well and offset wells
    
    Request body should contain:
    {
        "reference_well": {
            "well_id": "REF-001",
            "survey_points": [
                {
                    "depth": 0.0,
                    "inc": 0.0,
                    "az": 0.0,
                    "tvd": 0.0,
                    "north": 0.0,
                    "east": 0.0
                },
                ...
            ]
        },
        "offset_wells": [
            {
                "well_id": "OFFSET-001", 
                "survey_points": [...]
            },
            ...
        ]
    }
    
    Response will contain:
    {
        "results": [
            {
                "reference_well_id": "REF-001",
                "offset_well_id": "OFFSET-001",
                "distances": {
                    "0": {
                        "shortest_distance": 25.3,
                        "closest_point_coordinates": [1234.5, 6789.0, 2500.0],
                        "direction_vector": [0.8, 0.6, 0.0],
                        "segment_start_index": 15,
                        "segment_end_index": 16,
                        "interpolation_factor": 0.35,
                        "offset_well_measured_depth": 2485.7
                    },
                    ...
                }
            }
        ],
        "calculation_time": 1.25
    }
    """
    try:
        # Validate input
        try:
            request_data = MinimumDistanceRequest(**request.json)
        except ValidationError as e:
            logger.error(f"Input validation failed: {e}")
            return jsonify({
                'error': 'Invalid input data',
                'details': e.errors()
            }), 400
        
        logger.info(f"Calculating minimum distances for reference well {request_data.reference_well.well_id} "
                   f"against {len(request_data.offset_wells)} offset wells")
        
        # Perform calculation
        results, calc_time = distance_service.calculate_minimum_distances(
            request_data.reference_well,
            request_data.offset_wells
        )
        
        # Create response
        response = MinimumDistanceResponse(
            results=results,
            calculation_time=calc_time
        )
        
        logger.info(f"Calculation completed in {calc_time:.3f} seconds")
        
        return jsonify(response.model_dump())
    
    except Exception as e:
        logger.error(f"Calculation failed: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.route('/calculate-minimum-distance/batch', methods=['POST'])
def calculate_minimum_distance_batch():
    """
    Calculate minimum distances for multiple reference wells (batch processing)
    
    Request body should contain:
    {
        "calculations": [
            {
                "reference_well": {...},
                "offset_wells": [...]
            },
            ...
        ]
    }
    """
    try:
        data = request.json
        if not data or 'calculations' not in data:
            return jsonify({
                'error': 'Invalid input data',
                'message': 'Expected "calculations" array in request body'
            }), 400
        
        batch_results = []
        total_time = 0
        
        for idx, calc_data in enumerate(data['calculations']):
            try:
                # Validate each calculation request
                calc_request = MinimumDistanceRequest(**calc_data)
                
                logger.info(f"Processing batch item {idx+1}/{len(data['calculations'])}: "
                           f"reference well {calc_request.reference_well.well_id}")
                
                # Perform calculation
                results, calc_time = distance_service.calculate_minimum_distances(
                    calc_request.reference_well,
                    calc_request.offset_wells
                )
                
                batch_results.append({
                    'reference_well_id': calc_request.reference_well.well_id,
                    'results': [r.model_dump() for r in results],
                    'calculation_time': calc_time
                })
                
                total_time += calc_time
                
            except ValidationError as e:
                logger.error(f"Validation failed for batch item {idx+1}: {e}")
                batch_results.append({
                    'error': f'Invalid input data for batch item {idx+1}',
                    'details': e.errors()
                })
            except Exception as e:
                logger.error(f"Calculation failed for batch item {idx+1}: {e}")
                batch_results.append({
                    'error': f'Calculation failed for batch item {idx+1}',
                    'message': str(e)
                })
        
        logger.info(f"Batch processing completed in {total_time:.3f} seconds total")
        
        return jsonify({
            'batch_results': batch_results,
            'total_calculation_time': total_time,
            'processed_count': len(data['calculations'])
        })
    
    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.route('/calculate-nev-covariance', methods=['POST'])
def calculate_nev_covariance():
    """
    Calculate NEV covariance matrices for wells with error models
    
    Request body should contain:
    {
        "wells": [
            {
                "well_id": "WELL-001",
                "trajectory": {
                    "well_id": "WELL-001",
                    "survey_points": [...]
                },
                "global_constants": {
                    "g_field": 9.82,
                    "b_field": 51137.759223,
                    "dip": 72.058675,
                    "declination": 0.447619,
                    "grid_convergence": -0.440182
                },
                "ipm_models": [
                    {
                        "model_name": "model_1",
                        "sensors": [
                            {
                                "code": "abxy",
                                "magnitude": 0.002,
                                "mode": "S",
                                "depth_formula": "0",
                                "inclination_formula": "np.cos(inc)/G_Field",
                                "azimuth_formula": "0"
                            },
                            ...
                        ]
                    }
                ],
                "model_constants": [
                    {
                        "earth_rotation": 15.041,
                        "latitude": 59.16524656,
                        "running_speed": 2880.0,
                        ...
                    }
                ],
                "tie_ons": [
                    {
                        "start_depth": 0.0,
                        "end_depth": 198.0
                    }
                ],
                "survey_legs": [[...]],
                "mudline_depth": 197.4
            }
        ],
        "site_uncertainty": {
            "sigma_north": 0.0,
            "sigma_east": 0.0, 
            "sigma_vertical": 0.0
        },
        "well_uncertainty": {
            "sigma_north": 0.0,
            "sigma_east": 0.0,
            "sigma_vertical": 0.0
        }
    }
    
    Response will contain:
    {
        "results": [
            {
                "well_id": "WELL-001",
                "stations": [
                    {
                        "station_index": 0,
                        "depth": 0.0,
                        "total_covariance": {
                            "nn": 1.5e-6,
                            "ne": 0.0,
                            "nv": 0.0,
                            "en": 0.0,
                            "ee": 1.5e-6,
                            "ev": 0.0,
                            "vn": 0.0,
                            "ve": 0.0,
                            "vv": 2.1e-6
                        },
                        "sensor_contributions": {
                            "abxy": {...},
                            ...
                        }
                    },
                    ...
                ],
                "calculation_time": 0.125
            }
        ],
        "total_calculation_time": 0.125
    }
    """
    try:
        # Validate input
        try:
            request_data = NEVCovarianceRequest(**request.json)
        except ValidationError as e:
            logger.error(f"Input validation failed: {e}")
            return jsonify({
                'error': 'Invalid input data',
                'details': e.errors()
            }), 400
        
        logger.info(f"Calculating NEV covariance for {len(request_data.wells)} wells")
        
        # Perform calculation
        response = nev_service.calculate_nev_covariance(request_data)
        
        logger.info(f"NEV calculation completed in {response.total_calculation_time:.3f} seconds")
        
        return jsonify(response.model_dump())
    
    except Exception as e:
        logger.error(f"NEV calculation failed: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.route('/calculate-nev-covariance/batch', methods=['POST'])
def calculate_nev_covariance_batch():
    """
    Calculate NEV covariance matrices for multiple well sets (batch processing)
    
    Request body should contain:
    {
        "calculations": [
            {
                "wells": [...],
                "site_uncertainty": {...},
                "well_uncertainty": {...}
            },
            ...
        ]
    }
    """
    try:
        data = request.json
        if not data or 'calculations' not in data:
            return jsonify({
                'error': 'Invalid input data',
                'message': 'Expected "calculations" array in request body'
            }), 400
        
        batch_results = []
        total_time = 0
        
        for idx, calc_data in enumerate(data['calculations']):
            try:
                # Validate each calculation request
                calc_request = NEVCovarianceRequest(**calc_data)
                
                logger.info(f"Processing NEV batch item {idx+1}/{len(data['calculations'])}: "
                           f"{len(calc_request.wells)} wells")
                
                # Perform calculation
                response = nev_service.calculate_nev_covariance(calc_request)
                
                batch_results.append({
                    'calculation_index': idx,
                    'results': response.results,
                    'calculation_time': response.total_calculation_time
                })
                
                total_time += response.total_calculation_time
                
            except ValidationError as e:
                logger.error(f"Validation failed for NEV batch item {idx+1}: {e}")
                batch_results.append({
                    'calculation_index': idx,
                    'error': f'Invalid input data for batch item {idx+1}',
                    'details': e.errors()
                })
            except Exception as e:
                logger.error(f"NEV calculation failed for batch item {idx+1}: {e}")
                batch_results.append({
                    'calculation_index': idx,
                    'error': f'NEV calculation failed for batch item {idx+1}',
                    'message': str(e)
                })
        
        logger.info(f"NEV batch processing completed in {total_time:.3f} seconds total")
        
        return jsonify({
            'batch_results': batch_results,
            'total_calculation_time': total_time,
            'processed_count': len(data['calculations'])
        })
    
    except Exception as e:
        logger.error(f"NEV batch processing failed: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)