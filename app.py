"""
Flask API for Anti-Collision Calculations
"""
from flask import Flask, request, jsonify
from pydantic import ValidationError
import traceback
import logging

from models.input_schemas import MinimumDistanceRequest, MinimumDistanceResponse
from services.distance_service import DistanceCalculationService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Initialize services
distance_service = DistanceCalculationService()


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'anti-collision-api',
        'version': '1.0.0'
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
        
        return jsonify(response.dict())
    
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
                    'results': [r.dict() for r in results],
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


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)