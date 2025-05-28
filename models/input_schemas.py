"""
Input validation schemas for anti-collision API
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator


class SurveyPoint(BaseModel):
    """Single survey station point"""
    depth: float = Field(..., description="Measured depth")
    inc: float = Field(..., description="Inclination in degrees")
    az: float = Field(..., description="Azimuth in degrees") 
    tvd: float = Field(..., description="True vertical depth")
    north: float = Field(..., description="North coordinate")
    east: float = Field(..., description="East coordinate")
    

class WellTrajectory(BaseModel):
    """Well trajectory data"""
    well_id: str = Field(..., description="Unique well identifier")
    survey_points: List[SurveyPoint] = Field(..., description="List of survey stations")
    
    @validator('survey_points')
    def validate_survey_points(cls, v):
        if len(v) < 2:
            raise ValueError("Well trajectory must have at least 2 survey points")
        return v


class MinimumDistanceRequest(BaseModel):
    """Request for minimum distance calculation"""
    reference_well: WellTrajectory = Field(..., description="Reference well trajectory")
    offset_wells: List[WellTrajectory] = Field(..., description="Offset well trajectories")
    
    @validator('offset_wells')
    def validate_offset_wells(cls, v):
        if len(v) == 0:
            raise ValueError("At least one offset well is required")
        return v


class DistanceResult(BaseModel):
    """Result for single distance calculation between two survey points"""
    shortest_distance: float = Field(..., description="Shortest distance between reference point and offset well (meters)")
    segment_start_index: int = Field(..., description="Starting survey station index in offset well segment")
    segment_end_index: int = Field(..., description="Ending survey station index in offset well segment") 
    interpolation_factor: float = Field(..., description="Interpolation parameter (0-1) along offset well segment")
    closest_point_coordinates: List[float] = Field(..., description="Coordinates of closest point on offset well [North, East, TVD]")
    direction_vector: List[float] = Field(..., description="Unit direction vector from reference point to closest point [North, East, TVD]")
    offset_well_measured_depth: float = Field(..., description="Measured depth at closest point on offset well")


class WellDistanceResult(BaseModel):
    """Distance results for one well pair"""
    reference_well_id: str = Field(..., description="ID of the reference well")
    offset_well_id: str = Field(..., description="ID of the offset well")
    distances: Dict[int, DistanceResult] = Field(..., description="Distance results keyed by reference survey station index")


class MinimumDistanceResponse(BaseModel):
    """Complete response for minimum distance calculation"""
    results: List[WellDistanceResult] = Field(..., description="Distance results for all well pairs")
    calculation_time: float = Field(..., description="Total calculation time in seconds")