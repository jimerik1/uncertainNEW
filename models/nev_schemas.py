"""
Extended input validation schemas for NEV covariance API
"""
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, validator
from models.input_schemas import WellTrajectory, SurveyPoint


class IPMSensor(BaseModel):
    """Single IPM sensor definition"""
    code: str = Field(..., description="Sensor code/identifier")
    term_description: Optional[str] = Field(None, description="Description of the sensor term")
    wt_fn: Optional[str] = Field(None, description="Weight function")
    sensor_type: Optional[str] = Field(None, description="Sensor type (I, A, L, D)")
    magnitude: float = Field(..., description="Sensor magnitude/uncertainty")
    units: Optional[str] = Field(None, description="Units")
    mode: str = Field(..., description="Propagation mode (G, W, S, R)")
    wt_fn_comment: Optional[str] = Field(None, description="Weight function comment")
    depth_formula: str = Field(..., description="Depth propagation formula")
    inclination_formula: str = Field(..., description="Inclination propagation formula") 
    azimuth_formula: str = Field(..., description="Azimuth propagation formula")
    propagation: Optional[str] = Field(None, description="Propagation type")
    type_field: Optional[str] = Field(None, description="Type field")
    initialized: bool = Field(False, description="Whether sensor is initialized")
    h: float = Field(0.0, description="Height parameter")
    inc0: float = Field(0.0, description="Initial inclination")
    inc1: float = Field(180.0, description="Final inclination")
    d_init: float = Field(0.0, description="Initial depth")


class IPMModel(BaseModel):
    """IPM sensor model containing multiple sensors"""
    name: str = Field(..., description="Name/identifier for this model")
    sensors: List[IPMSensor] = Field(..., description="List of sensors in this model")
    
    @validator('sensors')
    def validate_sensors(cls, v):
        if len(v) == 0:
            raise ValueError("Model must contain at least one sensor")
        return v


class GlobalConstants(BaseModel):
    """Global constants for error calculations"""
    g_field: float = Field(9.82, description="Gravity field strength (m/s²)")
    b_field: float = Field(..., description="Magnetic field strength (nT)")
    dip: float = Field(..., description="Magnetic dip angle (degrees)")
    declination: float = Field(..., description="Magnetic declination (degrees)")
    grid_convergence: float = Field(..., description="Grid convergence (degrees)")


class ModelConstants(BaseModel):
    """Model constants for error calculations"""
    earth_rotation: float = Field(15.041, description="Earth rotation rate (deg/h)")
    latitude: float = Field(..., description="Latitude (degrees)")
    running_speed: float = Field(2880.0, description="Running speed (m/h)")
    inc_init: float = Field(180.0, description="Initialization inclination constraint (degrees)")
    noise_reduction_factor: float = Field(1.0, description="Noise reduction factor")
    min_distance: float = Field(0.0, description="Minimum distance between initializations (m)")
    cant_angle: float = Field(20.0, description="Cant angle (degrees)")
    inc_init_constraint: float = Field(15.0, description="Inclination initialization constraint (degrees)")
    az_init_constraint: float = Field(45.0, description="Azimuth initialization constraint (degrees)")
    depth_init: float = Field(1000.0, description="Depth initialization (m)")
    survey_start: int = Field(0, description="Survey start index")


class SurveyLeg(BaseModel):
    """Survey leg definition"""
    start_depth: float = Field(..., description="Start depth of survey leg (m)")
    end_depth: float = Field(..., description="End depth of survey leg (m)")


class TieOn(BaseModel):
    """Tie-on definition between models"""
    start_depth: float = Field(..., description="Start depth for this model (m)")
    end_depth: float = Field(..., description="End depth for this model (m)")


class WellErrorModel(BaseModel):
    """Complete error model for a single well"""
    well_id: str = Field(..., description="Well identifier")
    trajectory: WellTrajectory = Field(..., description="Well trajectory")
    global_constants: GlobalConstants = Field(..., description="Global constants")
    ipm_models: List[IPMModel] = Field(..., description="List of IPM sensor models")
    constants: List[ModelConstants] = Field(..., description="Model constants for each IPM model")
    tie_ons: List[TieOn] = Field(..., description="Tie-on intervals for each model")
    survey_legs: List[List[SurveyLeg]] = Field(..., description="Survey legs for each model")
    mudline_depth: float = Field(0.0, description="Mudline depth (m)")
    
    @validator('ipm_models')
    def validate_ipm_models(cls, v):
        if len(v) == 0:
            raise ValueError("At least one IPM model is required")
        return v
    
    @validator('constants')
    def validate_constants_count(cls, v, values):
        if 'ipm_models' in values and len(v) != len(values['ipm_models']):
            raise ValueError("Number of model constants must match number of IPM models")
        return v
    
    @validator('tie_ons')
    def validate_tie_ons_count(cls, v, values):
        if 'ipm_models' in values and len(v) != len(values['ipm_models']):
            raise ValueError("Number of tie-ons must match number of IPM models")
        return v


class SiteUncertainty(BaseModel):
    """Site-level uncertainty (affects all wells from same site)"""
    sigma_north: float = Field(0.0, description="North uncertainty standard deviation (m)")
    sigma_east: float = Field(0.0, description="East uncertainty standard deviation (m)")
    sigma_vertical: float = Field(0.0, description="Vertical uncertainty standard deviation (m)")


class WellUncertainty(BaseModel):
    """Well-level uncertainty (affects all wellbores in same well)"""
    sigma_north: float = Field(0.0, description="North uncertainty standard deviation (m)")
    sigma_east: float = Field(0.0, description="East uncertainty standard deviation (m)")
    sigma_vertical: float = Field(0.0, description="Vertical uncertainty standard deviation (m)")


class NEVCovarianceRequest(BaseModel):
    """Request for NEV covariance matrix calculation"""
    wells: List[WellErrorModel] = Field(..., description="List of wells with error models")
    site_uncertainty: Optional[SiteUncertainty] = Field(None, description="Site-level uncertainty")
    well_uncertainty: Optional[WellUncertainty] = Field(None, description="Well-level uncertainty")
    
    @validator('wells')
    def validate_wells(cls, v):
        if len(v) == 0:
            raise ValueError("At least one well is required")
        return v


class NEVMatrix(BaseModel):
    """3x3 NEV covariance matrix"""
    nn: float = Field(..., description="North-North covariance")
    ne: float = Field(..., description="North-East covariance") 
    nv: float = Field(..., description="North-Vertical covariance")
    en: float = Field(..., description="East-North covariance")
    ee: float = Field(..., description="East-East covariance")
    ev: float = Field(..., description="East-Vertical covariance")
    vn: float = Field(..., description="Vertical-North covariance")
    ve: float = Field(..., description="Vertical-East covariance")
    vv: float = Field(..., description="Vertical-Vertical covariance")


class NEVStationResult(BaseModel):
    """NEV covariance result for a single survey station"""
    station_index: int = Field(..., description="Survey station index")
    depth: float = Field(..., description="Measured depth at this station")
    total_covariance: NEVMatrix = Field(..., description="Total NEV covariance matrix")
    sensor_contributions: Dict[str, NEVMatrix] = Field(default_factory=dict, description="Individual sensor contributions")


class NEVWellResult(BaseModel):
    """NEV covariance results for a single well"""
    well_id: str = Field(..., description="Well identifier")
    stations: List[NEVStationResult] = Field(..., description="Results for each survey station")
    calculation_time: float = Field(..., description="Calculation time for this well (seconds)")


class NEVCovarianceResponse(BaseModel):
    """Complete response for NEV covariance calculation"""
    results: List[NEVWellResult] = Field(..., description="Results for all wells")
    total_calculation_time: float = Field(..., description="Total calculation time (seconds)")