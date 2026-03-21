"""Pydantic schemas for physics calculation and analysis endpoints."""
from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class SingleCalculationRequest(BaseModel):
    composition: Dict[str, float] = Field(..., description="Element symbols to weight %")
    frequency_mhz: float = Field(..., gt=0.001, lt=100000)
    thickness_mm: float = Field(..., gt=0.001, lt=100)
    grain_size_um: Optional[float] = Field(None, gt=0)

    class Config:
        json_schema_extra = {
            "example": {
                "composition": {"Cu": 100},
                "frequency_mhz": 1000,
                "thickness_mm": 1.0,
                "grain_size_um": 50.0,
            }
        }


class CalculationResponse(BaseModel):
    shielding_effectiveness_db: float
    reflection_loss_db: float
    absorption_loss_db: float
    multiple_reflection_db: float
    skin_depth_um: float
    effective_conductivity: float
    execution_time_ms: float
    confidence: Optional[float] = None
    confidence_level: Optional[str] = None


class FrequencySweepRequest(BaseModel):
    composition: Dict[str, float]
    thickness_mm: float = Field(..., gt=0.001, lt=100)
    freq_start_mhz: float = Field(..., gt=0.001)
    freq_end_mhz: float = Field(..., gt=0.001)
    num_points: int = Field(100, gt=1, le=1000)
    grain_size_um: Optional[float] = Field(None, gt=0)


class ThicknessSweepRequest(BaseModel):
    composition: Dict[str, float]
    frequency_mhz: float = Field(..., gt=0.001)
    thickness_start_mm: float = Field(0.1, gt=0.001)
    thickness_end_mm: float = Field(10.0, gt=0.001)
    num_points: int = Field(100, gt=1, le=1000)
    grain_size_um: Optional[float] = Field(None, gt=0)


class GrainSizeSweepRequest(BaseModel):
    composition: Dict[str, float]
    frequency_mhz: float = Field(..., gt=0.001)
    thickness_mm: float = Field(..., gt=0.001)
    grain_start_um: float = Field(0.01, gt=0)
    grain_end_um: float = Field(100.0, gt=0)
    num_points: int = Field(100, gt=1, le=1000)


class CoolingRateSweepRequest(BaseModel):
    composition: Dict[str, float]
    frequency_mhz: float = Field(..., gt=0.001)
    thickness_mm: float = Field(..., gt=0.001)
    cooling_rate_min: float = Field(0.1, gt=0)
    cooling_rate_max: float = Field(1000.0, gt=0)
    num_points: int = Field(50, gt=1, le=500)


class ThicknessOptimizationRequest(BaseModel):
    composition: Dict[str, float]
    frequency_mhz: float = Field(..., gt=0.001)
    target_se_db: float = Field(..., gt=0)
    thickness_max_mm: float = Field(10.0, gt=0.001)
    grain_size_um: Optional[float] = Field(None, gt=0)
