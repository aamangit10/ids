from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional


class FlowPacketItem(BaseModel):
    packet_index: int = Field(..., description="0-indexed packet order in flow")
    length: float = Field(..., description="Packet payload/header length in bytes")
    protocol: int = Field(6, description="IP protocol number (6=TCP, 17=UDP)")
    flags: int = Field(0, description="TCP flags bitmask")
    inter_arrival_time: float = Field(0.0, description="Inter-arrival time since previous packet in ms")
    direction: int = Field(0, description="Packet direction (0=Forward, 1=Backward)")


class PredictFlowRequest(BaseModel):
    flow_id: Optional[str] = Field("flow_sample_01", description="Non-sensitive flow tracking ID")
    packets: List[FlowPacketItem] = Field(..., description="List of packet tokens (up to 32 max)")
    model_name: Optional[str] = Field("transformer", description="Model choice: transformer, random_forest, xgboost, logistic_regression")


class HoldoutExperimentRequest(BaseModel):
    attack_name: str = Field("Web Attack", description="Attack class to hold out from training partition")
    model_type: str = Field("random_forest", description="Baseline model type: random_forest, xgboost, logistic_regression")
    data_dir: str = Field("demo_data", description="Directory path containing dataset CSVs")


class ExplainRequest(BaseModel):
    sample_features: Optional[Dict[str, float]] = Field(None, description="Tabular feature key-value dictionary for SHAP explanation")
    flow_packets: Optional[List[FlowPacketItem]] = Field(None, description="Packet sequence for Integrated Gradients explanation")
    model_name: str = Field("random_forest", description="Model name to explain")


class HealthResponse(BaseModel):
    status: str
    message: str
    version: str
    device: str
    demo_mode: bool
