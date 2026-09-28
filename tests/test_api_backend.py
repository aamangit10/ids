import pytest
from backend.app.main import health_check, list_models, predict_csv, predict_packet_flow, run_holdout
from backend.app.schemas.models import PredictFlowRequest, HoldoutExperimentRequest, FlowPacketItem


def test_api_health_direct():
    """Verify health_check endpoint function returns valid status."""
    res = health_check()
    assert res.status == "OK"
    assert res.version == "1.0.0"


def test_api_models_direct():
    """Verify list_models endpoint function returns model catalog."""
    res = list_models()
    assert "models" in res
    assert len(res["models"]) >= 4


def test_api_predict_csv_direct():
    """Verify predict_csv endpoint function predicts risk for sample."""
    sample = {
        "Destination Port": 80,
        "Flow Duration": 15000.0,
        "Total Fwd Packets": 10,
        "Total Backward Packets": 8,
        "Total Length of Fwd Packets": 1200.0,
        "Total Length of Bwd Packets": 800.0,
        "Fwd Packet Length Max": 300.0,
        "Flow IAT Mean": 150.0,
        "SYN Flag Count": 1,
        "ACK Flag Count": 1
    }
    res = predict_csv(sample)
    assert res["prediction"] in ["BENIGN", "MALICIOUS"]
    assert 0.0 <= res["malicious_probability"] <= 1.0


def test_api_predict_flow_direct():
    """Verify predict_packet_flow endpoint function analyzes packet sequence."""
    req = PredictFlowRequest(
        flow_id="test_flow_01",
        packets=[
            FlowPacketItem(packet_index=0, length=150.0, protocol=6, flags=2, inter_arrival_time=0.0, direction=0),
            FlowPacketItem(packet_index=1, length=60.0, protocol=6, flags=18, inter_arrival_time=1.5, direction=1),
            FlowPacketItem(packet_index=2, length=500.0, protocol=6, flags=16, inter_arrival_time=10.0, direction=0)
        ]
    )
    res = predict_packet_flow(req)
    assert res["prediction"] in ["BENIGN", "MALICIOUS"]
    assert "packet_timeline_attributions" in res


def test_api_holdout_experiment_direct():
    """Verify run_holdout endpoint function executes held-out experiment."""
    req = HoldoutExperimentRequest(
        attack_name="Web Attack",
        model_type="random_forest",
        data_dir="demo_data"
    )
    res = run_holdout(req)
    assert res["held_out_train_samples_count"] == 0
    assert "held_out_attack_recall" in res
