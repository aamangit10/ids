import os
import json
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any

# Standard feature set mirroring CICIDS2017 Machine Learning CSVs
CICIDS_FEATURE_COLUMNS = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean",
    "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean",
    "Fwd IAT Std", "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Fwd PSH Flags",
    "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length",
    "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length",
    "Max Packet Length", "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
    "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Header Length.1", "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets",
    "Subflow Bwd Bytes", "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean", "Active Std",
    "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
    # Data leakage columns (to be cleaned by preprocessor)
    "Flow ID", "Source IP", "Source Port", "Destination IP", "Timestamp"
]

ATTACK_CLASSES = [
    "BENIGN",
    "DDoS",
    "DoS Slowloris",
    "PortScan",
    "Bot",
    "Web Attack",
    "Brute Force",
    "FTP-Patator"
]


def generate_synthetic_tabular_dataset(n_samples: int = 2400, seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic tabular dataset modeled after CICIDS2017.
    Includes data leakage columns (Source IP, Timestamp) to verify preprocessor stripping.
    """
    np.random.seed(seed)
    data = []

    labels = np.random.choice(
        ATTACK_CLASSES,
        size=n_samples,
        p=[0.55, 0.10, 0.08, 0.10, 0.05, 0.05, 0.04, 0.03]
    )

    for i, label in enumerate(labels):
        row = {}
        # Base metadata (Data Leakage columns)
        src_ip = f"192.168.10.{np.random.randint(2, 250)}"
        dst_ip = f"172.16.0.{np.random.randint(2, 50)}"
        src_port = int(np.random.randint(1024, 65535))
        dst_port = 80 if label in ["BENIGN", "Web Attack"] else (443 if label == "DDoS" else int(np.random.randint(21, 8080)))
        
        row["Flow ID"] = f"{src_ip}-{dst_ip}-{src_port}-{dst_port}-6"
        row["Source IP"] = src_ip
        row["Source Port"] = src_port
        row["Destination IP"] = dst_ip
        row["Destination Port"] = dst_port
        row["Timestamp"] = "05/07/2017 09:30:00 AM"

        # Feature distributions based on attack types
        if label == "BENIGN":
            duration = np.random.exponential(scale=50000)
            fwd_pkts = np.random.poisson(lam=12) + 1
            bwd_pkts = np.random.poisson(lam=10) + 1
            fwd_len_mean = np.random.normal(loc=150, scale=40)
            iat_mean = np.random.normal(loc=2000, scale=300)
            syn_count = np.random.binomial(n=1, p=0.3)
            ack_count = np.random.binomial(n=1, p=0.9)
            rst_count = 0
        elif label == "DDoS":
            duration = np.random.exponential(scale=1000)
            fwd_pkts = np.random.poisson(lam=80) + 10
            bwd_pkts = np.random.poisson(lam=2) + 1
            fwd_len_mean = np.random.normal(loc=800, scale=100)
            iat_mean = np.random.exponential(scale=10)  # Very short IAT
            syn_count = 1
            ack_count = 0
            rst_count = np.random.binomial(n=1, p=0.4)
        elif label == "PortScan":
            duration = np.random.uniform(10, 500)
            fwd_pkts = 2
            bwd_pkts = 1
            fwd_len_mean = 40
            iat_mean = np.random.uniform(5, 50)
            syn_count = 1
            ack_count = 0
            rst_count = 1
        elif label == "Web Attack":
            duration = np.random.exponential(scale=30000)
            fwd_pkts = np.random.poisson(lam=25) + 2
            bwd_pkts = np.random.poisson(lam=20) + 2
            fwd_len_mean = np.random.normal(loc=450, scale=120)  # High payload variability
            iat_mean = np.random.normal(loc=1200, scale=200)
            syn_count = 1
            ack_count = 1
            rst_count = 0
        else:  # DoS, Bot, Brute Force, FTP-Patator
            duration = np.random.exponential(scale=15000)
            fwd_pkts = np.random.poisson(lam=30) + 1
            bwd_pkts = np.random.poisson(lam=5) + 1
            fwd_len_mean = np.random.normal(loc=300, scale=80)
            iat_mean = np.random.exponential(scale=100)
            syn_count = 1
            ack_count = 0
            rst_count = 0

        row["Flow Duration"] = max(1.0, float(duration))
        row["Total Fwd Packets"] = max(1, int(fwd_pkts))
        row["Total Backward Packets"] = max(0, int(bwd_pkts))
        
        fwd_len_max = float(abs(fwd_len_mean * 1.8 + np.random.normal(0, 20)))
        bwd_len_max = float(abs(fwd_len_mean * 1.5 + np.random.normal(0, 20)))
        
        row["Fwd Packet Length Max"] = fwd_len_max
        row["Fwd Packet Length Min"] = float(abs(np.random.normal(20, 5)))
        row["Fwd Packet Length Mean"] = float(abs(fwd_len_mean))
        row["Fwd Packet Length Std"] = float(abs(fwd_len_mean * 0.3))
        
        row["Bwd Packet Length Max"] = bwd_len_max
        row["Bwd Packet Length Min"] = 0.0
        row["Bwd Packet Length Mean"] = float(abs(fwd_len_mean * 0.8))
        row["Bwd Packet Length Std"] = float(abs(fwd_len_mean * 0.2))

        row["Total Length of Fwd Packets"] = float(row["Total Fwd Packets"] * row["Fwd Packet Length Mean"])
        row["Total Length of Bwd Packets"] = float(row["Total Backward Packets"] * row["Bwd Packet Length Mean"])

        row["Flow Bytes/s"] = float((row["Total Length of Fwd Packets"] + row["Total Length of Bwd Packets"]) / (row["Flow Duration"] / 1e6 + 1e-5))
        row["Flow Packets/s"] = float((row["Total Fwd Packets"] + row["Total Backward Packets"]) / (row["Flow Duration"] / 1e6 + 1e-5))

        row["Flow IAT Mean"] = float(abs(iat_mean))
        row["Flow IAT Std"] = float(abs(iat_mean * 0.5))
        row["Flow IAT Max"] = float(abs(iat_mean * 2.5))
        row["Flow IAT Min"] = float(abs(iat_mean * 0.1))

        row["Fwd IAT Total"] = float(row["Flow Duration"])
        row["Fwd IAT Mean"] = float(abs(iat_mean * 1.1))
        row["Fwd IAT Std"] = float(abs(iat_mean * 0.4))
        row["Bwd IAT Total"] = float(row["Flow Duration"] * 0.8)
        row["Bwd IAT Mean"] = float(abs(iat_mean * 1.2))
        row["Bwd IAT Std"] = float(abs(iat_mean * 0.3))

        row["Fwd PSH Flags"] = 0
        row["Bwd PSH Flags"] = 0
        row["Fwd URG Flags"] = 0
        row["Bwd URG Flags"] = 0
        row["Fwd Header Length"] = int(row["Total Fwd Packets"] * 20)
        row["Bwd Header Length"] = int(row["Total Backward Packets"] * 20)
        row["Fwd Header Length.1"] = row["Fwd Header Length"]

        row["Fwd Packets/s"] = float(row["Total Fwd Packets"] / (row["Flow Duration"] / 1e6 + 1e-5))
        row["Bwd Packets/s"] = float(row["Total Backward Packets"] / (row["Flow Duration"] / 1e6 + 1e-5))

        row["Min Packet Length"] = 0.0
        row["Max Packet Length"] = max(fwd_len_max, bwd_len_max)
        row["Packet Length Mean"] = float((row["Fwd Packet Length Mean"] + row["Bwd Packet Length Mean"]) / 2)
        row["Packet Length Std"] = float(abs(row["Packet Length Mean"] * 0.4))
        row["Packet Length Variance"] = float(row["Packet Length Std"] ** 2)

        row["FIN Flag Count"] = 0
        row["SYN Flag Count"] = int(syn_count)
        row["RST Flag Count"] = int(rst_count)
        row["PSH Flag Count"] = int(np.random.binomial(n=1, p=0.4))
        row["ACK Flag Count"] = int(ack_count)
        row["URG Flag Count"] = 0
        row["CWE Flag Count"] = 0
        row["ECE Flag Count"] = 0

        row["Down/Up Ratio"] = float(row["Total Backward Packets"] / (row["Total Fwd Packets"] + 1e-5))
        row["Average Packet Size"] = float(row["Packet Length Mean"])
        row["Avg Fwd Segment Size"] = float(row["Fwd Packet Length Mean"])
        row["Avg Bwd Segment Size"] = float(row["Bwd Packet Length Mean"])

        row["Subflow Fwd Packets"] = row["Total Fwd Packets"]
        row["Subflow Fwd Bytes"] = row["Total Length of Fwd Packets"]
        row["Subflow Bwd Packets"] = row["Total Backward Packets"]
        row["Subflow Bwd Bytes"] = row["Total Length of Bwd Packets"]

        row["Init_Win_bytes_forward"] = int(8192)
        row["Init_Win_bytes_backward"] = int(8192)
        row["act_data_pkt_fwd"] = max(1, int(row["Total Fwd Packets"] * 0.8))
        row["min_seg_size_forward"] = 20

        row["Active Mean"] = float(np.random.exponential(scale=1000))
        row["Active Std"] = 0.0
        row["Active Max"] = float(row["Active Mean"])
        row["Active Min"] = float(row["Active Mean"])
        row["Idle Mean"] = float(np.random.exponential(scale=5000))
        row["Idle Std"] = 0.0
        row["Idle Max"] = float(row["Idle Mean"])
        row["Idle Min"] = float(row["Idle Mean"])

        row["Label"] = label
        data.append(row)

    df = pd.DataFrame(data)
    return df


def generate_synthetic_packet_sequences(n_flows: int = 600, max_seq_len: int = 32, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Generates synthetic packet sequence flows (Packet Mode).
    Each flow is grouped by 5-tuple, but IP addresses are excluded from token vectors.
    Packet features per token: [length, protocol, flags, inter_arrival_time, direction].
    """
    np.random.seed(seed)
    flows = []

    for i in range(n_flows):
        label = np.random.choice(ATTACK_CLASSES, p=[0.50, 0.12, 0.10, 0.10, 0.06, 0.05, 0.04, 0.03])
        num_packets = int(np.random.randint(4, max_seq_len + 1))
        
        packets = []
        for p_idx in range(num_packets):
            # Direction: 0 = Forward, 1 = Backward
            direction = 0 if p_idx % 2 == 0 or label in ["DDoS", "PortScan"] else 1
            
            if label == "BENIGN":
                pkt_len = float(np.random.normal(180, 40))
                iat = float(abs(np.random.normal(15, 5)))
                flags = 2 if p_idx == 0 else (16 if p_idx > 1 else 18)  # SYN -> SYN-ACK -> ACK
            elif label == "DDoS":
                pkt_len = float(np.random.normal(850, 50))
                iat = float(abs(np.random.exponential(0.5)))
                flags = 2  # SYN flood
            elif label == "PortScan":
                pkt_len = 40.0
                iat = float(abs(np.random.uniform(0.1, 2.0)))
                flags = 2 if direction == 0 else 4  # SYN / RST
            elif label == "Web Attack":
                pkt_len = float(np.random.normal(520, 150))
                iat = float(abs(np.random.normal(8.0, 3.0)))
                flags = 24  # PSH-ACK
            else:
                pkt_len = float(np.random.normal(300, 80))
                iat = float(abs(np.random.exponential(3.0)))
                flags = 16

            packets.append({
                "packet_index": p_idx,
                "length": max(20.0, float(pkt_len)),
                "protocol": 6,  # TCP
                "flags": int(flags),
                "inter_arrival_time": max(0.0, float(iat)),
                "direction": int(direction)
            })

        flow_entry = {
            "flow_id": f"flow_{i:05d}",
            "label": label,
            "is_attack": 0 if label == "BENIGN" else 1,
            "packets": packets
        }
        flows.append(flow_entry)

    return flows


def create_demo_data_if_missing(demo_dir: str = "demo_data") -> Tuple[str, str]:
    """Ensure synthetic demo dataset files exist and return file paths."""
    os.makedirs(demo_dir, exist_ok=True)
    csv_path = os.path.join(demo_dir, "cicids2017_demo.csv")
    json_path = os.path.join(demo_dir, "packet_flows_demo.json")

    if not os.path.exists(csv_path):
        df = generate_synthetic_tabular_dataset(n_samples=2500, seed=42)
        df.to_csv(csv_path, index=False)

    if not os.path.exists(json_path):
        flows = generate_synthetic_packet_sequences(n_flows=600, seed=42)
        with open(json_path, "w") as f:
            json.dump(flows, f, indent=2)

    return csv_path, json_path
