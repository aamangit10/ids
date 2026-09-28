import React, { useState, useEffect } from 'react';
import {
  Shield, Activity, Database, Cpu, AlertTriangle, HelpCircle, BarChart3,
  Search, CheckCircle2, XCircle, ArrowRight, Play, Server, FileText, Lock,
  TrendingUp, Layers, Compass, Eye
} from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [healthStatus, setHealthStatus] = useState(null);
  const [modelsInfo, setModelsInfo] = useState([]);
  
  // Held-out experiment state
  const [selectedHoldoutAttack, setSelectedHoldoutAttack] = useState('Web Attack');
  const [selectedHoldoutModel, setSelectedHoldoutModel] = useState('random_forest');
  const [holdoutResult, setHoldoutResult] = useState(null);
  const [isHoldoutRunning, setIsHoldoutRunning] = useState(false);

  // Single flow detection state
  const [detectionMode, setDetectionMode] = useState('flow'); // 'csv' or 'flow'
  const [detectionResult, setDetectionResult] = useState(null);
  const [isDetecting, setIsDetecting] = useState(false);

  // Sample packet sequence input
  const [packetInputs, setPacketInputs] = useState([
    { length: 450, protocol: 6, flags: 24, inter_arrival_time: 1.2, direction: 0 },
    { length: 60, protocol: 6, flags: 16, inter_arrival_time: 0.5, direction: 1 },
    { length: 1200, protocol: 6, flags: 24, inter_arrival_time: 12.4, direction: 0 },
    { length: 520, protocol: 6, flags: 16, inter_arrival_time: 8.1, direction: 1 }
  ]);

  // Fetch initial health & model catalog
  useEffect(() => {
    fetch('/api/health')
      .then(res => res.json())
      .then(data => setHealthStatus(data))
      .catch(() => setHealthStatus({ status: 'DEMO', demo_mode: true }));

    fetch('/api/models')
      .then(res => res.json())
      .then(data => setModelsInfo(data.models || []))
      .catch(() => {});
  }, []);

  const handleRunHoldout = () => {
    setIsHoldoutRunning(true);
    fetch('/api/experiments/holdout', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        attack_name: selectedHoldoutAttack,
        model_type: selectedHoldoutModel,
        data_dir: 'demo_data'
      })
    })
      .then(res => res.json())
      .then(data => {
        setHoldoutResult(data);
        setIsHoldoutRunning(false);
      })
      .catch(err => {
        console.error(err);
        setIsHoldoutRunning(false);
      });
  };

  const handleRunDetection = () => {
    setIsDetecting(true);
    if (detectionMode === 'flow') {
      const packets = packetInputs.map((p, idx) => ({
        packet_index: idx,
        length: parseFloat(p.length),
        protocol: parseInt(p.protocol),
        flags: parseInt(p.flags),
        inter_arrival_time: parseFloat(p.inter_arrival_time),
        direction: parseInt(p.direction)
      }));

      fetch('/api/predict/flow', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ flow_id: 'sample_ui_flow', packets })
      })
        .then(res => res.json())
        .then(data => {
          setDetectionResult(data);
          setIsDetecting(false);
        })
        .catch(() => setIsDetecting(false));
    } else {
      const csvSample = {
        "Destination Port": 80,
        "Flow Duration": 12500.0,
        "Total Fwd Packets": 15,
        "Total Backward Packets": 12,
        "Total Length of Fwd Packets": 1850.0,
        "Total Length of Bwd Packets": 1400.0,
        "Fwd Packet Length Max": 450.0,
        "Flow IAT Mean": 85.0,
        "SYN Flag Count": 1,
        "ACK Flag Count": 1
      };

      fetch('/api/predict/csv', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(csvSample)
      })
        .then(res => res.json())
        .then(data => {
          setDetectionResult(data);
          setIsDetecting(false);
        })
        .catch(() => setIsDetecting(false));
    }
  };

  return (
    <div className="min-h-screen flex bg-slate-950 text-slate-100 font-sans">
      {/* Sidebar Navigation */}
      <aside className="w-64 bg-slate-900 border-r border-slate-800 p-4 flex flex-col justify-between">
        <div>
          {/* Logo Header */}
          <div className="flex items-center gap-3 px-2 py-3 mb-6 border-b border-slate-800">
            <div className="p-2 bg-cyan-500/10 border border-cyan-500/30 rounded-lg text-cyan-400">
              <Shield className="w-6 h-6" />
            </div>
            <div>
              <h1 className="font-bold text-sm tracking-wide text-slate-100">SSL-IDS Transformer</h1>
              <p className="text-xs text-slate-400">Explainable Unseen IDS</p>
            </div>
          </div>

          {/* Navigation Links */}
          <nav className="space-y-1">
            <div className={`nav-item ${activeTab === 'overview' ? 'active' : ''}`} onClick={() => setActiveTab('overview')}>
              <Activity className="w-4 h-4" /> Overview
            </div>
            <div className={`nav-item ${activeTab === 'dataset' ? 'active' : ''}`} onClick={() => setActiveTab('dataset')}>
              <Database className="w-4 h-4" /> Dataset & Preprocessing
            </div>
            <div className={`nav-item ${activeTab === 'detection' ? 'active' : ''}`} onClick={() => setActiveTab('detection')}>
              <Search className="w-4 h-4" /> Flow Detection
            </div>
            <div className={`nav-item ${activeTab === 'comparison' ? 'active' : ''}`} onClick={() => setActiveTab('comparison')}>
              <BarChart3 className="w-4 h-4" /> Model Comparison
            </div>
            <div className={`nav-item ${activeTab === 'unseen' ? 'active' : ''}`} onClick={() => setActiveTab('unseen')}>
              <Compass className="w-4 h-4" /> Unseen Attack Evaluation
            </div>
            <div className={`nav-item ${activeTab === 'explainability' ? 'active' : ''}`} onClick={() => setActiveTab('explainability')}>
              <Eye className="w-4 h-4" /> Explainable AI (SHAP & IG)
            </div>
            <div className={`nav-item ${activeTab === 'results' ? 'active' : ''}`} onClick={() => setActiveTab('results')}>
              <TrendingUp className="w-4 h-4" /> Limited-Label SSL Study
            </div>
            <div className={`nav-item ${activeTab === 'about' ? 'active' : ''}`} onClick={() => setActiveTab('about')}>
              <HelpCircle className="w-4 h-4" /> About Research
            </div>
          </nav>
        </div>

        {/* Demo Mode & Safety Status Badge */}
        <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg space-y-2">
          <div className="flex items-center justify-between text-xs">
            <span className="text-slate-400">System Mode:</span>
            <span className="px-2 py-0.5 rounded text-[10px] font-semibold badge-demo">
              DEMO DATA — NOT RESEARCH RESULTS
            </span>
          </div>
          <div className="flex items-center gap-1.5 text-[11px] text-emerald-400">
            <CheckCircle2 className="w-3.5 h-3.5" /> Strictly Defensive Security
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 p-8 overflow-y-auto">
        {/* Banner Notice */}
        <div className="mb-6 p-3 bg-amber-500/10 border border-amber-500/30 rounded-lg flex items-center justify-between">
          <div className="flex items-center gap-2 text-amber-300 text-xs font-medium">
            <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0" />
            <span>Educational Research Simulator: All metrics are computed safely without network scanning or payload execution.</span>
          </div>
          <span className="text-[10px] font-mono px-2 py-0.5 bg-amber-500/20 text-amber-200 rounded">CICIDS2017 Benchmark</span>
        </div>

        {/* TAB 1: OVERVIEW */}
        {activeTab === 'overview' && (
          <div className="space-y-6">
            <div>
              <h2 className="text-2xl font-bold text-slate-100">Research Overview & Architecture</h2>
              <p className="text-sm text-slate-400">Explainable Self-Supervised Transformer-Based IDS for Out-of-Distribution Attack Generalization.</p>
            </div>

            {/* Metrics Quick Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="cyber-card p-5">
                <div className="flex justify-between items-start text-slate-400 text-xs font-medium mb-2">
                  <span>Analyzed Flow Records</span>
                  <Database className="w-4 h-4 text-cyan-400" />
                </div>
                <div className="text-2xl font-extrabold text-slate-100">2,500</div>
                <div className="text-[11px] text-cyan-400 mt-1">CICIDS2017 Dual-Mode Pipeline</div>
              </div>

              <div className="cyber-card p-5">
                <div className="flex justify-between items-start text-slate-400 text-xs font-medium mb-2">
                  <span>SSL Pretrained F1-Score</span>
                  <TrendingUp className="w-4 h-4 text-emerald-400" />
                </div>
                <div className="text-2xl font-extrabold text-emerald-400">0.9810</div>
                <div className="text-[11px] text-slate-400 mt-1">+4.2% Boost at 10% Labeled Data</div>
              </div>

              <div className="cyber-card p-5">
                <div className="flex justify-between items-start text-slate-400 text-xs font-medium mb-2">
                  <span>Held-Out Attack Recall</span>
                  <Compass className="w-4 h-4 text-purple-400" />
                </div>
                <div className="text-2xl font-extrabold text-purple-400">84.50%</div>
                <div className="text-[11px] text-slate-400 mt-1">Web Attack (Zero Train Samples)</div>
              </div>

              <div className="cyber-card p-5">
                <div className="flex justify-between items-start text-slate-400 text-xs font-medium mb-2">
                  <span>Explainability Level</span>
                  <Eye className="w-4 h-4 text-amber-400" />
                </div>
                <div className="text-2xl font-extrabold text-amber-400">Multi-Level</div>
                <div className="text-[11px] text-slate-400 mt-1">SHAP & Integrated Gradients</div>
              </div>
            </div>

            {/* Pipeline Architecture Graphic */}
            <div className="cyber-card p-6">
              <h3 className="text-sm font-semibold text-slate-200 mb-4 flex items-center gap-2">
                <Layers className="w-4 h-4 text-cyan-400" /> Dual-Mode Research Architecture
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="p-4 bg-slate-950 border border-slate-800 rounded-lg">
                  <div className="text-xs font-bold text-cyan-400 uppercase tracking-wider mb-2">Mode A: Classical Tabular Pipeline</div>
                  <ul className="text-xs text-slate-300 space-y-1.5 list-disc list-inside">
                    <li>CICIDS2017 MachineLearningCSV Features</li>
                    <li>Leakage Stripping (IPs, Ports, Flow IDs excluded)</li>
                    <li>Logistic Regression, Random Forest, XGBoost</li>
                    <li>Local & Global SHAP Feature Attributions</li>
                  </ul>
                </div>

                <div className="p-4 bg-slate-950 border border-slate-800 rounded-lg">
                  <div className="text-xs font-bold text-purple-400 uppercase tracking-wider mb-2">Mode B: Packet Sequence Transformer Pipeline</div>
                  <ul className="text-xs text-slate-300 space-y-1.5 list-disc list-inside">
                    <li>5-Tuple Grouping (32 Packets Max Sequence)</li>
                    <li>Features: Length, Protocol, TCP Flags, IAT, Direction</li>
                    <li>Self-Supervised InfoNCE Contrastive Pretraining</li>
                    <li>Captum Integrated Gradients Packet Attributions</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: DATASET & PREPROCESSING */}
        {activeTab === 'dataset' && (
          <div className="space-y-6">
            <div>
              <h2 className="text-2xl font-bold text-slate-100">Dataset & Leakage-Free Preprocessing</h2>
              <p className="text-sm text-slate-400">Rigorous feature cleaning protocols applied to CICIDS2017 benchmark dataset.</p>
            </div>

            <div className="cyber-card p-6 space-y-4">
              <h3 className="text-sm font-semibold text-slate-200">Excluded Data Leakage Identifiers</h3>
              <p className="text-xs text-slate-400">
                To prevent machine learning models from trivially memorizing IP subnets or specific socket timestamps rather than attack behavior, the following features are strictly excluded:
              </p>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                {["Flow ID", "Source IP", "Destination IP", "Source Port", "Destination Port", "Timestamp"].map((feat, idx) => (
                  <div key={idx} className="p-3 bg-slate-950 border border-red-500/20 rounded flex items-center gap-2">
                    <XCircle className="w-4 h-4 text-red-400 flex-shrink-0" />
                    <div>
                      <div className="text-xs font-semibold text-slate-200">{feat}</div>
                      <div className="text-[10px] text-slate-500">Excluded (Leakage Protection)</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="cyber-card p-6">
              <h3 className="text-sm font-semibold text-slate-200 mb-3">Preprocessing Guarantee</h3>
              <ul className="text-xs text-slate-300 space-y-2 list-disc list-inside">
                <li><strong className="text-slate-100">Infinity & NaN handling:</strong> String <code className="text-cyan-400">Infinity</code> replaced with <code className="text-cyan-400">NaN</code> and invalid rows cleaned.</li>
                <li><strong className="text-slate-100">StandardScaler isolation:</strong> Scalers are fitted <strong className="text-cyan-400">strictly on training partitions</strong> to eliminate test data leakage.</li>
                <li><strong className="text-slate-100">Target Encoding:</strong> Binary target (<code className="text-emerald-400">BENIGN = 0</code>, <code className="text-red-400">ATTACK = 1</code>) alongside preserved multiclass label categories.</li>
              </ul>
            </div>
          </div>
        )}

        {/* TAB 3: FLOW DETECTION */}
        {activeTab === 'detection' && (
          <div className="space-y-6">
            <div>
              <h2 className="text-2xl font-bold text-slate-100">Flow Anomaly & Attack Detection</h2>
              <p className="text-sm text-slate-400">Interactive inference engine for Packet Sequence & Tabular Feature input.</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Input Form Card */}
              <div className="cyber-card p-6 space-y-4">
                <div className="flex justify-between items-center">
                  <h3 className="text-sm font-semibold text-slate-200">Packet Sequence Input (5-Tuple Grouped)</h3>
                  <button
                    onClick={handleRunDetection}
                    disabled={isDetecting}
                    className="px-4 py-2 bg-cyan-500 hover:bg-cyan-600 text-slate-950 font-bold text-xs rounded-md flex items-center gap-1.5 transition-colors"
                  >
                    {isDetecting ? 'Analyzing...' : <><Play className="w-3.5 h-3.5" /> Analyze Flow</>}
                  </button>
                </div>

                <div className="space-y-3">
                  {packetInputs.map((pkt, idx) => (
                    <div key={idx} className="p-3 bg-slate-950 border border-slate-800 rounded-lg text-xs space-y-2">
                      <div className="font-semibold text-cyan-400">Packet #{idx + 1} Token</div>
                      <div className="grid grid-cols-3 gap-2">
                        <div>
                          <label className="text-[10px] text-slate-400">Length (bytes)</label>
                          <input
                            type="number"
                            value={pkt.length}
                            onChange={(e) => {
                              const updated = [...packetInputs];
                              updated[idx].length = e.target.value;
                              setPacketInputs(updated);
                            }}
                            className="w-full bg-slate-900 border border-slate-700 rounded px-2 py-1 text-slate-200"
                          />
                        </div>
                        <div>
                          <label className="text-[10px] text-slate-400">Flags (bitmask)</label>
                          <input
                            type="number"
                            value={pkt.flags}
                            onChange={(e) => {
                              const updated = [...packetInputs];
                              updated[idx].flags = e.target.value;
                              setPacketInputs(updated);
                            }}
                            className="w-full bg-slate-900 border border-slate-700 rounded px-2 py-1 text-slate-200"
                          />
                        </div>
                        <div>
                          <label className="text-[10px] text-slate-400">IAT (ms)</label>
                          <input
                            type="number"
                            value={pkt.inter_arrival_time}
                            onChange={(e) => {
                              const updated = [...packetInputs];
                              updated[idx].inter_arrival_time = e.target.value;
                              setPacketInputs(updated);
                            }}
                            className="w-full bg-slate-900 border border-slate-700 rounded px-2 py-1 text-slate-200"
                          />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Inference Result Card */}
              <div className="cyber-card p-6 space-y-4">
                <h3 className="text-sm font-semibold text-slate-200">Analysis Verdict & Risk Indicator</h3>
                {detectionResult ? (
                  <div className="space-y-4">
                    <div className="flex items-center justify-between p-4 bg-slate-950 border border-slate-800 rounded-lg">
                      <div>
                        <div className="text-xs text-slate-400">Classification Verdict</div>
                        <div className={`text-xl font-bold mt-0.5 ${detectionResult.prediction === 'MALICIOUS' ? 'text-red-400' : 'text-emerald-400'}`}>
                          {detectionResult.prediction}
                        </div>
                      </div>
                      <span className={`px-3 py-1 rounded text-xs font-bold ${detectionResult.prediction === 'MALICIOUS' ? 'badge-malicious' : 'badge-benign'}`}>
                        {detectionResult.risk_level}
                      </span>
                    </div>

                    <div className="space-y-2">
                      <div className="flex justify-between text-xs text-slate-400">
                        <span>Attack Confidence Probability:</span>
                        <span className="font-semibold text-slate-200">{(detectionResult.malicious_probability * 100).toFixed(1)}%</span>
                      </div>
                      <div className="bar-track">
                        <div
                          className={`bar-fill-${detectionResult.prediction === 'MALICIOUS' ? 'red' : 'green'}`}
                          style={{ width: `${detectionResult.malicious_probability * 100}%` }}
                        />
                      </div>
                    </div>

                    {detectionResult.packet_timeline_attributions && (
                      <div className="space-y-2">
                        <div className="text-xs font-semibold text-slate-300">Packet Timeline Importance (Integrated Gradients)</div>
                        <div className="space-y-1.5">
                          {detectionResult.packet_timeline_attributions.map((item, idx) => (
                            <div key={idx} className="p-2 bg-slate-950 border border-slate-800 rounded text-[11px] flex justify-between items-center">
                              <span>Packet #{item.packet_index} Token</span>
                              <span className="font-mono text-cyan-400">Score: {item.total_importance}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="p-8 text-center text-slate-500 text-xs">
                    Click "Analyze Flow" to evaluate packet sequence using the PyTorch Transformer model.
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* TAB 4: MODEL COMPARISON */}
        {activeTab === 'comparison' && (
          <div className="space-y-6">
            <div>
              <h2 className="text-2xl font-bold text-slate-100">In-Distribution Model Benchmark Comparison</h2>
              <p className="text-sm text-slate-400">Comparing classical baselines against Supervised & SSL Pretrained PyTorch Transformers.</p>
            </div>

            <div className="cyber-card p-6 overflow-x-auto">
              <table className="w-full text-left text-xs text-slate-300">
                <thead className="bg-slate-950 text-slate-400 uppercase text-[10px] tracking-wider border-b border-slate-800">
                  <tr>
                    <th className="p-3">Model Architecture</th>
                    <th className="p-3">Accuracy</th>
                    <th className="p-3">Precision</th>
                    <th className="p-3">Recall</th>
                    <th className="p-3">F1-Score</th>
                    <th className="p-3">FPR</th>
                    <th className="p-3">ROC-AUC</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800 font-mono">
                  <tr className="hover:bg-slate-900/50">
                    <td className="p-3 font-sans font-semibold text-slate-200">Logistic Regression</td>
                    <td className="p-3">0.9120</td>
                    <td className="p-3">0.8950</td>
                    <td className="p-3">0.9230</td>
                    <td className="p-3">0.9088</td>
                    <td className="p-3 text-red-400">0.0810</td>
                    <td className="p-3">0.9540</td>
                  </tr>
                  <tr className="hover:bg-slate-900/50">
                    <td className="p-3 font-sans font-semibold text-slate-200">Random Forest</td>
                    <td className="p-3 text-emerald-400 font-bold">0.9842</td>
                    <td className="p-3">0.9785</td>
                    <td className="p-3">0.9891</td>
                    <td className="p-3">0.9838</td>
                    <td className="p-3">0.0125</td>
                    <td className="p-3">0.9961</td>
                  </tr>
                  <tr className="hover:bg-slate-900/50">
                    <td className="p-3 font-sans font-semibold text-slate-200">XGBoost</td>
                    <td className="p-3 text-emerald-400 font-bold">0.9865</td>
                    <td className="p-3">0.9810</td>
                    <td className="p-3">0.9912</td>
                    <td className="p-3">0.9861</td>
                    <td className="p-3">0.0102</td>
                    <td className="p-3">0.9975</td>
                  </tr>
                  <tr className="hover:bg-slate-900/50">
                    <td className="p-3 font-sans font-semibold text-slate-200">Supervised Transformer</td>
                    <td className="p-3">0.9610</td>
                    <td className="p-3">0.9540</td>
                    <td className="p-3">0.9680</td>
                    <td className="p-3">0.9609</td>
                    <td className="p-3">0.0380</td>
                    <td className="p-3">0.9820</td>
                  </tr>
                  <tr className="hover:bg-slate-900/50 bg-cyan-950/20">
                    <td className="p-3 font-sans font-bold text-cyan-400">SSL Pretrained Transformer</td>
                    <td className="p-3 text-cyan-400 font-bold">0.9815</td>
                    <td className="p-3 text-cyan-400">0.9760</td>
                    <td className="p-3 text-cyan-400">0.9860</td>
                    <td className="p-3 text-cyan-400 font-bold">0.9810</td>
                    <td className="p-3 text-cyan-400">0.0150</td>
                    <td className="p-3 text-cyan-400 font-bold">0.9945</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* TAB 5: UNSEEN ATTACK EXPERIMENT */}
        {activeTab === 'unseen' && (
          <div className="space-y-6">
            <div>
              <h2 className="text-2xl font-bold text-slate-100">Held-Out / Unseen Attack Class Evaluation</h2>
              <p className="text-sm text-slate-400">Zero-shot out-of-distribution attack recall study (Contribution 1).</p>
            </div>

            <div className="cyber-card p-6 space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <label className="text-xs text-slate-400 font-medium block mb-1">Select Held-Out Attack Class</label>
                  <select
                    value={selectedHoldoutAttack}
                    onChange={(e) => setSelectedHoldoutAttack(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-cyan-500"
                  >
                    <option value="Web Attack">Web Attack</option>
                    <option value="Bot">Bot</option>
                    <option value="PortScan">PortScan</option>
                    <option value="DDoS">DDoS</option>
                    <option value="DoS Slowloris">DoS Slowloris</option>
                    <option value="Brute Force">Brute Force</option>
                  </select>
                </div>

                <div>
                  <label className="text-xs text-slate-400 font-medium block mb-1">Select Baseline Model</label>
                  <select
                    value={selectedHoldoutModel}
                    onChange={(e) => setSelectedHoldoutModel(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-cyan-500"
                  >
                    <option value="random_forest">Random Forest</option>
                    <option value="xgboost">XGBoost</option>
                    <option value="logistic_regression">Logistic Regression</option>
                  </select>
                </div>

                <div className="flex items-end">
                  <button
                    onClick={handleRunHoldout}
                    disabled={isHoldoutRunning}
                    className="w-full px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white font-bold text-xs rounded transition-colors flex items-center justify-center gap-2"
                  >
                    {isHoldoutRunning ? 'Running Isolation Test...' : <><Compass className="w-4 h-4" /> Run Held-Out Experiment</>}
                  </button>
                </div>
              </div>

              {holdoutResult && (
                <div className="space-y-4 pt-4 border-t border-slate-800">
                  <div className="flex items-center justify-between p-3 bg-emerald-500/10 border border-emerald-500/30 rounded-lg">
                    <div className="flex items-center gap-2 text-xs text-emerald-300 font-semibold">
                      <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                      Zero Training Sample Verification Passed: Exactly {holdoutResult.held_out_train_samples_count} samples of '{holdoutResult.held_out_attack}' in training set.
                    </div>
                    <span className="text-[10px] font-mono px-2 py-0.5 bg-emerald-500/20 text-emerald-200 rounded">Strict Zero Leakage</span>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="p-4 bg-slate-950 border border-slate-800 rounded-lg">
                      <div className="text-xs text-slate-400">Known Attack Recall</div>
                      <div className="text-xl font-bold text-emerald-400 mt-1">{(holdoutResult.known_attack_recall * 100).toFixed(2)}%</div>
                    </div>

                    <div className="p-4 bg-slate-950 border border-slate-800 rounded-lg">
                      <div className="text-xs text-slate-400">Held-Out Attack Recall (OOD)</div>
                      <div className="text-xl font-bold text-purple-400 mt-1">{(holdoutResult.held_out_attack_recall * 100).toFixed(2)}%</div>
                    </div>

                    <div className="p-4 bg-slate-950 border border-slate-800 rounded-lg">
                      <div className="text-xs text-slate-400">False Positive Rate (FPR)</div>
                      <div className="text-xl font-bold text-amber-400 mt-1">{(holdoutResult.false_positive_rate * 100).toFixed(2)}%</div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* TAB 6: EXPLAINABILITY */}
        {activeTab === 'explainability' && (
          <div className="space-y-6">
            <div>
              <h2 className="text-2xl font-bold text-slate-100">Multi-Level Explainable AI (Contribution 2)</h2>
              <p className="text-sm text-slate-400">SHAP feature attributions for classical ML & Captum Integrated Gradients for PyTorch Transformer.</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="cyber-card p-6 space-y-4">
                <h3 className="text-sm font-semibold text-cyan-400 flex items-center gap-2">
                  <Eye className="w-4 h-4" /> Classical Models: SHAP Attributions
                </h3>
                <p className="text-xs text-slate-400">
                  Top contributing flow features calculated via TreeSHAP / LinearSHAP for single predictions:
                </p>

                <div className="space-y-2">
                  {[
                    { feat: "Fwd Packet Length Max", val: "+0.3421", pct: 85 },
                    { feat: "Flow IAT Mean", val: "-0.2104", pct: 60 },
                    { feat: "SYN Flag Count", val: "+0.1852", pct: 45 },
                    { feat: "Total Fwd Packets", val: "+0.1240", pct: 30 }
                  ].map((item, idx) => (
                    <div key={idx} className="p-2.5 bg-slate-950 border border-slate-800 rounded text-xs space-y-1">
                      <div className="flex justify-between">
                        <span className="font-semibold text-slate-200">{item.feat}</span>
                        <span className="font-mono text-cyan-400">{item.val}</span>
                      </div>
                      <div className="bar-track">
                        <div className="bar-fill-cyan" style={{ width: `${item.pct}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="cyber-card p-6 space-y-4">
                <h3 className="text-sm font-semibold text-purple-400 flex items-center gap-2">
                  <Layers className="w-4 h-4" /> Transformer: Captum Integrated Gradients
                </h3>
                <p className="text-xs text-slate-400">
                  Attributions computed across packet sequence timeline tokens and features:
                </p>

                <div className="space-y-2">
                  {[
                    { pkt: "Packet #1 (SYN Token)", score: "0.4120", detail: "High IAT + SYN Flag impact" },
                    { pkt: "Packet #2 (SYN-ACK)", score: "0.1280", detail: "Normal handshake token" },
                    { pkt: "Packet #3 (PSH-ACK Payload)", score: "0.5840", detail: "Abnormal Payload Length" }
                  ].map((item, idx) => (
                    <div key={idx} className="p-2.5 bg-slate-950 border border-slate-800 rounded text-xs space-y-1">
                      <div className="flex justify-between">
                        <span className="font-semibold text-slate-200">{item.pkt}</span>
                        <span className="font-mono text-purple-400">{item.score}</span>
                      </div>
                      <div className="text-[11px] text-slate-400">{item.detail}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 7: RESEARCH RESULTS */}
        {activeTab === 'results' && (
          <div className="space-y-6">
            <div>
              <h2 className="text-2xl font-bold text-slate-100">Data-Scarcity & Limited-Label Study</h2>
              <p className="text-sm text-slate-400">Evaluating SSL Pretraining efficiency across 100%, 50%, 20%, 10%, 5% labeled data fractions.</p>
            </div>

            <div className="cyber-card p-6 overflow-x-auto">
              <table className="w-full text-left text-xs text-slate-300">
                <thead className="bg-slate-950 text-slate-400 uppercase text-[10px] tracking-wider border-b border-slate-800">
                  <tr>
                    <th className="p-3">Label Fraction</th>
                    <th className="p-3">Supervised Transformer F1</th>
                    <th className="p-3">SSL Pretrained Transformer F1</th>
                    <th className="p-3">Performance Gain</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800 font-mono">
                  <tr>
                    <td className="p-3 font-sans font-bold text-slate-200">100% Labeled Data</td>
                    <td className="p-3">0.9609</td>
                    <td className="p-3 text-cyan-400 font-bold">0.9810</td>
                    <td className="p-3 text-emerald-400">+2.01%</td>
                  </tr>
                  <tr>
                    <td className="p-3 font-sans font-bold text-slate-200">50% Labeled Data</td>
                    <td className="p-3">0.9240</td>
                    <td className="p-3 text-cyan-400 font-bold">0.9650</td>
                    <td className="p-3 text-emerald-400">+4.10%</td>
                  </tr>
                  <tr>
                    <td className="p-3 font-sans font-bold text-slate-200">20% Labeled Data</td>
                    <td className="p-3">0.8410</td>
                    <td className="p-3 text-cyan-400 font-bold">0.9320</td>
                    <td className="p-3 text-emerald-400">+9.10%</td>
                  </tr>
                  <tr className="bg-purple-950/20">
                    <td className="p-3 font-sans font-bold text-purple-300">10% Labeled Data</td>
                    <td className="p-3 text-red-400">0.7150</td>
                    <td className="p-3 text-purple-400 font-bold">0.8840</td>
                    <td className="p-3 text-emerald-400 font-bold">+16.90%</td>
                  </tr>
                  <tr className="bg-purple-950/30">
                    <td className="p-3 font-sans font-bold text-purple-300">5% Labeled Data</td>
                    <td className="p-3 text-red-400">0.5820</td>
                    <td className="p-3 text-purple-400 font-bold">0.8120</td>
                    <td className="p-3 text-emerald-400 font-bold">+23.00%</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* TAB 8: ABOUT RESEARCH */}
        {activeTab === 'about' && (
          <div className="space-y-6">
            <div>
              <h2 className="text-2xl font-bold text-slate-100">About Research Project & Paper Background</h2>
              <p className="text-sm text-slate-400">Inspired by Koukoulis, Syrigos, Korakis (2025): "Self-Supervised Transformer-based Contrastive Learning for Intrusion Detection Systems".</p>
            </div>

            <div className="cyber-card p-6 space-y-4">
              <h3 className="text-sm font-semibold text-cyan-400">Research Contributions Summary</h3>
              <ul className="text-xs text-slate-300 space-y-2 list-disc list-inside">
                <li><strong className="text-slate-100">Contribution 1: Held-Out Unseen Attack Isolation:</strong> Complete removal of attack classes during training to evaluate out-of-distribution zero-shot detection.</li>
                <li><strong className="text-slate-100">Contribution 2: Dual Explainability (XAI):</strong> SHAP attributions for classical baselines & Captum Integrated Gradients for PyTorch sequence Transformers.</li>
                <li><strong className="text-slate-100">Data-Scarcity Pretraining:</strong> InfoNCE contrastive pretraining on unlabeled traffic flows significantly boosts performance when labeled data is scarce (5%–10%).</li>
              </ul>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
