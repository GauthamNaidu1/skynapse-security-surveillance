import React, { useState, useEffect } from 'react';
import { BrainCircuit, Zap, ShieldCheck, AlertOctagon, Bot } from 'lucide-react';
import { motion } from 'motion/react';

const LOCATIONS = [
  "Hitech City", "Charminar", "Banjara Hills", "Secunderabad", "Gachibowli",
  "Jubilee Hills", "Ameerpet", "LB Nagar", "Madhapur", "Begumpet",
  "Kukatpally", "Dilsukhnagar", "Mehdipatnam", "Uppal", "Miyapur",
  "Kondapur", "Tarnaka", "Abids", "Nampally", "Malakpet"
];

export default function Predictor() {
  const [inputs, setInputs] = useState({
    hour: 12,
    density: 50,
    location_id: 0
  });
  const [prediction, setPrediction] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchPrediction = async () => {
      setLoading(true);
      try {
        const res = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(inputs)
        });
        const data = await res.json();
        setPrediction(data);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    const timer = setTimeout(fetchPrediction, 150);
    return () => clearTimeout(timer);
  }, [inputs]);

  return (
    <div className="p-6 space-y-6 h-full overflow-y-auto bg-slate-950 text-slate-200">
      <header className="mb-8 border-b border-slate-800 pb-4">
        <h2 className="text-2xl font-bold text-white flex items-center gap-2">
          <BrainCircuit className="text-purple-500" />
          AI Threat Predictor
        </h2>
        <p className="text-slate-500 text-sm">Ensemble Model Simulation & Tactical AI</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Input Panel */}
        <div className="bg-slate-900/50 p-6 rounded-xl border border-slate-800 space-y-6">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold text-slate-300">Scenario Parameters</h3>
            {loading ? (
              <div className="text-xs text-purple-400 animate-pulse flex items-center gap-1"><Zap className="w-3 h-3" /> UPDATING...</div>
            ) : (
              <div className="text-xs text-green-500 flex items-center gap-1 opacity-80"><Zap className="w-3 h-3" /> LIVE READY</div>
            )}
          </div>
          
          <div className="space-y-2">
            <label className="text-sm text-slate-400">Location</label>
            <select
              value={inputs.location_id}
              onChange={(e) => setInputs({ ...inputs, location_id: parseInt(e.target.value) })}
              className="w-full bg-slate-950 border border-slate-700 rounded-lg p-3 text-white focus:ring-2 focus:ring-purple-500 outline-none"
            >
              {LOCATIONS.map((loc, idx) => (
                <option key={idx} value={idx}>{loc}</option>
              ))}
            </select>
          </div>

          <div className="space-y-2">
            <div className="flex justify-between">
              <label className="text-sm text-slate-400">Hour of Day (0-23)</label>
              <span className="text-sm font-mono text-purple-400">{inputs.hour}:00</span>
            </div>
            <input
              type="range"
              min="0"
              max="23"
              value={inputs.hour}
              onChange={(e) => setInputs({ ...inputs, hour: parseInt(e.target.value) })}
              className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-purple-500"
            />
          </div>

          <div className="space-y-2">
            <div className="flex justify-between">
              <label className="text-sm text-slate-400">Crowd Density (%)</label>
              <span className="text-sm font-mono text-purple-400">{inputs.density}%</span>
            </div>
            <input
              type="range"
              min="0"
              max="100"
              value={inputs.density}
              onChange={(e) => setInputs({ ...inputs, density: parseInt(e.target.value) })}
              className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-purple-500"
            />
          </div>
          
          <div className="p-4 bg-purple-900/10 rounded-lg border border-purple-500/20 text-xs text-purple-300">
            Adjust sliders to see real-time threat prediction updates.
          </div>
        </div>

        {/* Results Panel */}
        <div className="space-y-6">
          {prediction && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-slate-900/50 p-6 rounded-xl border border-slate-800 relative overflow-hidden"
            >
              <div className="absolute top-0 right-0 p-4 opacity-10">
                <BrainCircuit className="w-32 h-32" />
              </div>
              
              <h3 className="text-lg font-semibold text-slate-300 mb-6">Model Output</h3>
              
              <div className="grid grid-cols-2 gap-4 mb-6">
                <div className="bg-slate-950 p-4 rounded-lg border border-slate-800">
                  <div className="text-xs text-slate-500 uppercase">Threat Level</div>
                  <div className={`text-2xl font-bold mt-1 ${
                    ['text-green-500', 'text-yellow-500', 'text-orange-500', 'text-red-500'][prediction.threat_level]
                  }`}>
                    {['Low', 'Medium', 'High', 'Critical'][prediction.threat_level]}
                  </div>
                </div>
                <div className="bg-slate-950 p-4 rounded-lg border border-slate-800">
                  <div className="text-xs text-slate-500 uppercase">Confidence</div>
                  <div className="text-2xl font-bold text-white mt-1">
                    {(prediction.confidence * 100).toFixed(1)}%
                  </div>
                </div>
              </div>

              <div className="space-y-2 mb-6">
                <div className="text-xs text-slate-500 uppercase mb-1">Probability Distribution</div>
                <div className="flex h-4 rounded-full overflow-hidden bg-slate-800">
                  {prediction.probabilities.map((prob: number, idx: number) => (
                    <div
                      key={idx}
                      style={{ width: `${prob * 100}%` }}
                      className={`${['bg-green-500', 'bg-yellow-500', 'bg-orange-500', 'bg-red-500'][idx]} h-full`}
                      title={['Low', 'Medium', 'High', 'Critical'][idx]}
                    />
                  ))}
                </div>
                <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                  <span>LOW</span>
                  <span>MED</span>
                  <span>HIGH</span>
                  <span>CRIT</span>
                </div>
              </div>
            </motion.div>
          )}
        </div>
      </div>
    </div>
  );
}
