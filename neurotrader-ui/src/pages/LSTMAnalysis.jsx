import React, { useState, useEffect } from 'react';
import { 
  Cpu, 
  Layers, 
  Lock, 
  ArrowUpRight, 
  ArrowDownRight, 
  RefreshCw, 
  Sparkles,
  BarChart3,
  Hash
} from 'lucide-react';
import { 
  ResponsiveContainer, 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  Tooltip, 
  CartesianGrid, 
  Cell 
} from 'recharts';
import { api } from '../services/api';

const STOCKS = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC"];

export default function LSTMAnalysis() {
  const [stock, setStock] = useState("RELIANCE");
  const [step, setStep] = useState(0);
  const [predictionData, setPredictionData] = useState(null);
  const [embeddingData, setEmbeddingData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchLSTMData() {
      setLoading(true);
      try {
        const [pred, emb] = await Promise.all([
          api.getLSTMPrediction(stock, "test", step),
          api.getLSTMEmbedding(stock, "test", step)
        ]);
        setPredictionData(pred);
        setEmbeddingData(emb);
      } catch (err) {
        console.error("Failed to load LSTM analysis:", err);
      } finally {
        setLoading(false);
      }
    }
    fetchLSTMData();
  }, [stock, step]);

  const chartData = embeddingData ? embeddingData.values.map((val, idx) => ({
    name: `e${idx}`,
    dim: idx,
    value: val,
  })) : [];

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Control Strip */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Asset:</label>
          <div className="flex items-center gap-1.5">
            {STOCKS.map((sym) => (
              <button
                key={sym}
                onClick={() => setStock(sym)}
                className={`px-3 py-1.5 rounded-lg text-xs font-mono font-medium transition-all ${
                  stock === sym
                    ? 'bg-cyan-500 text-slate-950 font-bold shadow-md shadow-cyan-500/20'
                    : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                }`}
              >
                {sym}
              </button>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-3">
          <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Sequence Step:</label>
          <input
            type="range"
            min="0"
            max="150"
            value={step}
            onChange={(e) => setStep(parseInt(e.target.value))}
            className="w-40 accent-cyan-500 cursor-pointer"
          />
          <span className="font-mono text-xs bg-slate-800 px-2.5 py-1 rounded border border-slate-700 text-cyan-300">
            Step {step}
          </span>
        </div>
      </div>

      {loading ? (
        <div className="h-96 flex items-center justify-center bg-slate-900/60 rounded-xl border border-slate-800">
          <div className="flex items-center gap-3 text-cyan-400 text-sm">
            <RefreshCw className="w-5 h-5 animate-spin" />
            Running frozen LSTM encoder on 30-day sequence...
          </div>
        </div>
      ) : predictionData && embeddingData ? (
        <>
          {/* Top Prediction & Architecture Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Predicted Direction */}
            <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-6 relative overflow-hidden">
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Forward 1-Day Return</span>
                <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-mono bg-slate-800 text-slate-300">
                  Target: {predictionData.target_date}
                </span>
              </div>
              <div className="flex items-baseline gap-3 my-2">
                <span className={`text-3xl font-extrabold font-mono ${
                  predictionData.prediction.predicted_direction === 'UP' ? 'text-emerald-400' : 'text-rose-400'
                }`}>
                  {predictionData.prediction.predicted_next_day_return > 0 ? '+' : ''}
                  {(predictionData.prediction.predicted_next_day_return * 100).toFixed(4)}%
                </span>
                <span className={`inline-flex items-center text-xs font-bold px-2 py-0.5 rounded ${
                  predictionData.prediction.predicted_direction === 'UP' 
                    ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' 
                    : 'bg-rose-500/10 text-rose-400 border border-rose-500/20'
                }`}>
                  {predictionData.prediction.predicted_direction === 'UP' ? (
                    <><ArrowUpRight className="w-3.5 h-3.5 mr-1" /> BUY BIAS</>
                  ) : (
                    <><ArrowDownRight className="w-3.5 h-3.5 mr-1" /> SELL BIAS</>
                  )}
                </span>
              </div>
              <p className="text-xs text-slate-500 mt-2">
                Supervised regression prediction from 32-D bottleneck layer
              </p>
            </div>

            {/* Encoder Specification */}
            <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-6">
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Encoder Checkpoint</span>
                <span className="inline-flex items-center gap-1 text-[10px] font-mono text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
                  <Lock className="w-3 h-3" /> Frozen Weights
                </span>
              </div>
              <div className="text-sm font-semibold text-slate-200 mt-1">
                improved_rnn_encoder.pt
              </div>
              <div className="mt-3 space-y-1.5 text-xs text-slate-400 font-mono">
                <div>Architecture: 2-Layer LSTM (hidden=64)</div>
                <div>Embedding Dimension: 32 Continuous</div>
                <div>Directional Accuracy: 55.53% (Held-out Test)</div>
              </div>
            </div>

            {/* Embedding Vector Stats */}
            <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-6">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">32-D Latent Vector Stats</span>
              <div className="grid grid-cols-2 gap-3 mt-3 text-xs font-mono">
                <div className="p-2.5 rounded bg-slate-950/60 border border-slate-800">
                  <div className="text-slate-500 text-[10px]">L2 Norm</div>
                  <div className="text-cyan-300 font-bold text-sm mt-0.5">{embeddingData.statistics.norm}</div>
                </div>
                <div className="p-2.5 rounded bg-slate-950/60 border border-slate-800">
                  <div className="text-slate-500 text-[10px]">Vector Mean</div>
                  <div className="text-slate-200 font-bold text-sm mt-0.5">{embeddingData.statistics.mean}</div>
                </div>
                <div className="p-2.5 rounded bg-slate-950/60 border border-slate-800">
                  <div className="text-slate-500 text-[10px]">Vector Min</div>
                  <div className="text-slate-200 font-bold text-sm mt-0.5">{embeddingData.statistics.min}</div>
                </div>
                <div className="p-2.5 rounded bg-slate-950/60 border border-slate-800">
                  <div className="text-slate-500 text-[10px]">Vector Max</div>
                  <div className="text-slate-200 font-bold text-sm mt-0.5">{embeddingData.statistics.max}</div>
                </div>
              </div>
            </div>
          </div>

          {/* 32-D Embedding Vector Visualizer */}
          <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-base font-semibold text-slate-200 flex items-center gap-2">
                  <BarChart3 className="w-4 h-4 text-cyan-400" />
                  32-Dimensional Latent Market State Embedding Vector (e₀ ... e₃₁)
                </h3>
                <p className="text-xs text-slate-500">
                  Extracted from the last timestep hidden state of the frozen LSTM encoder. Provided to PPO as features [0:32].
                </p>
              </div>
            </div>

            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                  <XAxis dataKey="name" stroke="#64748b" fontSize={10} tickLine={false} />
                  <YAxis stroke="#64748b" fontSize={10} tickLine={false} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#090d16', borderColor: '#334155', borderRadius: '8px' }}
                    formatter={(val) => [val.toFixed(4), 'Latent Value']}
                  />
                  <Bar dataKey="value" radius={[3, 3, 0, 0]}>
                    {chartData.map((entry, index) => (
                      <Cell 
                        key={`cell-${index}`} 
                        fill={entry.value >= 0 ? '#06b6d4' : '#f43f5e'} 
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Discrete Value Chips */}
            <div className="mt-4 pt-4 border-t border-slate-800">
              <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block mb-2">
                Raw 32-D Vector Array:
              </span>
              <div className="grid grid-cols-4 sm:grid-cols-8 gap-1.5 font-mono text-[10px]">
                {embeddingData.values.map((v, i) => (
                  <div key={i} className="p-1.5 rounded bg-slate-950 border border-slate-800/80 text-center">
                    <span className="text-slate-500 text-[9px] block">e{i}</span>
                    <span className={v >= 0 ? 'text-cyan-400' : 'text-rose-400'}>{v.toFixed(3)}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </>
      ) : null}
    </div>
  );
}
