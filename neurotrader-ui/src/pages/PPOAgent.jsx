import React, { useState, useEffect } from 'react';
import { 
  Brain, 
  Layers, 
  Sliders, 
  PieChart, 
  CheckCircle2, 
  AlertCircle,
  HelpCircle,
  Activity,
  RefreshCw,
  Wallet
} from 'lucide-react';
import { api } from '../services/api';

const STOCKS = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC"];

export default function PPOAgent() {
  const [stock, setStock] = useState("RELIANCE");
  const [step, setStep] = useState(0);
  const [inPosition, setInPosition] = useState(false);
  const [portfolioValue, setPortfolioValue] = useState(100000.0);
  const [decision, setDecision] = useState(null);
  const [loading, setLoading] = useState(true);

  async function evaluateDecision() {
    setLoading(true);
    try {
      const res = await api.getPPOdecision({
        stock,
        split: "test",
        step,
        in_position: inPosition,
        portfolio_value: portfolioValue,
        initial_capital: 100000.0
      });
      setDecision(res);
    } catch (err) {
      console.error("Failed to fetch PPO decision:", err);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    evaluateDecision();
  }, [stock, step, inPosition, portfolioValue]);

  const actionColors = {
    HOLD: "text-amber-400 bg-amber-500/10 border-amber-500/30",
    BUY: "text-emerald-400 bg-emerald-500/10 border-emerald-500/30",
    SELL: "text-rose-400 bg-rose-500/10 border-rose-500/30",
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Control Strip */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-4">
          Interactive PPO State Constructor
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
          <div>
            <label className="text-xs text-slate-400 block mb-1.5">Asset:</label>
            <select
              value={stock}
              onChange={(e) => setStock(e.target.value)}
              className="w-full bg-slate-950 border border-slate-700 rounded-lg p-2 text-xs font-mono text-slate-200"
            >
              {STOCKS.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>

          <div>
            <label className="text-xs text-slate-400 block mb-1.5">Market Step (0 - 150):</label>
            <div className="flex items-center gap-2">
              <input
                type="range"
                min="0"
                max="150"
                value={step}
                onChange={(e) => setStep(parseInt(e.target.value))}
                className="w-full accent-cyan-500 cursor-pointer"
              />
              <span className="font-mono text-xs text-cyan-300 w-12">{step}</span>
            </div>
          </div>

          <div>
            <label className="text-xs text-slate-400 block mb-1.5">Agent Portfolio Holding:</label>
            <div className="flex rounded-lg bg-slate-950 p-1 border border-slate-800">
              <button
                onClick={() => setInPosition(false)}
                className={`flex-1 py-1 rounded text-xs font-mono transition-all ${
                  !inPosition ? 'bg-cyan-500 text-slate-950 font-bold' : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                100% CASH
              </button>
              <button
                onClick={() => setInPosition(true)}
                className={`flex-1 py-1 rounded text-xs font-mono transition-all ${
                  inPosition ? 'bg-emerald-500 text-slate-950 font-bold' : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                IN-POSITION
              </button>
            </div>
          </div>

          <div>
            <label className="text-xs text-slate-400 block mb-1.5">Portfolio Value (₹):</label>
            <input
              type="number"
              value={portfolioValue}
              onChange={(e) => setPortfolioValue(parseFloat(e.target.value) || 100000.0)}
              className="w-full bg-slate-950 border border-slate-700 rounded-lg p-2 text-xs font-mono text-slate-200"
            />
          </div>
        </div>
      </div>

      {loading ? (
        <div className="h-96 flex items-center justify-center bg-slate-900/60 rounded-xl border border-slate-800">
          <div className="flex items-center gap-3 text-cyan-400 text-sm">
            <RefreshCw className="w-5 h-5 animate-spin" />
            Computing 49-D hybrid state and running PPO Actor-Critic policy...
          </div>
        </div>
      ) : decision ? (
        <>
          {/* Policy Decision & Critic Output */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Selected Action Card */}
            <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-6 relative overflow-hidden">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block mb-2">
                PPO Actor Selected Action
              </span>
              <div className="flex items-center gap-4 my-2">
                <span className={`px-4 py-2 rounded-xl text-2xl font-black font-mono border ${actionColors[decision.action_name]}`}>
                  {decision.action_name}
                </span>
                <div>
                  <span className="text-xs text-slate-400 block">Selected Action ID</span>
                  <span className="text-sm font-mono text-slate-200 font-bold">Discrete({decision.selected_action})</span>
                </div>
              </div>
              <p className="text-xs text-slate-400 mt-4 leading-relaxed bg-slate-950/60 p-3 rounded-lg border border-slate-800/80">
                {decision.explanation}
              </p>
            </div>

            {/* Action Probabilities Breakdown */}
            <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-6">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block mb-3">
                Softmax Action Probabilities
              </span>
              <div className="space-y-3 font-mono text-xs">
                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-amber-400 font-semibold">0: HOLD</span>
                    <span>{(decision.probabilities.HOLD * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-slate-950 h-2.5 rounded-full overflow-hidden border border-slate-800">
                    <div className="bg-amber-400 h-full rounded-full" style={{ width: `${decision.probabilities.HOLD * 100}%` }}></div>
                  </div>
                </div>

                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-emerald-400 font-semibold">1: BUY</span>
                    <span>{(decision.probabilities.BUY * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-slate-950 h-2.5 rounded-full overflow-hidden border border-slate-800">
                    <div className="bg-emerald-400 h-full rounded-full" style={{ width: `${decision.probabilities.BUY * 100}%` }}></div>
                  </div>
                </div>

                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-rose-400 font-semibold">2: SELL</span>
                    <span>{(decision.probabilities.SELL * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-slate-950 h-2.5 rounded-full overflow-hidden border border-slate-800">
                    <div className="bg-rose-400 h-full rounded-full" style={{ width: `${decision.probabilities.SELL * 100}%` }}></div>
                  </div>
                </div>
              </div>
            </div>

            {/* Critic Value Estimate V(s) */}
            <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-6 flex flex-col justify-between">
              <div>
                <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block mb-2">
                  Critic State-Value Estimate V(s)
                </span>
                <div className="text-3xl font-extrabold font-mono text-cyan-400 my-2">
                  {decision.critic_value > 0 ? '+' : ''}{decision.critic_value.toFixed(4)}
                </div>
                <p className="text-xs text-slate-400 mt-2 leading-relaxed">
                  The decoupled critic head estimates expected cumulative discounted return from state s_t under policy π.
                </p>
              </div>
              <div className="pt-3 border-t border-slate-800 text-[11px] font-mono text-slate-400">
                Network: Linear(49→64→64→1) • LayerNorm
              </div>
            </div>
          </div>

          {/* 49-D Observation Decomposition */}
          <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-6">
            <h3 className="text-base font-semibold text-slate-200 mb-4 flex items-center gap-2">
              <Layers className="w-4 h-4 text-cyan-400" />
              49-Dimensional Hybrid Observation Decomposition
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
              {/* Component 1 */}
              <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-semibold text-cyan-400">[0 : 32] LSTM State Embedding</span>
                  <span className="font-mono text-[10px] bg-cyan-500/10 text-cyan-300 px-2 py-0.5 rounded">32 dims</span>
                </div>
                <p className="text-slate-400 text-xs mb-3">
                  Latent compression of historical 30-day multivariate time series extracted from frozen encoder.
                </p>
                <div className="text-[11px] font-mono text-slate-400">
                  Status: Low-noise, stationary state vector
                </div>
              </div>

              {/* Component 2 */}
              <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-semibold text-emerald-400">[32 : 46] Current Technical Features</span>
                  <span className="font-mono text-[10px] bg-emerald-500/10 text-emerald-300 px-2 py-0.5 rounded">14 dims</span>
                </div>
                <p className="text-slate-400 text-xs mb-3">
                  Stationary technical indicators at close of day t (RSI, MACD, Bollinger Bands, ATR, ROC, SMA/EMA).
                </p>
                <div className="text-[11px] font-mono text-slate-400">
                  Status: Explicit momentum & volatility signals
                </div>
              </div>

              {/* Component 3 */}
              <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-semibold text-amber-400">[46 : 49] Portfolio Internal State</span>
                  <span className="font-mono text-[10px] bg-amber-500/10 text-amber-300 px-2 py-0.5 rounded">3 dims</span>
                </div>
                <div className="space-y-1.5 font-mono text-[11px] text-slate-300">
                  <div className="flex justify-between">
                    <span>Position Fraction:</span>
                    <span className="text-amber-400">{decision.observation_49d.portfolio_variables.position_fraction}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Cash Ratio:</span>
                    <span className="text-amber-400">{decision.observation_49d.portfolio_variables.cash_ratio}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Norm Portfolio Val:</span>
                    <span className="text-amber-400">{decision.observation_49d.portfolio_variables.normalized_value}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </>
      ) : null}
    </div>
  );
}
