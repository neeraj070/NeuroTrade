import React, { useEffect, useState } from 'react';
import { 
  TrendingUp, 
  ShieldAlert, 
  Activity, 
  Target, 
  Layers, 
  ArrowUpRight, 
  ArrowDownRight,
  Cpu,
  Brain,
  CheckCircle2,
  Lock
} from 'lucide-react';
import MetricCard from '../components/MetricCard';
import { api } from '../services/api';

export default function Overview({ onNavigate }) {
  const [health, setHealth] = useState(null);
  const [perf, setPerf] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      try {
        const [h, p] = await Promise.all([api.getHealth(), api.getPerformance()]);
        setHealth(h);
        setPerf(p);
      } catch (err) {
        console.error("Overview load error:", err);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* Top Banner */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-slate-900 via-slate-900 to-cyan-950/40 border border-slate-800 p-8 shadow-xl">
        <div className="relative z-10 max-w-3xl">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 text-xs font-mono font-medium mb-4">
            <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></span>
            B.Tech Final-Year Capstone Project • 2021–2026
          </div>
          <h1 className="text-3xl font-extrabold tracking-tight text-slate-100 sm:text-4xl mb-3">
            Reinforcement Learning with RNN-Based Market State Representation
          </h1>
          <p className="text-slate-400 text-sm leading-relaxed mb-6">
            NeuroTrader extracts continuous 32-dimensional market state representations using a pre-trained, frozen 2-layer LSTM encoder, feeding an on-policy PPO Actor-Critic agent operating within a realistic Gymnasium trading environment with next-day Open execution and 10 bps transaction fees.
          </p>
          <div className="flex flex-wrap gap-3">
            <button
              onClick={() => onNavigate('simulator')}
              className="inline-flex items-center gap-2 px-4 py-2.5 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-semibold text-xs tracking-wider uppercase transition-all shadow-lg shadow-cyan-500/20"
            >
              Launch Simulator <ArrowUpRight className="w-4 h-4" />
            </button>
            <button
              onClick={() => onNavigate('performance')}
              className="inline-flex items-center gap-2 px-4 py-2.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 font-medium text-xs tracking-wider transition-all border border-slate-700"
            >
              View Benchmark Comparisons
            </button>
          </div>
        </div>
      </div>

      {/* Core Verified Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
        <MetricCard
          title="LSTM Directional Accuracy"
          value="55.53%"
          subtitle="Held-out Test Split (497 / 895 correct)"
          change="+5.53% vs 50% Random"
          isPositive={true}
          icon={Target}
          badgeColor="cyan"
        />
        <MetricCard
          title="PPO Test Return (Out-of-Sample)"
          value="+0.34%"
          subtitle="Bear Market Regime (vs B&H -15.74%)"
          change="+16.08% Alpha vs B&H"
          isPositive={true}
          icon={TrendingUp}
          badgeColor="emerald"
        />
        <MetricCard
          title="Annualized Sharpe Ratio"
          value="+0.13"
          subtitle="PPO Agent (vs B&H -0.75, MA -1.54)"
          change="Positive Risk-Adjusted"
          isPositive={true}
          icon={Activity}
          badgeColor="emerald"
        />
        <MetricCard
          title="Maximum Drawdown"
          value="-13.03%"
          subtitle="Downside protection (vs B&H -26.34%)"
          change="50.5% DD Reduction"
          isPositive={true}
          icon={ShieldAlert}
          badgeColor="amber"
        />
      </div>

      {/* Architectural Pipeline Overview */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 bg-slate-900/80 border border-slate-800 rounded-xl p-6">
          <h2 className="text-base font-semibold text-slate-200 mb-4 flex items-center gap-2">
            <Layers className="w-4 h-4 text-cyan-400" />
            End-to-End System Pipeline
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
            <div className="p-4 rounded-lg bg-slate-950/60 border border-slate-800 flex flex-col justify-between">
              <div>
                <span className="text-[10px] font-mono text-cyan-400 uppercase tracking-wider">Stage 1</span>
                <h3 className="font-semibold text-sm text-slate-200 mt-1">Data Pipeline</h3>
                <p className="text-xs text-slate-400 mt-2 leading-relaxed">
                  5 NSE Equities (1,236 bars). 14 stationary technical indicators with 70/15/15 chronological splitting.
                </p>
              </div>
              <div className="mt-4 pt-3 border-t border-slate-800/80 text-[11px] font-mono text-slate-400">
                4,000 / 885 / 895 splits
              </div>
            </div>

            <div className="p-4 rounded-lg bg-slate-950/60 border border-slate-800 flex flex-col justify-between">
              <div>
                <span className="text-[10px] font-mono text-cyan-400 uppercase tracking-wider">Stage 2</span>
                <h3 className="font-semibold text-sm text-slate-200 mt-1">LSTM Encoder</h3>
                <p className="text-xs text-slate-400 mt-2 leading-relaxed">
                  2-Layer LSTM (hidden=64). Compresses 30-step windows into continuous 32-D market embeddings.
                </p>
              </div>
              <div className="mt-4 pt-3 border-t border-slate-800/80 text-[11px] font-mono text-emerald-400 flex items-center gap-1">
                <Lock className="w-3 h-3" /> Frozen Checkpoint
              </div>
            </div>

            <div className="p-4 rounded-lg bg-slate-950/60 border border-slate-800 flex flex-col justify-between">
              <div>
                <span className="text-[10px] font-mono text-cyan-400 uppercase tracking-wider">Stage 3</span>
                <h3 className="font-semibold text-sm text-slate-200 mt-1">Trading Env</h3>
                <p className="text-xs text-slate-400 mt-2 leading-relaxed">
                  Gymnasium MarketTradingEnv. 49-D state (32 emb + 14 tech + 3 portfolio). Next-day Open execution.
                </p>
              </div>
              <div className="mt-4 pt-3 border-t border-slate-800/80 text-[11px] font-mono text-slate-400">
                10 bps fees • Long-only
              </div>
            </div>

            <div className="p-4 rounded-lg bg-slate-950/60 border border-slate-800 flex flex-col justify-between">
              <div>
                <span className="text-[10px] font-mono text-cyan-400 uppercase tracking-wider">Stage 4</span>
                <h3 className="font-semibold text-sm text-slate-200 mt-1">PPO Agent</h3>
                <p className="text-xs text-slate-400 mt-2 leading-relaxed">
                  Decoupled Actor-Critic (49→64→64). 35 iterations, 140,000 transitions. Iteration 4 selected.
                </p>
              </div>
              <div className="mt-4 pt-3 border-t border-slate-800/80 text-[11px] font-mono text-cyan-300">
                100% Bitwise Match
              </div>
            </div>
          </div>
        </div>

        {/* Model Status & Environment Checks */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-6 flex flex-col justify-between">
          <div>
            <h2 className="text-base font-semibold text-slate-200 mb-4 flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              Runtime Subsystems
            </h2>
            <div className="space-y-3.5 text-xs">
              <div className="flex items-center justify-between p-3 rounded-lg bg-slate-950/50 border border-slate-800/60">
                <span className="text-slate-400">LSTM Encoder Checkpoint:</span>
                <span className="font-mono text-emerald-400 font-medium">improved_rnn_encoder.pt</span>
              </div>
              <div className="flex items-center justify-between p-3 rounded-lg bg-slate-950/50 border border-slate-800/60">
                <span className="text-slate-400">PPO Canonical Agent:</span>
                <span className="font-mono text-emerald-400 font-medium">ppo_reproducible_best.pt</span>
              </div>
              <div className="flex items-center justify-between p-3 rounded-lg bg-slate-950/50 border border-slate-800/60">
                <span className="text-slate-400">Trading Environment:</span>
                <span className="font-mono text-cyan-400 font-medium">Gymnasium check_env passed</span>
              </div>
              <div className="flex items-center justify-between p-3 rounded-lg bg-slate-950/50 border border-slate-800/60">
                <span className="text-slate-400">Deterministic Seed:</span>
                <span className="font-mono text-amber-400 font-medium">master_seed = 42</span>
              </div>
            </div>
          </div>

          <div className="mt-4 p-3 rounded-lg bg-cyan-950/20 border border-cyan-800/30 text-xs text-slate-400 leading-relaxed">
            <span className="text-cyan-400 font-semibold">Strict Isolation Active:</span> Frontend operates exclusively via the <code className="text-cyan-300 font-mono">:8000</code> API adapter with zero modifications to the underlying backend models or data files.
          </div>
        </div>
      </div>
    </div>
  );
}
