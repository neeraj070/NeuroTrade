import React, { useState } from 'react';
import { 
  Play, 
  RotateCcw, 
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  Receipt, 
  Layers, 
  ShieldCheck, 
  RefreshCw,
  Clock,
  ArrowUpRight,
  ArrowDownRight
} from 'lucide-react';
import { 
  ResponsiveContainer, 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  Tooltip, 
  CartesianGrid, 
  Legend 
} from 'recharts';
import { api } from '../services/api';

const STOCKS = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC"];

export default function TradingSimulator() {
  const [stock, setStock] = useState("RELIANCE");
  const [split, setSplit] = useState("test");
  const [capital, setCapital] = useState(100000.0);
  const [feeBps, setFeeBps] = useState(10); // 10 bps = 0.001
  const [simulationResult, setSimulationResult] = useState(null);
  const [running, setRunning] = useState(false);

  async function handleRunSimulation() {
    setRunning(true);
    try {
      const res = await api.runSimulation({
        stock,
        split,
        initial_capital: parseFloat(capital) || 100000.0,
        transaction_fee: feeBps / 10000.0
      });
      setSimulationResult(res);
    } catch (err) {
      console.error("Simulation run failed:", err);
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Simulation Configuration Card */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-base font-semibold text-slate-200">
              Interactive Backtest Trading Simulator
            </h3>
            <p className="text-xs text-slate-500">
              Executes isolated MarketTradingEnv instance using the canonical PPO agent policy (Iteration 4).
            </p>
          </div>
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-cyan-500/10 text-cyan-400 text-xs font-mono">
            <ShieldCheck className="w-3.5 h-3.5 text-cyan-400" /> Isolated Instance • Zero Live Trades
          </div>
        </div>

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
            <label className="text-xs text-slate-400 block mb-1.5">Evaluation Split:</label>
            <select
              value={split}
              onChange={(e) => setSplit(e.target.value)}
              className="w-full bg-slate-950 border border-slate-700 rounded-lg p-2 text-xs font-mono text-slate-200"
            >
              <option value="test">Held-out Test (Out-of-Sample)</option>
              <option value="val">Validation Split</option>
              <option value="train">Training Split</option>
            </select>
          </div>

          <div>
            <label className="text-xs text-slate-400 block mb-1.5">Initial Capital (₹):</label>
            <input
              type="number"
              value={capital}
              onChange={(e) => setCapital(e.target.value)}
              className="w-full bg-slate-950 border border-slate-700 rounded-lg p-2 text-xs font-mono text-slate-200"
            />
          </div>

          <div>
            <label className="text-xs text-slate-400 block mb-1.5">Transaction Fee (basis points):</label>
            <div className="flex items-center gap-2">
              <input
                type="number"
                value={feeBps}
                onChange={(e) => setFeeBps(parseInt(e.target.value) || 10)}
                className="w-full bg-slate-950 border border-slate-700 rounded-lg p-2 text-xs font-mono text-slate-200"
              />
              <span className="text-xs font-mono text-slate-400 whitespace-nowrap">10 bps = 0.1%</span>
            </div>
          </div>
        </div>

        <div className="mt-5 flex justify-end">
          <button
            onClick={handleRunSimulation}
            disabled={running}
            className="inline-flex items-center gap-2 px-6 py-2.5 rounded-lg bg-cyan-500 hover:bg-cyan-400 disabled:opacity-50 text-slate-950 font-bold text-xs uppercase tracking-wider transition-all shadow-lg shadow-cyan-500/20"
          >
            {running ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                Executing Session...
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-current" />
                Run Simulation
              </>
            )}
          </button>
        </div>
      </div>

      {simulationResult && (
        <>
          {/* Key Simulation Outcome Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
            <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5">
              <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">Final Portfolio Value</span>
              <div className="text-2xl font-bold font-mono text-slate-100 mt-2">
                ₹{simulationResult.final_value.toLocaleString()}
              </div>
              <div className={`text-xs font-semibold mt-1 ${
                simulationResult.total_return_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'
              }`}>
                {simulationResult.total_return_pct >= 0 ? '+' : ''}{simulationResult.total_return_pct.toFixed(2)}% Return
              </div>
            </div>

            <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5">
              <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">Buy & Hold Return</span>
              <div className="text-2xl font-bold font-mono text-slate-100 mt-2">
                {simulationResult.bh_return_pct >= 0 ? '+' : ''}{simulationResult.bh_return_pct.toFixed(2)}%
              </div>
              <div className={`text-xs font-semibold mt-1 ${
                (simulationResult.total_return_pct - simulationResult.bh_return_pct) >= 0 ? 'text-emerald-400' : 'text-rose-400'
              }`}>
                {(simulationResult.total_return_pct - simulationResult.bh_return_pct) >= 0 ? '+' : ''}
                {(simulationResult.total_return_pct - simulationResult.bh_return_pct).toFixed(2)}% Alpha vs B&H
              </div>
            </div>

            <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5">
              <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">Max Drawdown</span>
              <div className="text-2xl font-bold font-mono text-amber-400 mt-2">
                {simulationResult.max_drawdown_pct.toFixed(2)}%
              </div>
              <span className="text-xs text-slate-500 mt-1 block">Peak-to-trough capital decline</span>
            </div>

            <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5">
              <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">Trading Activity & Friction</span>
              <div className="text-2xl font-bold font-mono text-slate-100 mt-2">
                {simulationResult.total_trades} Trades
              </div>
              <span className="text-xs text-rose-400 mt-1 block">
                ₹{simulationResult.total_friction.toLocaleString()} in 10 bps fees
              </span>
            </div>
          </div>

          {/* Comparative Equity Curve Chart */}
          <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-base font-semibold text-slate-200">
                  Simulation Equity Curves ({stock} • {split.toUpperCase()})
                </h3>
                <p className="text-xs text-slate-500">
                  Comparing PPO dynamic policy vs Buy-and-Hold benchmark over the simulated period.
                </p>
              </div>
            </div>

            <div className="h-72 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={simulationResult.equity_curve} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                  <XAxis dataKey="step" stroke="#64748b" fontSize={11} tickLine={false} label={{ value: "Trading Days", position: "insideBottomRight", offset: -5, fill: "#64748b", fontSize: 10 }} />
                  <YAxis stroke="#64748b" fontSize={11} tickLine={false} tickFormatter={(v) => `₹${(v/1000).toFixed(0)}k`} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#090d16', borderColor: '#334155', borderRadius: '8px' }}
                    formatter={(val) => [`₹${val.toLocaleString()}`]}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="ppo_val" name="PPO Agent Policy" stroke="#10b981" strokeWidth={2.5} dot={false} />
                  <Line type="monotone" dataKey="bh_val" name="Buy & Hold" stroke="#3b82f6" strokeWidth={1.5} strokeDasharray="4 4" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Trade Execution History Table */}
          <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 overflow-hidden">
            <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-3">
              Executed Trades ({simulationResult.trade_history.length} Transactions)
            </h4>
            {simulationResult.trade_history.length === 0 ? (
              <div className="p-6 text-center text-xs text-slate-500">
                No trades triggered by PPO during this period (all-cash or passive holding).
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs font-mono">
                  <thead>
                    <tr className="border-b border-slate-800 text-slate-400">
                      <th className="py-2.5 px-3">Step</th>
                      <th className="py-2.5 px-3">Action</th>
                      <th className="py-2.5 px-3">Shares Held</th>
                      <th className="py-2.5 px-3">Portfolio Value</th>
                      <th className="py-2.5 px-3">Transaction Cost</th>
                      <th className="py-2.5 px-3">Step Reward</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 text-slate-300">
                    {simulationResult.trade_history.map((t, idx) => (
                      <tr key={idx} className="hover:bg-slate-800/40">
                        <td className="py-2 px-3 text-cyan-400">Day {t.step}</td>
                        <td className="py-2 px-3">
                          <span className={`inline-flex px-2 py-0.5 rounded font-bold ${
                            t.action === 'BUY' ? 'bg-emerald-500/10 text-emerald-400' : 'bg-rose-500/10 text-rose-400'
                          }`}>
                            {t.action}
                          </span>
                        </td>
                        <td className="py-2 px-3">{t.shares.toLocaleString()}</td>
                        <td className="py-2 px-3 font-semibold text-slate-100">₹{t.portfolio_value.toLocaleString()}</td>
                        <td className="py-2 px-3 text-rose-400">₹{t.cost.toFixed(2)}</td>
                        <td className="py-2 px-3 text-slate-400">{t.reward.toFixed(6)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
