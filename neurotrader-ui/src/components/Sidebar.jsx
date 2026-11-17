import React from 'react';
import { 
  LayoutDashboard, 
  Database, 
  Cpu, 
  Brain, 
  LineChart, 
  Award, 
  ShieldCheck, 
  ExternalLink 
} from 'lucide-react';

export default function Sidebar({ currentPage, setCurrentPage, healthStatus }) {
  const navItems = [
    { id: 'overview', label: 'Overview', icon: LayoutDashboard },
    { id: 'data', label: 'Data Explorer', icon: Database },
    { id: 'lstm', label: 'LSTM Analysis', icon: Cpu },
    { id: 'ppo', label: 'PPO Agent', icon: Brain },
    { id: 'simulator', label: 'Trading Simulator', icon: LineChart },
    { id: 'performance', label: 'Performance', icon: Award },
  ];

  return (
    <aside className="w-64 bg-slate-900 border-r border-slate-800 flex flex-col justify-between shrink-0 select-none">
      <div>
        {/* Brand Header */}
        <div className="p-6 border-b border-slate-800/80">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-gradient-to-tr from-cyan-600 to-emerald-500 flex items-center justify-center font-bold text-slate-950 text-xl shadow-lg shadow-cyan-500/20">
              N
            </div>
            <div>
              <h1 className="font-bold text-lg tracking-tight bg-gradient-to-r from-cyan-400 to-emerald-400 bg-clip-text text-transparent">
                NeuroTrader
              </h1>
              <p className="text-xs text-slate-400 font-mono">B.Tech Final Project</p>
            </div>
          </div>
        </div>

        {/* Navigation */}
        <nav className="p-4 space-y-1.5">
          <div className="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
            Navigation
          </div>
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = currentPage === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setCurrentPage(item.id)}
                className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-lg text-sm font-medium transition-all ${
                  isActive
                    ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 shadow-sm'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? 'text-cyan-400' : 'text-slate-400'}`} />
                {item.label}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Isolation Status Badge */}
      <div className="p-4 m-4 rounded-xl bg-slate-950/70 border border-slate-800">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-medium text-slate-400 flex items-center gap-1.5">
            <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" /> System State
          </span>
          <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-mono bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            {healthStatus?.status === 'online' ? 'CONNECTED' : 'DISCONNECTED'}
          </span>
        </div>
        <p className="text-[11px] text-slate-400 leading-relaxed">
          Isolated adapter running on <code className="text-cyan-300 font-mono">:8000</code>. Original backend untouched.
        </p>
      </div>
    </aside>
  );
}
