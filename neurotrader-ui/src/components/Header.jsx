import React from 'react';
import { ShieldCheck, Activity, Database, GitBranch } from 'lucide-react';

export default function Header({ currentPageTitle, healthStatus }) {
  return (
    <header className="h-16 bg-slate-900/60 backdrop-blur-md border-b border-slate-800 px-8 flex items-center justify-between shrink-0">
      <div>
        <h2 className="text-lg font-semibold text-slate-100 flex items-center gap-2">
          {currentPageTitle}
        </h2>
      </div>

      <div className="flex items-center gap-6">
        {/* Model status chips */}
        <div className="flex items-center gap-3 text-xs">
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-800/80 border border-slate-700/60 text-slate-300">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            <span className="text-slate-400">LSTM:</span>
            <span className="font-mono text-cyan-300">Frozen (55.5% Acc)</span>
          </div>

          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-800/80 border border-slate-700/60 text-slate-300">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            <span className="text-slate-400">PPO:</span>
            <span className="font-mono text-emerald-300">Iter 4 (+0.34%)</span>
          </div>

          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-800/80 border border-slate-700/60 text-slate-300">
            <Database className="w-3.5 h-3.5 text-cyan-400" />
            <span className="text-slate-400">NSE Equities:</span>
            <span className="font-mono text-slate-200">5 Stocks</span>
          </div>
        </div>
      </div>
    </header>
  );
}
