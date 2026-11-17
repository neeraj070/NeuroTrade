import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import Overview from './pages/Overview';
import DataExplorer from './pages/DataExplorer';
import LSTMAnalysis from './pages/LSTMAnalysis';
import PPOAgent from './pages/PPOAgent';
import TradingSimulator from './pages/TradingSimulator';
import Performance from './pages/Performance';
import { api } from './services/api';

export default function App() {
  const [currentPage, setCurrentPage] = useState('overview');
  const [healthStatus, setHealthStatus] = useState(null);

  useEffect(() => {
    async function checkHealth() {
      try {
        const res = await api.getHealth();
        setHealthStatus(res);
      } catch (err) {
        setHealthStatus({ status: 'offline', error: err.message });
      }
    }
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const pageTitles = {
    overview: 'System Architecture & Benchmark Overview',
    data: 'Multi-Asset Market Data & Feature Explorer',
    lstm: 'Frozen LSTM Representation & Latent Embedding Analysis',
    ppo: 'PPO Decoupled Actor-Critic Policy Decision Interface',
    simulator: 'Isolated Gymnasium Backtest Trading Simulator',
    performance: 'Audited Strategy Comparisons & Performance Benchmarks',
  };

  return (
    <div className="flex h-screen bg-slate-950 text-slate-100 overflow-hidden">
      {/* Sidebar Navigation */}
      <Sidebar 
        currentPage={currentPage} 
        setCurrentPage={setCurrentPage} 
        healthStatus={healthStatus} 
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <Header 
          currentPageTitle={pageTitles[currentPage] || 'NeuroTrader Dashboard'} 
          healthStatus={healthStatus} 
        />

        <main className="flex-1 overflow-y-auto p-8 bg-slate-950">
          {currentPage === 'overview' && <Overview onNavigate={setCurrentPage} />}
          {currentPage === 'data' && <DataExplorer />}
          {currentPage === 'lstm' && <LSTMAnalysis />}
          {currentPage === 'ppo' && <PPOAgent />}
          {currentPage === 'simulator' && <TradingSimulator />}
          {currentPage === 'performance' && <Performance />}
        </main>
      </div>
    </div>
  );
}
