/**
 * API Service for NeuroTrader UI
 * Connects to the isolated neurotrader-api adapter running on port 8000
 */

const API_BASE = 'http://127.0.0.1:8000/api';

async function fetchJSON(endpoint, options = {}) {
  try {
    const res = await fetch(`${API_BASE}${endpoint}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `HTTP Error ${res.status}`);
    }
    return await res.json();
  } catch (error) {
    console.error(`API Error on ${endpoint}:`, error);
    throw error;
  }
}

export const api = {
  getHealth: () => fetchJSON('/health'),
  getStocks: () => fetchJSON('/stocks'),
  getMarketData: (stock, split = 'test', limit = 60) => 
    fetchJSON(`/market-data?stock=${stock}&split=${split}&limit=${limit}`),
  getLSTMPrediction: (stock, split = 'test', step = 0) => 
    fetchJSON(`/lstm/prediction?stock=${stock}&split=${split}&step=${step}`),
  getLSTMEmbedding: (stock, split = 'test', step = 0) => 
    fetchJSON(`/lstm/embedding?stock=${stock}&split=${split}&step=${step}`),
  getPPOdecision: (payload) => 
    fetchJSON('/ppo/decision', { method: 'POST', body: JSON.stringify(payload) }),
  runSimulation: (payload) => 
    fetchJSON('/simulation/run', { method: 'POST', body: JSON.stringify(payload) }),
  getPerformance: () => fetchJSON('/performance'),
  getArtifacts: () => fetchJSON('/artifacts'),
};
