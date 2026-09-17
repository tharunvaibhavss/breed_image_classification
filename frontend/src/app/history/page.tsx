'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { History, Search, Filter, Calendar, CheckCircle2, Clock, ArrowRight } from 'lucide-react';

export default function HistoryPage() {
  const [searchQuery, setSearchQuery] = useState('');
  const [speciesFilter, setSpeciesFilter] = useState<'all' | 'cattle' | 'buffalo'>('all');

  const historyItems = [
    {
      id: 'SCAN-1092',
      date: '2026-08-27 21:45',
      breed: 'Gir Cattle',
      species: 'cattle',
      confidence: 94.8,
      status: 'success',
      latency_ms: 42.5,
      bbox: [25, 30, 350, 350],
      top3: ['Gir (94.8%)', 'Sahiwal (3.2%)', 'Ongole (1.1%)'],
    },
    {
      id: 'SCAN-1091',
      date: '2026-08-27 20:12',
      breed: 'Murrah Buffalo',
      species: 'buffalo',
      confidence: 92.1,
      status: 'success',
      latency_ms: 46.1,
      bbox: [10, 15, 400, 380],
      top3: ['Murrah (92.1%)', 'Jaffarabadi (5.4%)', 'Surti (1.8%)'],
    },
    {
      id: 'SCAN-1090',
      date: '2026-08-27 18:30',
      breed: 'Sahiwal Cattle',
      species: 'cattle',
      confidence: 89.5,
      status: 'success',
      latency_ms: 38.9,
      bbox: [40, 20, 320, 310],
      top3: ['Sahiwal (89.5%)', 'Gir (8.1%)', 'Ongole (1.6%)'],
    },
    {
      id: 'SCAN-1089',
      date: '2026-08-27 15:10',
      breed: 'Jaffarabadi Buffalo',
      species: 'buffalo',
      confidence: 91.2,
      status: 'success',
      latency_ms: 51.0,
      bbox: [0, 0, 400, 400],
      top3: ['Jaffarabadi (91.2%)', 'Surti (6.0%)', 'Murrah (2.1%)'],
    },
    {
      id: 'SCAN-1088',
      date: '2026-08-27 11:05',
      breed: 'Ongole Cattle',
      species: 'cattle',
      confidence: 88.4,
      status: 'success',
      latency_ms: 41.2,
      bbox: [15, 25, 380, 360],
      top3: ['Ongole (88.4%)', 'Gir (7.3%)', 'Sahiwal (3.1%)'],
    },
  ];

  const filteredItems = historyItems.filter((item) => {
    const matchesSpecies = speciesFilter === 'all' || item.species === speciesFilter;
    const matchesSearch =
      item.breed.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.id.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesSpecies && matchesSearch;
  });

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-extrabold text-white tracking-tight">
            Prediction History Logs
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Complete audit trail of past AI breed classification executions
          </p>
        </div>

        {/* Species Filter */}
        <div className="flex items-center space-x-2 bg-slate-900 p-1.5 rounded-xl border border-slate-800 text-xs">
          <button
            onClick={() => setSpeciesFilter('all')}
            className={`px-3 py-1.5 rounded-lg font-semibold transition ${
              speciesFilter === 'all' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'
            }`}
          >
            All Species
          </button>
          <button
            onClick={() => setSpeciesFilter('cattle')}
            className={`px-3 py-1.5 rounded-lg font-semibold transition ${
              speciesFilter === 'cattle' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'
            }`}
          >
            Cattle Only
          </button>
          <button
            onClick={() => setSpeciesFilter('buffalo')}
            className={`px-3 py-1.5 rounded-lg font-semibold transition ${
              speciesFilter === 'buffalo' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'
            }`}
          >
            Buffalo Only
          </button>
        </div>
      </div>

      {/* Search Input */}
      <div className="relative max-w-md">
        <Search className="w-4 h-4 text-slate-500 absolute left-3.5 top-3.5" />
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search by scan ID or breed name..."
          className="w-full bg-slate-900 border border-slate-800 rounded-xl py-2.5 pl-10 pr-4 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500 transition"
        />
      </div>

      {/* History Table / List */}
      <div className="glass-card rounded-3xl overflow-hidden border border-slate-800">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-900/80 border-b border-slate-800 text-slate-400 uppercase font-semibold">
              <tr>
                <th className="px-6 py-4">Scan ID & Time</th>
                <th className="px-6 py-4">Species</th>
                <th className="px-6 py-4">Top-1 Predicted Breed</th>
                <th className="px-6 py-4">Confidence</th>
                <th className="px-6 py-4">Latency</th>
                <th className="px-6 py-4 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {filteredItems.map((item) => (
                <tr key={item.id} className="hover:bg-slate-900/40 transition">
                  <td className="px-6 py-4">
                    <div className="font-bold text-white">{item.id}</div>
                    <div className="text-[11px] text-slate-500 flex items-center space-x-1 mt-0.5">
                      <Calendar className="w-3 h-3 text-slate-400" />
                      <span>{item.date}</span>
                    </div>
                  </td>

                  <td className="px-6 py-4">
                    <span className="px-2.5 py-1 rounded-md bg-indigo-500/10 text-indigo-300 font-semibold uppercase border border-indigo-500/20">
                      {item.species}
                    </span>
                  </td>

                  <td className="px-6 py-4">
                    <div className="font-bold text-slate-200">{item.breed}</div>
                    <div className="text-[10px] text-slate-500 mt-0.5">
                      Top-3: {item.top3.join(', ')}
                    </div>
                  </td>

                  <td className="px-6 py-4">
                    <span className="font-extrabold text-emerald-400">{item.confidence}%</span>
                  </td>

                  <td className="px-6 py-4 text-slate-400">
                    {item.latency_ms} ms
                  </td>

                  <td className="px-6 py-4 text-right">
                    <Link
                      href="/upload"
                      className="inline-flex items-center space-x-1 text-indigo-400 hover:text-indigo-300 font-semibold"
                    >
                      <span>Re-Analyze</span>
                      <ArrowRight className="w-3 h-3" />
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
