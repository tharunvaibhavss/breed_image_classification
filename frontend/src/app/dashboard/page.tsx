'use client';

import React from 'react';
import Link from 'next/link';
import {
  UploadCloud,
  CheckCircle2,
  Clock,
  Zap,
  Activity,
  Layers,
  Database,
  ArrowRight,
} from 'lucide-react';

export default function DashboardPage() {
  const stats = [
    { label: 'Total Scans Executed', value: '142', change: '+18 today', icon: Activity, color: 'from-indigo-500 to-purple-600' },
    { label: 'Cattle Identified', value: '84', change: '59.1% total', icon: Layers, color: 'from-purple-500 to-pink-600' },
    { label: 'Buffalo Identified', value: '58', change: '40.9% total', icon: Database, color: 'from-emerald-500 to-teal-600' },
    { label: 'Avg Inference Latency', value: '44.5 ms', change: 'CPU Optimized', icon: Zap, color: 'from-amber-500 to-orange-600' },
  ];

  const recentScans = [
    { id: '1', breed: 'Gir Cattle', animal: 'Cattle', confidence: '94.8%', status: 'success', time: '10 mins ago' },
    { id: '2', breed: 'Murrah Buffalo', animal: 'Buffalo', confidence: '92.1%', status: 'success', time: '42 mins ago' },
    { id: '3', breed: 'Sahiwal Cattle', animal: 'Cattle', confidence: '89.5%', status: 'success', time: '2 hours ago' },
    { id: '4', breed: 'Jaffarabadi Buffalo', animal: 'Buffalo', confidence: '91.2%', status: 'success', time: '5 hours ago' },
  ];

  return (
    <div className="space-y-8">
      {/* Dashboard Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-extrabold text-white tracking-tight">AI Research Dashboard</h1>
          <p className="text-xs text-slate-400 mt-1">
            Real-time livestock breed classification and explainability metrics
          </p>
        </div>
        <Link
          href="/upload"
          className="flex items-center space-x-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 text-white font-semibold text-xs shadow-lg shadow-indigo-600/30 hover:scale-105 transition"
        >
          <UploadCloud className="w-4 h-4" />
          <span>New Image Scan</span>
        </Link>
      </div>

      {/* Metrics Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat, idx) => {
          const Icon = stat.icon;
          return (
            <div key={idx} className="glass-card p-6 rounded-2xl space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-400">{stat.label}</span>
                <div className={`w-9 h-9 rounded-xl bg-gradient-to-tr ${stat.color} flex items-center justify-center text-white shadow-md`}>
                  <Icon className="w-4 h-4" />
                </div>
              </div>
              <div>
                <div className="text-3xl font-extrabold text-white">{stat.value}</div>
                <div className="text-[11px] font-medium text-emerald-400 mt-1">{stat.change}</div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Main Grid Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Quick Upload Widget */}
        <div className="lg:col-span-1 glass-card p-6 rounded-3xl space-y-6 flex flex-col justify-between">
          <div className="space-y-2">
            <h2 className="text-xl font-bold text-white">Instant Breed Recognition</h2>
            <p className="text-xs text-slate-400 leading-relaxed">
              Upload any livestock image to trigger YOLO bounding box detection, EfficientNet-B0 breed classification, and Grad-CAM explainability.
            </p>
          </div>

          <div className="border-2 border-dashed border-slate-700/80 hover:border-indigo-500/60 rounded-2xl p-8 text-center bg-slate-900/40 transition group">
            <UploadCloud className="w-10 h-10 text-indigo-400 mx-auto group-hover:scale-110 transition" />
            <p className="mt-3 text-xs font-semibold text-white">Drag & drop image here</p>
            <p className="text-[11px] text-slate-500 mt-1">Supports JPEG, PNG, WebP (max 10MB)</p>
            <Link
              href="/upload"
              className="inline-block mt-4 px-4 py-2 rounded-lg bg-indigo-600/20 text-indigo-300 text-xs font-semibold border border-indigo-500/30 hover:bg-indigo-600/30 transition"
            >
              Select Image File
            </Link>
          </div>

          <div className="text-[11px] text-slate-500 flex items-center justify-between pt-2 border-t border-slate-800">
            <span>Model: EfficientNet-B0</span>
            <span>Target Latency: &lt;50ms</span>
          </div>
        </div>

        {/* Recent Predictions Feed */}
        <div className="lg:col-span-2 glass-card p-6 rounded-3xl space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-bold text-white">Recent Classification Logs</h2>
            <Link href="/history" className="text-xs font-semibold text-indigo-400 hover:text-indigo-300 flex items-center space-x-1">
              <span>View History</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>

          <div className="space-y-3">
            {recentScans.map((scan) => (
              <div
                key={scan.id}
                className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 flex items-center justify-between hover:bg-slate-800/60 transition"
              >
                <div className="flex items-center space-x-4">
                  <div className="w-10 h-10 rounded-xl bg-indigo-600/20 border border-indigo-500/30 flex items-center justify-center">
                    <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                  </div>
                  <div>
                    <h3 className="text-sm font-bold text-white">{scan.breed}</h3>
                    <p className="text-xs text-slate-400">Species: {scan.animal} • {scan.time}</p>
                  </div>
                </div>

                <div className="text-right">
                  <div className="text-sm font-extrabold text-emerald-400">{scan.confidence}</div>
                  <span className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">
                    {scan.status}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
