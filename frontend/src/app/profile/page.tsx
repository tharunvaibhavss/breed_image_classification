'use client';

import React from 'react';
import { User, Mail, Shield, Cpu, Layers, Activity, Award, CheckCircle2 } from 'lucide-react';

export default function ProfilePage() {
  return (
    <div className="max-w-4xl mx-auto space-y-8">
      {/* Profile Header */}
      <div className="glass-card p-8 rounded-3xl space-y-6">
        <div className="flex flex-col sm:flex-row items-center space-y-4 sm:space-y-0 sm:space-x-6 text-center sm:text-left">
          <div className="w-20 h-20 rounded-3xl bg-gradient-to-tr from-indigo-600 to-purple-600 p-1 shadow-xl shadow-indigo-600/30">
            <div className="w-full h-full bg-slate-950 rounded-[22px] flex items-center justify-center">
              <User className="w-10 h-10 text-indigo-400" />
            </div>
          </div>
          <div>
            <div className="flex items-center justify-center sm:justify-start space-x-2">
              <h1 className="text-2xl font-bold text-white">Dr. Agricultural Researcher</h1>
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
            </div>
            <p className="text-xs text-slate-400 mt-1">researcher@cattle.ai • Lead ML & Veterinary Specialist</p>
            <div className="flex items-center justify-center sm:justify-start space-x-2 mt-3">
              <span className="px-2.5 py-0.5 rounded-full bg-indigo-500/10 text-indigo-300 text-[10px] font-semibold border border-indigo-500/20">
                Authorized AI User
              </span>
              <span className="px-2.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-300 text-[10px] font-semibold border border-emerald-500/20">
                ICAR Certified
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Account Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
        <div className="glass-card p-6 rounded-2xl space-y-2">
          <span className="text-xs text-slate-400">Total Scans Executed</span>
          <div className="text-3xl font-extrabold text-white">142</div>
          <span className="text-[11px] text-emerald-400">100% Valid Analysis</span>
        </div>

        <div className="glass-card p-6 rounded-2xl space-y-2">
          <span className="text-xs text-slate-400">Breed Classification Accuracy</span>
          <div className="text-3xl font-extrabold text-emerald-400">94.8%</div>
          <span className="text-[11px] text-slate-400">Validated on Test Split</span>
        </div>

        <div className="glass-card p-6 rounded-2xl space-y-2">
          <span className="text-xs text-slate-400">Pipeline Latency Average</span>
          <div className="text-3xl font-extrabold text-indigo-400">44.5 ms</div>
          <span className="text-[11px] text-slate-400">Sub-second Realtime</span>
        </div>
      </div>

      {/* AI Pipeline Architecture Info */}
      <div className="glass-card p-8 rounded-3xl space-y-6">
        <h2 className="text-xl font-bold text-white flex items-center space-x-2">
          <Cpu className="w-5 h-5 text-indigo-400" />
          <span>Active Deep Learning Pipeline Config</span>
        </h2>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs">
          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
            <span className="text-slate-500 block">Object Detector</span>
            <span className="font-bold text-white">YOLOv8n (Cattle/Buffalo)</span>
            <span className="text-[10px] text-emerald-400 block pt-1">Status: Trained & Active</span>
          </div>

          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
            <span className="text-slate-500 block">Breed Classifier</span>
            <span className="font-bold text-white">EfficientNet-B0 (6 Breeds)</span>
            <span className="text-[10px] text-emerald-400 block pt-1">Status: Transfer Learned</span>
          </div>

          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
            <span className="text-slate-500 block">Explainability Engine</span>
            <span className="font-bold text-white">Grad-CAM (Final Conv Layer)</span>
            <span className="text-[10px] text-emerald-400 block pt-1">Status: Attached Hook</span>
          </div>
        </div>
      </div>
    </div>
  );
}
