import React from 'react';
import Link from 'next/link';
import { Cpu, Github, Activity } from 'lucide-react';

export default function Footer() {
  return (
    <footer className="mt-auto border-t border-slate-800/80 bg-slate-950/80 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex flex-col md:flex-row items-center justify-between gap-4">
        {/* Left Side */}
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 rounded-lg bg-indigo-600/20 border border-indigo-500/30 flex items-center justify-center">
            <Cpu className="w-4 h-4 text-indigo-400" />
          </div>
          <div>
            <p className="text-sm font-semibold text-slate-200">
              AI-Powered Breed Recognition System
            </p>
            <p className="text-xs text-slate-500">
              Indigenous Indian Cattle & Buffalo Deep Learning Pipeline
            </p>
          </div>
        </div>

        {/* Status Indicator */}
        <div className="flex items-center space-x-2 bg-slate-900/80 px-3 py-1.5 rounded-full border border-slate-800 text-xs">
          <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
          <span className="text-slate-400">YOLOv8 + EfficientNet-B0 + Grad-CAM API Online</span>
        </div>

        {/* Right Side */}
        <div className="text-xs text-slate-500 flex items-center space-x-4">
          <span>© 2026 AI Breed Recognition</span>
          <Link
            href="https://github.com"
            target="_blank"
            className="hover:text-slate-300 transition flex items-center space-x-1"
          >
            <Github className="w-3.5 h-3.5" />
            <span>GitHub Repository</span>
          </Link>
        </div>
      </div>
    </footer>
  );
}
