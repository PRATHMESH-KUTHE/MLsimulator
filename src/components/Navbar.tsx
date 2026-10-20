import React from 'react';
import { Play, RotateCcw } from 'lucide-react';

export type ActiveTab =
  | 'studio'
  | 'preprocessing'
  | 'lab'
  | 'simulation'
  | 'experiments'
  | 'registry'
  | 'mlops';

interface NavbarProps {
  activeTab: ActiveTab;
  setActiveTab: (tab: ActiveTab) => void;
  onQuickRun?: () => void;
  datasetName?: string;
  hasTrainedModel?: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  onQuickRun,
  datasetName,
  hasTrainedModel,
}) => {
  return (
    <header className="flex items-center justify-between gap-8 px-6 py-3.5 border-b border-slate-800 bg-slate-950/90 backdrop-blur sticky top-0 z-40">
      {/* Zone 1: Single text wordmark */}
      <div className="flex items-center gap-3 shrink-0">
        <a
          href="#"
          onClick={(e) => {
            e.preventDefault();
            setActiveTab('studio');
          }}
          className="text-lg font-bold tracking-tight text-white whitespace-nowrap hover:text-indigo-400 transition-colors"
        >
          ML Simulator &amp; MLOps
        </a>
        {datasetName && (
          <span className="hidden lg:inline text-xs text-slate-500 font-mono">
            / {datasetName}
          </span>
        )}
      </div>

      {/* Zone 2: Single-line clean text navigation links */}
      <nav className="hidden md:flex items-center gap-6 text-sm font-medium text-slate-400">
        <button
          onClick={() => setActiveTab('studio')}
          className={`hover:text-white transition-colors whitespace-nowrap shrink-0 py-1 ${
            activeTab === 'studio' ? 'text-indigo-400 border-b-2 border-indigo-500 font-semibold' : ''
          }`}
        >
          01. Dataset Studio
        </button>
        <button
          onClick={() => setActiveTab('preprocessing')}
          className={`hover:text-white transition-colors whitespace-nowrap shrink-0 py-1 ${
            activeTab === 'preprocessing' ? 'text-indigo-400 border-b-2 border-indigo-500 font-semibold' : ''
          }`}
        >
          02. Preprocessing
        </button>
        <button
          onClick={() => setActiveTab('lab')}
          className={`hover:text-white transition-colors whitespace-nowrap shrink-0 py-1 ${
            activeTab === 'lab' ? 'text-indigo-400 border-b-2 border-indigo-500 font-semibold' : ''
          }`}
        >
          03. Algorithm Lab
        </button>
        <button
          onClick={() => setActiveTab('simulation')}
          className={`hover:text-white transition-colors whitespace-nowrap shrink-0 py-1 ${
            activeTab === 'simulation' ? 'text-indigo-400 border-b-2 border-indigo-500 font-semibold' : ''
          }`}
        >
          04. Simulation &amp; Viz
        </button>
        <button
          onClick={() => setActiveTab('experiments')}
          className={`hover:text-white transition-colors whitespace-nowrap shrink-0 py-1 ${
            activeTab === 'experiments' ? 'text-indigo-400 border-b-2 border-indigo-500 font-semibold' : ''
          }`}
        >
          05. Experiments
        </button>
        <button
          onClick={() => setActiveTab('registry')}
          className={`hover:text-white transition-colors whitespace-nowrap shrink-0 py-1 ${
            activeTab === 'registry' ? 'text-indigo-400 border-b-2 border-indigo-500 font-semibold' : ''
          }`}
        >
          06. Registry &amp; Deploy
        </button>
        <button
          onClick={() => setActiveTab('mlops')}
          className={`hover:text-white transition-colors whitespace-nowrap shrink-0 py-1 ${
            activeTab === 'mlops' ? 'text-indigo-400 border-b-2 border-indigo-500 font-semibold' : ''
          }`}
        >
          07. MLOps Lifecycle
        </button>
      </nav>

      {/* Zone 3: 1 Primary Action */}
      <div className="flex items-center gap-3 shrink-0">
        {hasTrainedModel ? (
          <button
            onClick={() => setActiveTab('simulation')}
            className="flex items-center gap-2 px-3.5 py-1.5 text-xs font-semibold text-white bg-indigo-600 rounded-lg hover:bg-indigo-500 transition-colors whitespace-nowrap shrink-0 shadow-sm shadow-indigo-600/30"
          >
            <Play className="w-3.5 h-3.5 fill-current" />
            Inspect Simulation
          </button>
        ) : (
          <button
            onClick={() => setActiveTab('lab')}
            className="flex items-center gap-2 px-3.5 py-1.5 text-xs font-semibold text-white bg-indigo-600 rounded-lg hover:bg-indigo-500 transition-colors whitespace-nowrap shrink-0 shadow-sm shadow-indigo-600/30"
          >
            <Play className="w-3.5 h-3.5 fill-current" />
            Train Active Model
          </button>
        )}
      </div>
    </header>
  );
};
