import React, { useState } from 'react';
import {
  RotateCcw,
  BarChart2,
  Trash2,
  CheckCircle,
  Clock,
  ArrowRight,
  Sliders,
  Sparkles,
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
} from 'recharts';
import { Experiment } from '../../ml/types.ts';

interface ExperimentTrackerProps {
  experiments: Experiment[];
  onRerunExperiment: (experiment: Experiment) => void;
  onClearHistory: () => void;
  onNavigateToRegistry: () => void;
}

export const ExperimentTracker: React.FC<ExperimentTrackerProps> = ({
  experiments,
  onRerunExperiment,
  onClearHistory,
  onNavigateToRegistry,
}) => {
  const [selectedExpIds, setSelectedExpIds] = useState<string[]>([]);

  // Toggle selection for comparison
  const handleToggleSelect = (id: string) => {
    if (selectedExpIds.includes(id)) {
      setSelectedExpIds(selectedExpIds.filter((e) => e !== id));
    } else {
      if (selectedExpIds.length >= 3) {
        setSelectedExpIds([...selectedExpIds.slice(1), id]);
      } else {
        setSelectedExpIds([...selectedExpIds, id]);
      }
    }
  };

  const selectedExperiments = experiments.filter((e) => selectedExpIds.includes(e.id));

  // Build comparison chart data
  const comparisonData = selectedExperiments.map((exp) => {
    let primaryMetric = 0;
    let metricName = 'Score';
    if (exp.metrics.task === 'regression') {
      primaryMetric = exp.metrics.metrics.r2;
      metricName = 'R² Score';
    } else if (exp.metrics.task === 'classification') {
      primaryMetric = exp.metrics.metrics.accuracy;
      metricName = 'Accuracy';
    } else if (exp.metrics.task === 'clustering') {
      primaryMetric = exp.metrics.metrics.silhouetteScore;
      metricName = 'Silhouette';
    }

    return {
      name: `${exp.algorithm.replace(/_/g, ' ')} (${exp.id.slice(-4)})`,
      score: Number(primaryMetric.toFixed(3)),
      durationMs: exp.trainingDurationMs,
      metricName,
    };
  });

  return (
    <div className="space-y-8 max-w-7xl mx-auto py-6 px-4">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white">05. Experiment Tracking</h1>
          <p className="text-sm text-slate-400 mt-1">
            Compare model runs, analyze performance metrics, inspect hyperparameters, and reproduce experiments.
          </p>
        </div>
        <div className="flex items-center gap-3">
          {experiments.length > 0 && (
            <button
              onClick={onClearHistory}
              className="flex items-center gap-2 px-3 py-2 text-xs font-semibold text-rose-400 hover:text-rose-300 bg-rose-950/20 border border-rose-900/60 rounded-lg transition-colors"
            >
              <Trash2 className="w-3.5 h-3.5" />
              Clear History
            </button>
          )}
          <button
            onClick={onNavigateToRegistry}
            className="flex items-center gap-2 px-4 py-2 text-xs font-semibold text-white bg-indigo-600 hover:bg-indigo-500 rounded-lg transition-colors shadow-sm"
          >
            Open Model Registry
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Side-by-side Experiment Comparison (if 2+ selected) */}
      {selectedExperiments.length >= 2 && (
        <div className="border border-indigo-900/60 bg-slate-950 p-6 rounded-xl space-y-6">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <h2 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
              <BarChart2 className="w-4 h-4 text-indigo-400" />
              Side-by-Side Experiment Comparison ({selectedExperiments.length} Runs)
            </h2>
            <button
              onClick={() => setSelectedExpIds([])}
              className="text-xs text-slate-400 hover:text-white"
            >
              Clear Selection
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
            {/* Comparative Bar Chart */}
            <div className="h-48 w-full bg-slate-900/40 p-3 rounded-xl border border-slate-800">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={comparisonData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="name" stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 10 }} />
                  <YAxis stroke="#64748b" tick={{ fill: '#64748b', fontSize: 10 }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: 8, fontSize: 11 }}
                  />
                  <Bar dataKey="score" fill="#6366f1" name="Score Metric" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Metrics Comparison Table */}
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs font-mono">
                <thead className="bg-slate-900 text-slate-400 border-b border-slate-800">
                  <tr>
                    <th className="py-2 px-3">Run ID</th>
                    <th className="py-2 px-3">Algorithm</th>
                    <th className="py-2 px-3">Primary Metric</th>
                    <th className="py-2 px-3">Duration</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {selectedExperiments.map((exp) => {
                    let scoreText = '';
                    if (exp.metrics.task === 'regression') {
                      scoreText = `R²: ${exp.metrics.metrics.r2} (RMSE: ${exp.metrics.metrics.rmse})`;
                    } else if (exp.metrics.task === 'classification') {
                      scoreText = `Acc: ${(exp.metrics.metrics.accuracy * 100).toFixed(1)}% (F1: ${exp.metrics.metrics.f1Score})`;
                    } else {
                      scoreText = `Silhouette: ${exp.metrics.metrics.silhouetteScore}`;
                    }

                    return (
                      <tr key={exp.id}>
                        <td className="py-2 px-3 text-slate-400">{exp.id.slice(-6)}</td>
                        <td className="py-2 px-3 text-white font-semibold">{exp.algorithm}</td>
                        <td className="py-2 px-3 text-indigo-400 font-bold">{scoreText}</td>
                        <td className="py-2 px-3 text-slate-400">{exp.trainingDurationMs} ms</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Experiments History Table */}
      <div className="border border-slate-800 rounded-xl bg-slate-950 p-6 space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-bold text-white uppercase tracking-wider">
            Logged Experiment Runs ({experiments.length})
          </h2>
          <span className="text-xs text-slate-500">
            Select checkboxes to compare up to 3 experiments side-by-side
          </span>
        </div>

        {experiments.length === 0 ? (
          <div className="text-center py-12 text-slate-500 text-xs space-y-2">
            <Clock className="w-8 h-8 mx-auto text-slate-600 mb-2" />
            <p>No experiment runs logged yet.</p>
            <p>Train any algorithm in the Algorithm Lab to record reproducibility runs.</p>
          </div>
        ) : (
          <div className="overflow-x-auto border border-slate-800 rounded-lg">
            <table className="w-full text-left text-xs font-mono">
              <thead className="bg-slate-900 text-slate-400 border-b border-slate-800">
                <tr>
                  <th className="py-2.5 px-3">Compare</th>
                  <th className="py-2.5 px-3">Run ID</th>
                  <th className="py-2.5 px-3">Timestamp</th>
                  <th className="py-2.5 px-3">Algorithm</th>
                  <th className="py-2.5 px-3">Dataset</th>
                  <th className="py-2.5 px-3">Test Metrics</th>
                  <th className="py-2.5 px-3">Duration</th>
                  <th className="py-2.5 px-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {experiments.map((exp) => {
                  const isChecked = selectedExpIds.includes(exp.id);
                  let metricsDisplay = '';
                  if (exp.metrics.task === 'regression') {
                    metricsDisplay = `R²=${exp.metrics.metrics.r2} · RMSE=${exp.metrics.metrics.rmse}`;
                  } else if (exp.metrics.task === 'classification') {
                    metricsDisplay = `Acc=${(exp.metrics.metrics.accuracy * 100).toFixed(1)}% · F1=${exp.metrics.metrics.f1Score}`;
                  } else {
                    metricsDisplay = `Silh=${exp.metrics.metrics.silhouetteScore} · WCSS=${exp.metrics.metrics.inertia}`;
                  }

                  return (
                    <tr
                      key={exp.id}
                      className={`hover:bg-slate-900/40 transition-colors ${
                        isChecked ? 'bg-indigo-950/20' : ''
                      }`}
                    >
                      <td className="py-2.5 px-3">
                        <input
                          type="checkbox"
                          checked={isChecked}
                          onChange={() => handleToggleSelect(exp.id)}
                          className="rounded bg-slate-900 border-slate-700 text-indigo-600 focus:ring-0 cursor-pointer"
                        />
                      </td>
                      <td className="py-2.5 px-3 text-slate-400">{exp.id.slice(-6)}</td>
                      <td className="py-2.5 px-3 text-slate-500">
                        {new Date(exp.timestamp).toLocaleTimeString()}
                      </td>
                      <td className="py-2.5 px-3 text-white font-semibold">
                        {exp.algorithm.replace(/_/g, ' ')}
                      </td>
                      <td className="py-2.5 px-3 text-slate-400">{exp.datasetName}</td>
                      <td className="py-2.5 px-3 text-indigo-400 font-bold">{metricsDisplay}</td>
                      <td className="py-2.5 px-3 text-slate-400 tabular-nums">
                        {exp.trainingDurationMs} ms
                      </td>
                      <td className="py-2.5 px-3 text-right">
                        <button
                          onClick={() => onRerunExperiment(exp)}
                          className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs text-indigo-400 hover:text-indigo-300 bg-indigo-950/40 border border-indigo-800 rounded transition-colors"
                          title="Rerun with exact seed & hyperparams"
                        >
                          <RotateCcw className="w-3 h-3" />
                          Rerun
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
