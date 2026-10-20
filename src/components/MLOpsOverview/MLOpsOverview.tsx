import React, { useState } from 'react';
import {
  Layers,
  Activity,
  AlertTriangle,
  RotateCcw,
  ShieldCheck,
  CheckCircle,
  Database,
  ArrowRight,
  TrendingDown,
  Sparkles,
} from 'lucide-react';
import { ModelArtifact, PreprocessedData } from '../../ml/types.ts';

interface MLOpsOverviewProps {
  activeModel: ModelArtifact | null;
  preprocessedData: PreprocessedData | null;
  onTriggerRetraining: () => void;
  onNavigateToStudio: () => void;
}

export const MLOpsOverview: React.FC<MLOpsOverviewProps> = ({
  activeModel,
  preprocessedData,
  onTriggerRetraining,
  onNavigateToStudio,
}) => {
  // Drift simulation state
  const [driftShiftFactor, setDriftShiftFactor] = useState<number>(0.0);

  // Compute reference baseline vs simulated inference stream distributions
  const featureNames = preprocessedData?.featureNames || [];
  const scalerParams = activeModel?.preprocessing.scalerParams;

  const driftMetrics = featureNames.map((name, idx) => {
    const baseMean = scalerParams?.means[idx] ?? 0;
    const baseStd = scalerParams?.stds[idx] ?? 1;

    // Simulated drifted mean in production stream
    const currentMean = baseMean + driftShiftFactor * baseStd;
    const zDivergence = Math.abs(currentMean - baseMean) / (baseStd || 1);
    const isDrifted = zDivergence > 0.8;

    return {
      feature: name,
      baseMean: Number(baseMean.toFixed(2)),
      currentMean: Number(currentMean.toFixed(2)),
      zDivergence: Number(zDivergence.toFixed(2)),
      isDrifted,
    };
  });

  const totalDrifted = driftMetrics.filter((m) => m.isDrifted).length;

  return (
    <div className="space-y-8 max-w-7xl mx-auto py-6 px-4">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white">07. MLOps Lifecycle &amp; Drift Monitoring</h1>
          <p className="text-sm text-slate-400 mt-1">
            End-to-end machine learning operations: data provenance, reproducibility, distribution drift, and continuous retraining loops.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={onNavigateToStudio}
            className="flex items-center gap-2 px-4 py-2 text-xs font-semibold text-white bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors border border-slate-700"
          >
            <Database className="w-4 h-4" />
            Dataset Provenance
          </button>
        </div>
      </div>

      {/* MLOps Circular Workflow Diagram */}
      <div className="border border-slate-800 rounded-xl bg-slate-950 p-6 space-y-6">
        <h2 className="text-sm font-bold text-white uppercase tracking-wider">
          The 6-Stage MLOps Continuous Engineering Loop
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-6 gap-3">
          {[
            {
              step: '01',
              title: 'Data Versioning',
              desc: 'Immutable dataset fingerprints, schema validation, and row checksums.',
              active: true,
            },
            {
              step: '02',
              title: 'Pipelines',
              desc: 'Leak-free ColumnTransformers fit on train only, frozen for inference.',
              active: true,
            },
            {
              step: '03',
              title: 'Experiment Tracking',
              desc: 'Deterministic seeds, parameter logs, metrics, and training run durations.',
              active: true,
            },
            {
              step: '04',
              title: 'Model Registry',
              desc: 'Packaged weights, metadata contracts, and Staging / Production flags.',
              active: true,
            },
            {
              step: '05',
              title: 'Inference Serving',
              desc: 'Low-latency REST API scoring with preprocessed transformation guarantees.',
              active: true,
            },
            {
              step: '06',
              title: 'Continuous Retraining',
              desc: 'Statistical drift detection triggering automated new experiment runs.',
              active: true,
            },
          ].map((item) => (
            <div
              key={item.step}
              className="bg-slate-900/60 p-4 rounded-xl border border-slate-800 flex flex-col justify-between"
            >
              <div>
                <div className="text-xs font-mono text-indigo-400 font-bold mb-1">{item.step}</div>
                <div className="text-sm font-semibold text-white mb-2">{item.title}</div>
                <p className="text-xs text-slate-400 leading-relaxed">{item.desc}</p>
              </div>
              <div className="mt-3 flex items-center gap-1.5 text-[10px] text-emerald-400 font-mono">
                <CheckCircle className="w-3 h-3" />
                <span>Verified Active</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Real-time Data Drift Simulator */}
      <div className="border border-slate-800 rounded-xl bg-slate-950 p-6 space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-4">
          <div>
            <h2 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
              <Activity className="w-4 h-4 text-indigo-400" />
              Production Feature Drift Simulator
            </h2>
            <p className="text-xs text-slate-400 mt-1">
              Simulate covariate input drift (distribution shift in real-world incoming user queries).
            </p>
          </div>

          <div className="flex items-center gap-3">
            <span className="text-xs text-slate-400">Inject Feature Shift:</span>
            <input
              type="range"
              min="0.0"
              max="2.5"
              step="0.25"
              value={driftShiftFactor}
              onChange={(e) => setDriftShiftFactor(Number(e.target.value))}
              className="w-36 accent-indigo-500 cursor-pointer"
            />
            <span className="text-xs font-mono text-indigo-400 w-12 text-right">
              +{driftShiftFactor} σ
            </span>
          </div>
        </div>

        {totalDrifted > 0 ? (
          <div className="p-4 bg-amber-950/40 border border-amber-800 rounded-xl flex items-center justify-between flex-wrap gap-4 text-xs text-amber-200">
            <div className="flex items-center gap-3">
              <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0" />
              <div>
                <strong>Covariate Drift Detected!</strong> {totalDrifted} feature(s) diverge by &gt;
                0.8 standard deviations from the training baseline. Model accuracy in production may degrade.
              </div>
            </div>

            <button
              onClick={onTriggerRetraining}
              className="flex items-center gap-2 px-3.5 py-1.5 text-xs font-semibold text-white bg-amber-600 hover:bg-amber-500 rounded-lg transition-colors shadow-sm"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              Trigger Retraining Run
            </button>
          </div>
        ) : (
          <div className="p-3 bg-emerald-950/30 border border-emerald-800 rounded-lg flex items-center gap-2 text-xs text-emerald-300">
            <CheckCircle className="w-4 h-4 shrink-0" />
            <span>
              All input distributions match the baseline reference training dataset. No distribution drift detected.
            </span>
          </div>
        )}

        {/* Drift Metrics Table */}
        <div className="overflow-x-auto border border-slate-800 rounded-lg">
          <table className="w-full text-left text-xs font-mono">
            <thead className="bg-slate-900 text-slate-400 border-b border-slate-800">
              <tr>
                <th className="py-2.5 px-4">Feature Name</th>
                <th className="py-2.5 px-4">Baseline Train Mean (μ)</th>
                <th className="py-2.5 px-4">Live Inference Mean</th>
                <th className="py-2.5 px-4">Divergence (Z-Score)</th>
                <th className="py-2.5 px-4 text-right">Monitoring Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {driftMetrics.map((row) => (
                <tr key={row.feature} className="hover:bg-slate-900/40">
                  <td className="py-2.5 px-4 text-white font-semibold">{row.feature}</td>
                  <td className="py-2.5 px-4 text-slate-400 tabular-nums">{row.baseMean}</td>
                  <td className="py-2.5 px-4 text-slate-300 tabular-nums">{row.currentMean}</td>
                  <td className="py-2.5 px-4 tabular-nums">
                    <span
                      className={`font-bold ${
                        row.isDrifted ? 'text-amber-400' : 'text-slate-400'
                      }`}
                    >
                      {row.zDivergence} σ
                    </span>
                  </td>
                  <td className="py-2.5 px-4 text-right">
                    {row.isDrifted ? (
                      <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-amber-950 border border-amber-800 text-amber-300">
                        DRIFT ALERT
                      </span>
                    ) : (
                      <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-emerald-950 border border-emerald-800 text-emerald-300">
                        STABLE
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
