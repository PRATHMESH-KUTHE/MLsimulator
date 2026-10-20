import React from 'react';
import {
  SlidersHorizontal,
  ArrowRight,
  ShieldCheck,
  CheckCircle,
  AlertTriangle,
  Info,
  Layers,
  Sparkles,
} from 'lucide-react';
import { PreprocessedData, PreprocessingConfig, TaskType } from '../../ml/types.ts';

interface PreprocessingStudioProps {
  config: PreprocessingConfig;
  onChangeConfig: (newConfig: PreprocessingConfig) => void;
  preprocessedData: PreprocessedData | null;
  onExecutePipeline: () => void;
  onProceedToLab: () => void;
  taskType: TaskType;
}

export const PreprocessingStudio: React.FC<PreprocessingStudioProps> = ({
  config,
  onChangeConfig,
  preprocessedData,
  onExecutePipeline,
  onProceedToLab,
  taskType,
}) => {
  const summary = preprocessedData?.summary;

  return (
    <div className="space-y-8 max-w-7xl mx-auto py-6 px-4">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white">02. Data Preprocessing Pipeline</h1>
          <p className="text-sm text-slate-400 mt-1">
            Transform raw features, handle missing data, encode categories, scale numbers, and partition train/test splits.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={onExecutePipeline}
            className="flex items-center gap-2 px-4 py-2 text-sm font-semibold text-white bg-indigo-600 hover:bg-indigo-500 rounded-lg transition-colors shadow-sm"
          >
            <Sparkles className="w-4 h-4" />
            Run Preprocessing Pipeline
          </button>
          {preprocessedData && (
            <button
              onClick={onProceedToLab}
              className="flex items-center gap-2 px-4 py-2 text-sm font-semibold text-white bg-emerald-600 hover:bg-emerald-500 rounded-lg transition-colors shadow-sm"
            >
              Proceed to Algorithm Lab
              <ArrowRight className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {/* Educational Data Leakage Banner */}
      <div className="border border-indigo-900/60 bg-indigo-950/30 rounded-xl p-5 flex items-start gap-4 text-xs text-indigo-200">
        <ShieldCheck className="w-6 h-6 text-indigo-400 shrink-0 mt-0.5" />
        <div className="space-y-1">
          <h4 className="font-semibold text-sm text-indigo-100">
            Strict Leakage Prevention Guarantee (Scikit-Learn Pipeline Contract)
          </h4>
          <p className="leading-relaxed text-indigo-300">
            Learned transformations (such as imputation statistics, categorical one-hot dictionaries, and standard scaler mean/std)
            are computed <strong>strictly on the training partition</strong>. The test partition is transformed using frozen parameters
            learned from train data alone. This guarantees zero data leakage into evaluation metrics.
          </p>
        </div>
      </div>

      {/* Preprocessing Controls Configuration Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* 1. Missing Values */}
        <div className="border border-slate-800 rounded-xl bg-slate-950 p-5 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800/80 pb-2.5">
            <h3 className="text-sm font-semibold text-white">1. Missing Value Imputation</h3>
            <span className="text-xs font-mono text-indigo-400">SimpleImputer</span>
          </div>
          <p className="text-xs text-slate-400 leading-relaxed">
            Replace empty cells or NaN values with robust training set statistics.
          </p>
          <div className="space-y-2">
            <label className="text-xs font-medium text-slate-300">Strategy</label>
            <select
              value={config.missingStrategy}
              onChange={(e) =>
                onChangeConfig({ ...config, missingStrategy: e.target.value as any })
              }
              className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
            >
              <option value="mean">Mean (numerical average of training split)</option>
              <option value="median">Median (robust to numerical outliers)</option>
              <option value="mode">Mode (most frequent value - best for categorical)</option>
              <option value="drop">Drop Rows (remove any record with missing values)</option>
            </select>
          </div>
        </div>

        {/* 2. Duplicates & Outliers */}
        <div className="border border-slate-800 rounded-xl bg-slate-950 p-5 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800/80 pb-2.5">
            <h3 className="text-sm font-semibold text-white">2. Cleaning &amp; Outliers</h3>
            <span className="text-xs font-mono text-indigo-400">IQR Filter</span>
          </div>
          <div className="space-y-3">
            <label className="flex items-center gap-2 cursor-pointer text-xs text-slate-300">
              <input
                type="checkbox"
                checked={config.removeDuplicates}
                onChange={(e) => onChangeConfig({ ...config, removeDuplicates: e.target.checked })}
                className="rounded bg-slate-900 border-slate-700 text-indigo-600 focus:ring-0"
              />
              <span>Remove duplicate rows</span>
            </label>

            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1.5">
                Outlier Treatment (IQR 1.5× rule)
              </label>
              <select
                value={config.outlierTreatment}
                onChange={(e) =>
                  onChangeConfig({ ...config, outlierTreatment: e.target.value as any })
                }
                className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
              >
                <option value="none">No outlier clipping (Keep raw values)</option>
                <option value="iqr_clip">Winsorize / Clip to [Q1 - 1.5×IQR, Q3 + 1.5×IQR]</option>
              </select>
            </div>
          </div>
        </div>

        {/* 3. Categorical Encoding */}
        <div className="border border-slate-800 rounded-xl bg-slate-950 p-5 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800/80 pb-2.5">
            <h3 className="text-sm font-semibold text-white">3. Categorical Encoding</h3>
            <span className="text-xs font-mono text-indigo-400">OneHotEncoder</span>
          </div>
          <p className="text-xs text-slate-400 leading-relaxed">
            Convert non-numerical text columns into numerical binary indicators.
          </p>
          <div className="space-y-2">
            <label className="text-xs font-medium text-slate-300">Encoding Method</label>
            <select
              value={config.encodingStrategy}
              onChange={(e) =>
                onChangeConfig({ ...config, encodingStrategy: e.target.value as any })
              }
              className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
            >
              <option value="onehot">One-Hot Encoding (OHE - binary columns per class)</option>
              <option value="label">Label Encoding (Ordinal integer mapping 0..K-1)</option>
              <option value="none">None (Skip categorical features)</option>
            </select>
          </div>
        </div>

        {/* 4. Feature Scaling */}
        <div className="border border-slate-800 rounded-xl bg-slate-950 p-5 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800/80 pb-2.5">
            <h3 className="text-sm font-semibold text-white">4. Feature Scaling</h3>
            <span className="text-xs font-mono text-indigo-400">StandardScaler</span>
          </div>
          <p className="text-xs text-slate-400 leading-relaxed">
            Essential for distance-based algorithms like KNN, Logistic Regression, and K-Means.
          </p>
          <div className="space-y-2">
            <label className="text-xs font-medium text-slate-300">Scaling Strategy</label>
            <select
              value={config.scalingStrategy}
              onChange={(e) =>
                onChangeConfig({ ...config, scalingStrategy: e.target.value as any })
              }
              className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
            >
              <option value="standard">StandardScaler (z = (x - μ) / σ, zero mean, unit variance)</option>
              <option value="minmax">MinMaxScaler (Scales strictly into [0, 1] range)</option>
              <option value="none">No Scaling (Keep raw units)</option>
            </select>
          </div>
        </div>

        {/* 5. Train / Test Split */}
        <div className="border border-slate-800 rounded-xl bg-slate-950 p-5 space-y-4 md:col-span-2">
          <div className="flex items-center justify-between border-b border-slate-800/80 pb-2.5">
            <h3 className="text-sm font-semibold text-white">5. Train / Test Split Partition</h3>
            <span className="text-xs font-mono text-indigo-400">train_test_split</span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <div className="flex justify-between text-xs font-medium text-slate-300 mb-2">
                <span>Hold-out Test Percentage</span>
                <span className="font-mono text-indigo-400">
                  {Math.round(config.testSplitRatio * 100)}% Test /{' '}
                  {100 - Math.round(config.testSplitRatio * 100)}% Train
                </span>
              </div>
              <input
                type="range"
                min="0.1"
                max="0.4"
                step="0.05"
                value={config.testSplitRatio}
                onChange={(e) =>
                  onChangeConfig({ ...config, testSplitRatio: Number(e.target.value) })
                }
                className="w-full accent-indigo-500 cursor-pointer"
              />
              <p className="text-xs text-slate-500 mt-1">
                Typical split is 20% test for solid statistical power without starving training.
              </p>
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1.5">
                Random Seed (Reproducibility)
              </label>
              <input
                type="number"
                value={config.randomSeed}
                onChange={(e) =>
                  onChangeConfig({ ...config, randomSeed: Number(e.target.value) })
                }
                className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-indigo-500 font-mono"
              />
              <p className="text-xs text-slate-500 mt-1">
                Ensures exact dataset partition reproducibility across experiments.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Before and After Comparison Dashboard */}
      {summary && (
        <div className="border border-slate-800 rounded-xl bg-slate-950 p-6 space-y-6">
          <div className="flex items-center justify-between border-b border-slate-800 pb-4">
            <h2 className="text-base font-bold text-white">
              Pipeline Execution Report &amp; Transformation Audit
            </h2>
            <span className="text-xs font-mono text-emerald-400 flex items-center gap-1.5">
              <CheckCircle className="w-4 h-4" />
              Pipeline Run Successful
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
            <div className="bg-slate-900/60 p-4 rounded-xl border border-slate-800">
              <div className="text-xs text-slate-500">Original Rows</div>
              <div className="text-xl font-bold text-white font-mono mt-1 tabular-nums">
                {summary.originalRowCount}
              </div>
            </div>

            <div className="bg-slate-900/60 p-4 rounded-xl border border-slate-800">
              <div className="text-xs text-slate-500">Processed Rows</div>
              <div className="text-xl font-bold text-indigo-400 font-mono mt-1 tabular-nums">
                {summary.processedRowCount}
              </div>
            </div>

            <div className="bg-slate-900/60 p-4 rounded-xl border border-slate-800">
              <div className="text-xs text-slate-500">Train Split Size</div>
              <div className="text-xl font-bold text-emerald-400 font-mono mt-1 tabular-nums">
                {summary.trainSize}
              </div>
            </div>

            <div className="bg-slate-900/60 p-4 rounded-xl border border-slate-800">
              <div className="text-xs text-slate-500">Test Split Size</div>
              <div className="text-xl font-bold text-amber-400 font-mono mt-1 tabular-nums">
                {summary.testSize}
              </div>
            </div>

            <div className="bg-slate-900/60 p-4 rounded-xl border border-slate-800">
              <div className="text-xs text-slate-500">Imputed Missing</div>
              <div className="text-xl font-bold text-slate-300 font-mono mt-1 tabular-nums">
                {summary.imputedValuesCount}
              </div>
            </div>

            <div className="bg-slate-900/60 p-4 rounded-xl border border-slate-800">
              <div className="text-xs text-slate-500">Final Features</div>
              <div className="text-xl font-bold text-slate-200 font-mono mt-1 tabular-nums">
                {summary.featureNames.length}
              </div>
            </div>
          </div>

          <div>
            <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
              Transformed Feature Space Names
            </h3>
            <div className="flex flex-wrap gap-2">
              {summary.featureNames.map((name, i) => (
                <span
                  key={i}
                  className="px-2.5 py-1 text-xs font-mono bg-slate-900 border border-slate-800 rounded-md text-slate-300"
                >
                  {name}
                </span>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
