import React, { useState } from 'react';
import {
  Database,
  Sliders,
  Upload,
  Sparkles,
  CheckCircle2,
  AlertCircle,
  FileText,
  ArrowRight,
  RefreshCw,
} from 'lucide-react';
import { Dataset, TaskType } from '../../ml/types.ts';
import {
  createDiabetesDataset,
  createIrisDataset,
  createMallCustomersDataset,
  generateSyntheticClassification,
  generateSyntheticClustering,
  generateSyntheticRegression,
  inspectColumns,
  parseCSV,
  computeFingerprint,
} from '../../ml/datasets.ts';

interface DatasetStudioProps {
  activeDataset: Dataset;
  onSelectDataset: (dataset: Dataset) => void;
  onProceedToPreprocessing: () => void;
  onUpdateTargetAndFeatures: (target: string, selectedFeatures: string[]) => void;
  selectedTarget: string;
  selectedFeatures: string[];
}

export const DatasetStudio: React.FC<DatasetStudioProps> = ({
  activeDataset,
  onSelectDataset,
  onProceedToPreprocessing,
  onUpdateTargetAndFeatures,
  selectedTarget,
  selectedFeatures,
}) => {
  const [sourceTab, setSourceTab] = useState<'sample' | 'synthetic' | 'upload'>('sample');

  // Synthetic Generator state
  const [synthTask, setSynthTask] = useState<TaskType>('classification');
  const [synthSamples, setSynthSamples] = useState<number>(150);
  const [synthFeatures, setSynthFeatures] = useState<number>(4);
  const [synthNoise, setSynthNoise] = useState<number>(0.2);
  const [synthClasses, setSynthClasses] = useState<number>(3);
  const [synthSeparation, setSynthSeparation] = useState<number>(1.2);
  const [synthSeed, setSynthSeed] = useState<number>(42);

  // Upload state
  const [uploadText, setUploadText] = useState<string>('');
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadFileName, setUploadFileName] = useState<string>('custom_data.csv');
  const [page, setPage] = useState<number>(0);
  const rowsPerPage = 8;

  // Handle Synthetic Generation
  const handleGenerateSynthetic = () => {
    let generated: Dataset;
    if (synthTask === 'regression') {
      generated = generateSyntheticRegression({
        samples: synthSamples,
        features: synthFeatures,
        noise: synthNoise,
        seed: synthSeed,
      });
    } else if (synthTask === 'clustering') {
      generated = generateSyntheticClustering({
        samples: synthSamples,
        features: synthFeatures,
        nClusters: synthClasses,
        clusterStd: 1.0 + synthNoise * 1.5,
        seed: synthSeed,
      });
    } else {
      generated = generateSyntheticClassification({
        samples: synthSamples,
        features: synthFeatures,
        nClasses: synthClasses,
        classSeparation: synthSeparation,
        noise: synthNoise,
        seed: synthSeed,
      });
    }
    onSelectDataset(generated);
  };

  // Handle CSV file upload
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    setUploadError(null);
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.name.endsWith('.csv') && file.type !== 'text/csv' && file.type !== 'application/vnd.ms-excel') {
      setUploadError('Please select a valid .csv file.');
      return;
    }

    if (file.size > 5 * 1024 * 1024) {
      setUploadError('File exceeds 5MB size limit.');
      return;
    }

    setUploadFileName(file.name);
    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const text = String(event.target?.result || '');
        const { columns, data } = parseCSV(text);
        const meta = inspectColumns(data, columns);
        const fp = computeFingerprint(data, columns);

        // Auto-detect target column
        let target: string | undefined;
        let task: TaskType = 'classification';
        const candidates = ['target', 'label', 'class', 'species', 'outcome', 'y', columns[columns.length - 1]];
        for (const c of candidates) {
          if (columns.includes(c)) {
            target = c;
            break;
          }
        }
        if (target && meta[target]) {
          task = meta[target].type === 'numerical' && meta[target].uniqueCount > 10 ? 'regression' : 'classification';
        }

        const newDataset: Dataset = {
          id: `upload-${Date.now()}`,
          name: file.name,
          source: 'upload',
          fingerprint: fp,
          version: '1.0.0',
          createdAt: new Date().toISOString(),
          columns,
          columnMeta: meta,
          data,
          suggestedTarget: target,
          suggestedTask: task,
          description: `Uploaded from ${file.name} with ${data.length} records.`,
        };

        onSelectDataset(newDataset);
      } catch (err: any) {
        setUploadError(err.message || 'Error parsing CSV file.');
      }
    };
    reader.onerror = () => setUploadError('Failed to read file from disk.');
    reader.readAsText(file);
  };

  // Handle Pasted CSV text
  const handlePastedCSV = () => {
    setUploadError(null);
    if (!uploadText.trim()) {
      setUploadError('Please paste CSV contents into the text area.');
      return;
    }
    try {
      const { columns, data } = parseCSV(uploadText);
      const meta = inspectColumns(data, columns);
      const fp = computeFingerprint(data, columns);

      let target = columns[columns.length - 1];
      let task: TaskType = meta[target]?.type === 'numerical' && meta[target]?.uniqueCount > 10 ? 'regression' : 'classification';

      const newDataset: Dataset = {
        id: `paste-${Date.now()}`,
        name: 'Pasted CSV Data',
        source: 'upload',
        fingerprint: fp,
        version: '1.0.0',
        createdAt: new Date().toISOString(),
        columns,
        columnMeta: meta,
        data,
        suggestedTarget: target,
        suggestedTask: task,
        description: `Pasted dataset with ${data.length} records and ${columns.length} columns.`,
      };
      onSelectDataset(newDataset);
    } catch (err: any) {
      setUploadError(err.message || 'Failed to parse pasted CSV data.');
    }
  };

  // Toggle feature selection
  const handleToggleFeature = (col: string) => {
    if (selectedFeatures.includes(col)) {
      if (selectedFeatures.length <= 1) return; // Keep at least 1 feature
      onUpdateTargetAndFeatures(
        selectedTarget,
        selectedFeatures.filter((f) => f !== col)
      );
    } else {
      onUpdateTargetAndFeatures(selectedTarget, [...selectedFeatures, col]);
    }
  };

  // Change target column
  const handleChangeTarget = (newTarget: string) => {
    const updatedFeatures = selectedFeatures.filter((f) => f !== newTarget);
    onUpdateTargetAndFeatures(newTarget, updatedFeatures);
  };

  const paginatedData = activeDataset.data.slice(page * rowsPerPage, (page + 1) * rowsPerPage);
  const totalPages = Math.ceil(activeDataset.data.length / rowsPerPage);

  return (
    <div className="space-y-8 max-w-7xl mx-auto py-6 px-4">
      {/* Header section */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white">01. Dataset Studio</h1>
          <p className="text-sm text-slate-400 mt-1">
            Collect, inspect, and configure feature targets for machine learning training.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={onProceedToPreprocessing}
            className="flex items-center gap-2 px-4 py-2 text-sm font-semibold text-white bg-indigo-600 hover:bg-indigo-500 rounded-lg transition-colors shadow-sm"
          >
            Configure Preprocessing
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Source Selector Segmented Tabs */}
      <div className="border border-slate-800 rounded-xl bg-slate-950 p-6 space-y-6">
        <div className="flex items-center justify-between flex-wrap gap-4 border-b border-slate-800 pb-4">
          <div className="text-sm font-semibold text-slate-200">Select Dataset Acquisition Method</div>
          <div className="flex items-center gap-1 bg-slate-900 p-1 rounded-lg border border-slate-800">
            <button
              onClick={() => setSourceTab('sample')}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
                sourceTab === 'sample' ? 'bg-indigo-600 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Curated Samples
            </button>
            <button
              onClick={() => setSourceTab('synthetic')}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
                sourceTab === 'synthetic' ? 'bg-indigo-600 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Synthetic Generator
            </button>
            <button
              onClick={() => setSourceTab('upload')}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
                sourceTab === 'upload' ? 'bg-indigo-600 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              CSV Upload / Paste
            </button>
          </div>
        </div>

        {/* Tab 1: Curated Samples */}
        {sourceTab === 'sample' && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div
              onClick={() => onSelectDataset(createIrisDataset())}
              className={`cursor-pointer p-4 rounded-xl border transition-all ${
                activeDataset.id === 'sample-iris'
                  ? 'border-indigo-500 bg-indigo-950/20 ring-1 ring-indigo-500/50'
                  : 'border-slate-800 bg-slate-900/60 hover:border-slate-700'
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-semibold text-emerald-400 font-mono">Classification</span>
                <span className="text-xs text-slate-500">150 rows · 4 features</span>
              </div>
              <h3 className="text-base font-semibold text-white mb-1">Iris Flower Species</h3>
              <p className="text-xs text-slate-400 leading-relaxed mb-3">
                Classic 3-class benchmark (Setosa, Versicolor, Virginica) with 4 sepal and petal measurements.
              </p>
              <div className="text-xs text-slate-500 font-mono">Target: species</div>
            </div>

            <div
              onClick={() => onSelectDataset(createDiabetesDataset())}
              className={`cursor-pointer p-4 rounded-xl border transition-all ${
                activeDataset.id === 'sample-diabetes'
                  ? 'border-indigo-500 bg-indigo-950/20 ring-1 ring-indigo-500/50'
                  : 'border-slate-800 bg-slate-900/60 hover:border-slate-700'
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-semibold text-sky-400 font-mono">Regression</span>
                <span className="text-xs text-slate-500">150 rows · 5 features</span>
              </div>
              <h3 className="text-base font-semibold text-white mb-1">Diabetes Progression</h3>
              <p className="text-xs text-slate-400 leading-relaxed mb-3">
                Physiological metrics (BMI, BP, cholesterol, glucose) predicting continuous disease progression.
              </p>
              <div className="text-xs text-slate-500 font-mono">Target: progression_index</div>
            </div>

            <div
              onClick={() => onSelectDataset(createMallCustomersDataset())}
              className={`cursor-pointer p-4 rounded-xl border transition-all ${
                activeDataset.id === 'sample-mall'
                  ? 'border-indigo-500 bg-indigo-950/20 ring-1 ring-indigo-500/50'
                  : 'border-slate-800 bg-slate-900/60 hover:border-slate-700'
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-semibold text-purple-400 font-mono">Clustering</span>
                <span className="text-xs text-slate-500">160 rows · 4 features</span>
              </div>
              <h3 className="text-base font-semibold text-white mb-1">Mall Customer Segments</h3>
              <p className="text-xs text-slate-400 leading-relaxed mb-3">
                Unsupervised customer profiling based on annual income, age, and spending behavior habits.
              </p>
              <div className="text-xs text-slate-500 font-mono">Target: None (Unsupervised)</div>
            </div>
          </div>
        )}

        {/* Tab 2: Synthetic Generator */}
        {sourceTab === 'synthetic' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-2">Learning Task</label>
                <select
                  value={synthTask}
                  onChange={(e) => setSynthTask(e.target.value as TaskType)}
                  className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-indigo-500"
                >
                  <option value="classification">Classification (make_classification)</option>
                  <option value="regression">Regression (make_regression)</option>
                  <option value="clustering">Clustering (make_blobs)</option>
                </select>
              </div>

              <div>
                <div className="flex justify-between text-xs font-semibold text-slate-300 mb-2">
                  <span>Sample Count</span>
                  <span className="font-mono text-indigo-400">{synthSamples}</span>
                </div>
                <input
                  type="range"
                  min="50"
                  max="400"
                  step="25"
                  value={synthSamples}
                  onChange={(e) => setSynthSamples(Number(e.target.value))}
                  className="w-full accent-indigo-500 cursor-pointer"
                />
              </div>

              <div>
                <div className="flex justify-between text-xs font-semibold text-slate-300 mb-2">
                  <span>Feature Count</span>
                  <span className="font-mono text-indigo-400">{synthFeatures}</span>
                </div>
                <input
                  type="range"
                  min="2"
                  max="8"
                  value={synthFeatures}
                  onChange={(e) => setSynthFeatures(Number(e.target.value))}
                  className="w-full accent-indigo-500 cursor-pointer"
                />
              </div>

              <div>
                <div className="flex justify-between text-xs font-semibold text-slate-300 mb-2">
                  <span>Noise Level</span>
                  <span className="font-mono text-indigo-400">{synthNoise.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.0"
                  max="1.0"
                  step="0.05"
                  value={synthNoise}
                  onChange={(e) => setSynthNoise(Number(e.target.value))}
                  className="w-full accent-indigo-500 cursor-pointer"
                />
              </div>

              {synthTask !== 'regression' && (
                <div>
                  <div className="flex justify-between text-xs font-semibold text-slate-300 mb-2">
                    <span>{synthTask === 'clustering' ? 'Clusters (k)' : 'Classes'}</span>
                    <span className="font-mono text-indigo-400">{synthClasses}</span>
                  </div>
                  <input
                    type="range"
                    min="2"
                    max="5"
                    value={synthClasses}
                    onChange={(e) => setSynthClasses(Number(e.target.value))}
                    className="w-full accent-indigo-500 cursor-pointer"
                  />
                </div>
              )}

              <div>
                <div className="flex justify-between text-xs font-semibold text-slate-300 mb-2">
                  <span>Random Seed (Reproducibility)</span>
                  <span className="font-mono text-indigo-400">{synthSeed}</span>
                </div>
                <input
                  type="number"
                  value={synthSeed}
                  onChange={(e) => setSynthSeed(Number(e.target.value))}
                  className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-1.5 text-sm text-slate-200 focus:outline-none focus:border-indigo-500"
                />
              </div>
            </div>

            <div className="flex justify-end">
              <button
                onClick={handleGenerateSynthetic}
                className="flex items-center gap-2 px-4 py-2 text-sm font-semibold text-white bg-indigo-600 hover:bg-indigo-500 rounded-lg transition-colors"
              >
                <Sparkles className="w-4 h-4" />
                Generate Synthetic Dataset
              </button>
            </div>
          </div>
        )}

        {/* Tab 3: CSV Upload */}
        {sourceTab === 'upload' && (
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* File Dropzone */}
              <div className="border-2 border-dashed border-slate-800 rounded-xl p-6 flex flex-col items-center justify-center text-center bg-slate-900/40 hover:bg-slate-900/70 transition-colors">
                <Upload className="w-8 h-8 text-indigo-400 mb-3" />
                <p className="text-sm font-medium text-slate-200 mb-1">Upload CSV file</p>
                <p className="text-xs text-slate-500 mb-4">Max file size 5MB · Comma delimited with header row</p>
                <label className="cursor-pointer px-4 py-2 text-xs font-semibold text-white bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors">
                  Browse File
                  <input type="file" accept=".csv" onChange={handleFileUpload} className="hidden" />
                </label>
              </div>

              {/* Paste Text */}
              <div className="space-y-2">
                <label className="text-xs font-semibold text-slate-300">Or Paste Raw CSV Text</label>
                <textarea
                  value={uploadText}
                  onChange={(e) => setUploadText(e.target.value)}
                  placeholder="feature1,feature2,feature3,target&#10;1.2,3.4,5.1,1&#10;2.3,1.8,4.9,0"
                  rows={4}
                  className="w-full bg-slate-900 border border-slate-800 rounded-lg p-3 text-xs font-mono text-slate-200 focus:outline-none focus:border-indigo-500"
                />
                <button
                  onClick={handlePastedCSV}
                  className="px-3.5 py-1.5 text-xs font-semibold text-white bg-indigo-600 hover:bg-indigo-500 rounded-lg transition-colors"
                >
                  Parse Pasted CSV
                </button>
              </div>
            </div>

            {uploadError && (
              <div className="flex items-center gap-2 p-3 bg-rose-950/40 border border-rose-800 rounded-lg text-rose-300 text-xs">
                <AlertCircle className="w-4 h-4 shrink-0" />
                <span>{uploadError}</span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Active Dataset Inspection Panel */}
      <div className="border border-slate-800 rounded-xl bg-slate-950 p-6 space-y-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-4">
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-lg font-bold text-white">{activeDataset.name}</h2>
              <span className="text-xs font-mono text-slate-500">[{activeDataset.fingerprint}]</span>
            </div>
            <p className="text-xs text-slate-400 mt-1">{activeDataset.description}</p>
          </div>
          <div className="flex items-center gap-4 text-xs font-mono text-slate-400">
            <span>Rows: <strong className="text-white tabular-nums">{activeDataset.data.length}</strong></span>
            <span>·</span>
            <span>Features: <strong className="text-white tabular-nums">{activeDataset.columns.length}</strong></span>
            <span>·</span>
            <span className="text-indigo-400 capitalize">{activeDataset.suggestedTask || 'supervised'}</span>
          </div>
        </div>

        {/* Feature & Target Selection Controls */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 bg-slate-900/60 p-4 rounded-xl border border-slate-800">
          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-2">
              Supervised Target Column (y)
            </label>
            <select
              value={selectedTarget}
              onChange={(e) => handleChangeTarget(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-indigo-500"
            >
              <option value="">-- No Target (Unsupervised / K-Means) --</option>
              {activeDataset.columns.map((col) => (
                <option key={col} value={col}>
                  {col} ({activeDataset.columnMeta[col]?.type}, {activeDataset.columnMeta[col]?.uniqueCount} unique)
                </option>
              ))}
            </select>
            <p className="text-xs text-slate-500 mt-1.5">
              Select the column your model will learn to predict. Leave unselected for K-Means Clustering.
            </p>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-2">
              Input Features Selection (X)
            </label>
            <div className="flex flex-wrap gap-2 max-h-28 overflow-y-auto p-1">
              {activeDataset.columns.map((col) => {
                const isTarget = col === selectedTarget;
                const isSelected = selectedFeatures.includes(col);
                return (
                  <button
                    key={col}
                    disabled={isTarget}
                    onClick={() => handleToggleFeature(col)}
                    className={`px-2.5 py-1 text-xs rounded-md border transition-colors ${
                      isTarget
                        ? 'opacity-40 cursor-not-allowed bg-slate-900 border-slate-800 text-slate-500'
                        : isSelected
                        ? 'bg-indigo-950/80 border-indigo-500 text-indigo-300'
                        : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    {col} {isTarget ? '(target)' : ''}
                  </button>
                );
              })}
            </div>
            <p className="text-xs text-slate-500 mt-1.5">
              {selectedFeatures.length} features selected for model input vector.
            </p>
          </div>
        </div>

        {/* Column Metadata Inspection Table */}
        <div>
          <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-3">
            Column Metadata &amp; Distribution Stats
          </h3>
          <div className="overflow-x-auto border border-slate-800 rounded-lg">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-900 text-slate-400 border-b border-slate-800">
                <tr>
                  <th className="py-2.5 px-4 font-medium">Column</th>
                  <th className="py-2.5 px-4 font-medium">Data Type</th>
                  <th className="py-2.5 px-4 font-medium">Missing</th>
                  <th className="py-2.5 px-4 font-medium">Unique Values</th>
                  <th className="py-2.5 px-4 font-medium">Mean / Mode</th>
                  <th className="py-2.5 px-4 font-medium">Min / Max</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono">
                {activeDataset.columns.map((col) => {
                  const m = activeDataset.columnMeta[col];
                  if (!m) return null;
                  return (
                    <tr key={col} className="hover:bg-slate-900/40">
                      <td className="py-2 px-4 text-slate-200 font-semibold">{col}</td>
                      <td className="py-2 px-4 text-slate-400">{m.type}</td>
                      <td className="py-2 px-4 text-slate-400 tabular-nums">{m.missingCount}</td>
                      <td className="py-2 px-4 text-slate-400 tabular-nums">{m.uniqueCount}</td>
                      <td className="py-2 px-4 text-slate-400 tabular-nums">
                        {m.mean !== undefined ? m.mean : String(m.sampleValues[0] || 'N/A')}
                      </td>
                      <td className="py-2 px-4 text-slate-400 tabular-nums">
                        {m.min !== undefined ? `${m.min} / ${m.max}` : 'N/A'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Raw Data Preview Table */}
        <div>
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
              Data Sample Records (Page {page + 1} of {totalPages || 1})
            </h3>
            <div className="flex items-center gap-2">
              <button
                disabled={page === 0}
                onClick={() => setPage((p) => Math.max(0, p - 1))}
                className="px-2.5 py-1 text-xs bg-slate-900 border border-slate-800 rounded disabled:opacity-40 text-slate-300 hover:bg-slate-800"
              >
                Previous
              </button>
              <button
                disabled={page >= totalPages - 1}
                onClick={() => setPage((p) => p + 1)}
                className="px-2.5 py-1 text-xs bg-slate-900 border border-slate-800 rounded disabled:opacity-40 text-slate-300 hover:bg-slate-800"
              >
                Next
              </button>
            </div>
          </div>
          <div className="overflow-x-auto border border-slate-800 rounded-lg">
            <table className="w-full text-left text-xs font-mono">
              <thead className="bg-slate-900 text-slate-400 border-b border-slate-800">
                <tr>
                  <th className="py-2.5 px-4 font-medium text-slate-500">#</th>
                  {activeDataset.columns.map((c) => (
                    <th key={c} className="py-2.5 px-4 font-medium">
                      {c}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {paginatedData.map((row, idx) => (
                  <tr key={idx} className="hover:bg-slate-900/40">
                    <td className="py-2 px-4 text-slate-600">{page * rowsPerPage + idx + 1}</td>
                    {activeDataset.columns.map((c) => (
                      <td key={c} className="py-2 px-4 text-slate-300 tabular-nums">
                        {String(row[c] ?? 'null')}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};
