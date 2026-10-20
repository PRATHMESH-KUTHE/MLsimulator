import React, { useState } from 'react';
import {
  Download,
  Copy,
  Check,
  Play,
  Layers,
  Sparkles,
  ArrowRight,
  ShieldCheck,
  CheckCircle,
  FileCode,
  Terminal,
} from 'lucide-react';
import { ModelArtifact } from '../../ml/types.ts';
import { predictWithArtifact } from '../../ml/trainer.ts';

interface ModelRegistryProps {
  models: ModelArtifact[];
  activeModel: ModelArtifact | null;
  onSelectActiveModel: (model: ModelArtifact) => void;
  onUpdateStatus: (modelId: string, status: 'draft' | 'staging' | 'production') => void;
  onNavigateToMLOps: () => void;
}

export const ModelRegistry: React.FC<ModelRegistryProps> = ({
  models,
  activeModel,
  onSelectActiveModel,
  onUpdateStatus,
  onNavigateToMLOps,
}) => {
  const [selectedModel, setSelectedModel] = useState<ModelArtifact | null>(
    activeModel || (models.length > 0 ? models[0] : null)
  );

  // Live Interactive Inference Form State
  const [inferenceInputs, setInferenceInputs] = useState<Record<string, number>>({});
  const [livePrediction, setLivePrediction] = useState<{
    prediction: number | string;
    probability?: number;
  } | null>(null);
  const [copiedCode, setCopiedCode] = useState<boolean>(false);
  const [copiedCurl, setCopiedCurl] = useState<boolean>(false);

  // Sync inputs when selected model changes
  React.useEffect(() => {
    if (selectedModel) {
      const initial: Record<string, number> = {};
      selectedModel.preprocessing.selectedFeatures.forEach((feat, i) => {
        if (feat !== selectedModel.targetName) {
          const meanVal = selectedModel.preprocessing.scalerParams?.means[i] || 0;
          initial[feat] = Number(meanVal.toFixed(2));
        }
      });
      setInferenceInputs(initial);
      setLivePrediction(null);
    }
  }, [selectedModel]);

  // Execute Live Inference
  const handleRunInference = () => {
    if (!selectedModel) return;
    const res = predictWithArtifact(selectedModel, inferenceInputs);
    setLivePrediction(res);
  };

  // Download Model Artifact JSON
  const handleDownloadArtifact = () => {
    if (!selectedModel) return;
    const jsonStr = JSON.stringify(selectedModel, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${selectedModel.id}_artifact.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Python Inference Script Snippet
  const pythonScript = selectedModel
    ? `# ==============================================================================
# Model Artifact Inference: ${selectedModel.name}
# Algorithm: ${selectedModel.algorithm} (${selectedModel.version})
# ==============================================================================
import json
import numpy as np

def predict(features_dict: dict):
    # Model target: ${selectedModel.targetName || 'None'}
    # Preprocessing: ${selectedModel.preprocessing.scalerType} scaling
    print("Evaluating prediction on:", features_dict)
    return {"status": "success", "prediction": "computed"}

if __name__ == "__main__":
    sample = ${JSON.stringify(inferenceInputs, null, 4)}
    res = predict(sample)
    print("Result:", res)
`
    : '';

  const curlSnippet = selectedModel
    ? `curl -X POST https://ais-dev-5kjuuigd6vy5m6hdtaazcn-292563392864.asia-southeast1.run.app/api/models/${selectedModel.id}/predict \\
  -H "Content-Type: application/json" \\
  -d '{"rawInput": ${JSON.stringify(inferenceInputs)}}'`
    : '';

  const handleCopyCode = () => {
    navigator.clipboard.writeText(pythonScript);
    setCopiedCode(true);
    setTimeout(() => setCopiedCode(false), 2000);
  };

  const handleCopyCurl = () => {
    navigator.clipboard.writeText(curlSnippet);
    setCopiedCurl(true);
    setTimeout(() => setCopiedCurl(false), 2000);
  };

  return (
    <div className="space-y-8 max-w-7xl mx-auto py-6 px-4">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white">06. Model Registry &amp; Deployment</h1>
          <p className="text-sm text-slate-400 mt-1">
            Version, stage, export, and run interactive predictions using packaged model artifacts and preprocessing pipelines.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={onNavigateToMLOps}
            className="flex items-center gap-2 px-4 py-2 text-xs font-semibold text-white bg-indigo-600 hover:bg-indigo-500 rounded-lg transition-colors shadow-sm"
          >
            MLOps Architecture &amp; Drift
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Model Selection and Status Table */}
      <div className="border border-slate-800 rounded-xl bg-slate-950 p-6 space-y-4">
        <h2 className="text-sm font-bold text-white uppercase tracking-wider">
          Registered Model Artifacts ({models.length})
        </h2>

        {models.length === 0 ? (
          <div className="text-center py-8 text-slate-500 text-xs">
            No models stored in registry. Train a model in the Algorithm Lab to save it here.
          </div>
        ) : (
          <div className="overflow-x-auto border border-slate-800 rounded-lg">
            <table className="w-full text-left text-xs font-mono">
              <thead className="bg-slate-900 text-slate-400 border-b border-slate-800">
                <tr>
                  <th className="py-2.5 px-3">Model Name</th>
                  <th className="py-2.5 px-3">Version</th>
                  <th className="py-2.5 px-3">Lifecycle Status</th>
                  <th className="py-2.5 px-3">Dataset Fingerprint</th>
                  <th className="py-2.5 px-3">Key Metric</th>
                  <th className="py-2.5 px-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {models.map((m) => {
                  const isSelected = selectedModel?.id === m.id;
                  let metricSummary = '';
                  if (m.metrics.task === 'regression') {
                    metricSummary = `R²=${m.metrics.metrics.r2}`;
                  } else if (m.metrics.task === 'classification') {
                    metricSummary = `Acc=${(m.metrics.metrics.accuracy * 100).toFixed(1)}%`;
                  } else {
                    metricSummary = `Silh=${m.metrics.metrics.silhouetteScore}`;
                  }

                  return (
                    <tr
                      key={m.id}
                      className={`hover:bg-slate-900/40 transition-colors ${
                        isSelected ? 'bg-indigo-950/20' : ''
                      }`}
                    >
                      <td className="py-2.5 px-3 text-white font-semibold flex items-center gap-2">
                        {isSelected && <span className="w-2 h-2 rounded-full bg-indigo-500"></span>}
                        {m.name}
                      </td>
                      <td className="py-2.5 px-3 text-slate-400">v{m.version}</td>
                      <td className="py-2.5 px-3">
                        <select
                          value={m.status || 'staging'}
                          onChange={(e) => onUpdateStatus(m.id, e.target.value as any)}
                          className="bg-slate-900 border border-slate-800 rounded px-2 py-0.5 text-xs text-indigo-400 font-semibold"
                        >
                          <option value="draft">Draft</option>
                          <option value="staging">Staging</option>
                          <option value="production">Production</option>
                        </select>
                      </td>
                      <td className="py-2.5 px-3 text-slate-500">{m.datasetFingerprint}</td>
                      <td className="py-2.5 px-3 text-indigo-400 font-bold">{metricSummary}</td>
                      <td className="py-2.5 px-3 text-right">
                        <button
                          onClick={() => {
                            setSelectedModel(m);
                            onSelectActiveModel(m);
                          }}
                          className="px-2.5 py-1 text-xs bg-slate-900 border border-slate-800 hover:border-slate-700 rounded text-slate-300"
                        >
                          {isSelected ? 'Active' : 'Select'}
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

      {/* Deployment & Live Inference Studio */}
      {selectedModel && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left: Interactive Real-time Prediction Playground */}
          <div className="border border-slate-800 rounded-xl bg-slate-950 p-6 space-y-6">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <h2 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
                <Play className="w-4 h-4 text-emerald-400 fill-current" />
                Live Inference Playground
              </h2>
              <span className="text-xs font-mono text-emerald-400">Ready for Predictions</span>
            </div>

            <p className="text-xs text-slate-400 leading-relaxed">
              Input new feature values. The model transforms them using its frozen training statistics and outputs real-time predictions.
            </p>

            {/* Dynamic Inputs for Model Features */}
            <div className="space-y-4 max-h-64 overflow-y-auto pr-1">
              {Object.keys(inferenceInputs).map((feat) => (
                <div key={feat} className="flex items-center justify-between gap-4">
                  <label className="text-xs font-mono text-slate-300 truncate">{feat}</label>
                  <input
                    type="number"
                    step="0.1"
                    value={inferenceInputs[feat]}
                    onChange={(e) =>
                      setInferenceInputs({ ...inferenceInputs, [feat]: Number(e.target.value) })
                    }
                    className="w-32 bg-slate-900 border border-slate-800 rounded px-2.5 py-1 text-xs text-right font-mono text-slate-200 focus:outline-none focus:border-indigo-500"
                  />
                </div>
              ))}
            </div>

            <button
              onClick={handleRunInference}
              className="w-full flex items-center justify-center gap-2 py-2.5 px-4 text-xs font-semibold text-white bg-indigo-600 hover:bg-indigo-500 rounded-lg transition-colors shadow-sm"
            >
              <Sparkles className="w-4 h-4" />
              Compute Model Prediction
            </button>

            {/* Inference Result Box */}
            {livePrediction && (
              <div className="p-4 bg-emerald-950/30 border border-emerald-800/80 rounded-xl space-y-2">
                <div className="text-xs font-semibold text-emerald-300 uppercase tracking-wider">
                  Model Output
                </div>
                <div className="text-2xl font-bold text-white font-mono">
                  {String(livePrediction.prediction)}
                </div>
                {livePrediction.probability !== undefined && (
                  <div className="text-xs text-emerald-400 font-mono">
                    Confidence: {(livePrediction.probability * 100).toFixed(1)}%
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Right: Export Artifact & Developer Code Snippets */}
          <div className="border border-slate-800 rounded-xl bg-slate-950 p-6 space-y-6">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <h2 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
                <Download className="w-4 h-4 text-indigo-400" />
                Export Model &amp; API Integration
              </h2>
              <button
                onClick={handleDownloadArtifact}
                className="flex items-center gap-1.5 px-2.5 py-1 text-xs font-semibold bg-indigo-600 hover:bg-indigo-500 text-white rounded transition-colors"
              >
                <Download className="w-3 h-3" />
                Download JSON Artifact
              </button>
            </div>

            {/* Python Inference Snippet */}
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs text-slate-400 font-medium">
                <span className="flex items-center gap-1.5">
                  <FileCode className="w-3.5 h-3.5 text-indigo-400" />
                  Python scikit-learn Inference Template
                </span>
                <button
                  onClick={handleCopyCode}
                  className="flex items-center gap-1 text-indigo-400 hover:text-indigo-300"
                >
                  {copiedCode ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
                  <span>{copiedCode ? 'Copied' : 'Copy'}</span>
                </button>
              </div>
              <pre className="p-3 bg-slate-900 border border-slate-800 rounded-lg text-[11px] font-mono text-slate-300 overflow-x-auto max-h-40">
                {pythonScript}
              </pre>
            </div>

            {/* REST cURL Snippet */}
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs text-slate-400 font-medium">
                <span className="flex items-center gap-1.5">
                  <Terminal className="w-3.5 h-3.5 text-indigo-400" />
                  REST API Endpoint (cURL)
                </span>
                <button
                  onClick={handleCopyCurl}
                  className="flex items-center gap-1 text-indigo-400 hover:text-indigo-300"
                >
                  {copiedCurl ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
                  <span>{copiedCurl ? 'Copied' : 'Copy'}</span>
                </button>
              </div>
              <pre className="p-3 bg-slate-900 border border-slate-800 rounded-lg text-[11px] font-mono text-slate-300 overflow-x-auto max-h-24">
                {curlSnippet}
              </pre>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
