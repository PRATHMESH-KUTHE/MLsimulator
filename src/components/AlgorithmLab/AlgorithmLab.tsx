import React, { useState } from 'react';
import {
  Play,
  RotateCcw,
  CheckCircle,
  AlertTriangle,
  Clock,
  Sparkles,
  Info,
  ArrowRight,
  Sliders,
} from 'lucide-react';
import {
  AlgorithmId,
  Experiment,
  Hyperparameters,
  ModelArtifact,
  PreprocessedData,
} from '../../ml/types.ts';
import { getAlgorithmCompatibility, trainModel } from '../../ml/trainer.ts';

interface AlgorithmLabProps {
  preprocessedData: PreprocessedData | null;
  datasetName: string;
  datasetFingerprint: string;
  onModelTrained: (artifact: ModelArtifact, experiment: Experiment) => void;
  onProceedToSimulation: () => void;
  activeArtifact: ModelArtifact | null;
}

export const AlgorithmLab: React.FC<AlgorithmLabProps> = ({
  preprocessedData,
  datasetName,
  datasetFingerprint,
  onModelTrained,
  onProceedToSimulation,
  activeArtifact,
}) => {
  const [selectedAlgo, setSelectedAlgo] = useState<AlgorithmId>('linear_regression');
  const [isTraining, setIsTraining] = useState(false);
  const [trainingError, setTrainingError] = useState<string | null>(null);

  // Hyperparameters State
  const [hyperparams, setHyperparams] = useState<Hyperparameters>({
    learningRate: 0.05,
    epochs: 100,
    regularizationC: 1.0,
    fitIntercept: true,
    maxDepth: 4,
    minSamplesSplit: 2,
    criterion: 'gini',
    nEstimators: 10,
    nNeighbors: 5,
    distanceMetric: 'euclidean',
    weights: 'uniform',
    nClusters: 3,
    maxIterations: 20,
    randomSeed: 42,
  });

  const currentCompat = getAlgorithmCompatibility(selectedAlgo);
  const isCompatible = preprocessedData ? currentCompat.taskType === preprocessedData.taskType : true;

  // Auto-switch to compatible algorithm if dataset changed
  React.useEffect(() => {
    if (preprocessedData) {
      if (preprocessedData.taskType === 'regression' && currentCompat.taskType !== 'regression') {
        setSelectedAlgo('linear_regression');
      } else if (preprocessedData.taskType === 'classification' && currentCompat.taskType !== 'classification') {
        setSelectedAlgo('decision_tree_classifier');
      } else if (preprocessedData.taskType === 'clustering' && currentCompat.taskType !== 'clustering') {
        setSelectedAlgo('kmeans_clustering');
      }
    }
  }, [preprocessedData?.taskType]);

  const handleTrain = async () => {
    if (!preprocessedData) {
      setTrainingError('Please run data preprocessing first before training.');
      return;
    }
    setTrainingError(null);
    setIsTraining(true);

    try {
      // Simulate micro-tick so UI renders loading state smoothly
      await new Promise((r) => setTimeout(r, 60));
      const { artifact, experiment } = trainModel(
        selectedAlgo,
        preprocessedData,
        datasetName,
        datasetFingerprint,
        hyperparams
      );
      onModelTrained(artifact, experiment);
    } catch (err: any) {
      setTrainingError(err.message || 'Model training failed.');
    } finally {
      setIsTraining(false);
    }
  };

  return (
    <div className="space-y-8 max-w-7xl mx-auto py-6 px-4">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white">03. Algorithm Lab</h1>
          <p className="text-sm text-slate-400 mt-1">
            Select an algorithm, tune hyperparameters, and train models on the preprocessed training partition.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={handleTrain}
            disabled={isTraining || !isCompatible}
            className="flex items-center gap-2 px-5 py-2 text-sm font-semibold text-white bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg transition-colors shadow-sm shadow-indigo-600/30"
          >
            {isTraining ? (
              <>
                <RotateCcw className="w-4 h-4 animate-spin" />
                Training Model...
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-current" />
                Train {currentCompat.name}
              </>
            )}
          </button>
          {activeArtifact && (
            <button
              onClick={onProceedToSimulation}
              className="flex items-center gap-2 px-4 py-2 text-sm font-semibold text-white bg-emerald-600 hover:bg-emerald-500 rounded-lg transition-colors shadow-sm"
            >
              Open Interactive Simulation
              <ArrowRight className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {/* Incompatibility Warning */}
      {!isCompatible && (
        <div className="flex items-center gap-3 p-4 bg-amber-950/40 border border-amber-800 rounded-xl text-amber-200 text-xs">
          <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0" />
          <div>
            <strong>Task Incompatibility: </strong>
            {currentCompat.name} is designed for <em>{currentCompat.taskType}</em> tasks, but active preprocessed data is configured for <em>{preprocessedData?.taskType}</em>. Please select a compatible algorithm or return to Dataset Studio.
          </div>
        </div>
      )}

      {trainingError && (
        <div className="flex items-center gap-3 p-4 bg-rose-950/40 border border-rose-800 rounded-xl text-rose-200 text-xs">
          <AlertTriangle className="w-5 h-5 text-rose-400 shrink-0" />
          <span>{trainingError}</span>
        </div>
      )}

      {/* Main Algorithm Selection Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left 2 Cols: Algorithm Selector & Hyperparameters */}
        <div className="lg:col-span-2 space-y-6">
          {/* Algorithm Cards */}
          <div className="border border-slate-800 rounded-xl bg-slate-950 p-6 space-y-5">
            <h2 className="text-sm font-bold text-white uppercase tracking-wider">
              Choose Machine Learning Algorithm
            </h2>

            <div className="space-y-4">
              <div>
                <div className="text-xs font-semibold text-slate-400 mb-2">Supervised - Regression</div>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  {[
                    { id: 'linear_regression', name: 'Linear Regression', desc: 'OLS Normal Equation & SGD' },
                    { id: 'decision_tree_regressor', name: 'Decision Tree Regressor', desc: 'Recursive MSE variance splits' },
                    { id: 'random_forest_regressor', name: 'Random Forest Regressor', desc: 'Bootstrap bagged tree ensemble' },
                  ].map((algo) => (
                    <button
                      key={algo.id}
                      onClick={() => setSelectedAlgo(algo.id as AlgorithmId)}
                      className={`text-left p-3.5 rounded-xl border transition-all ${
                        selectedAlgo === algo.id
                          ? 'border-indigo-500 bg-indigo-950/30 ring-1 ring-indigo-500/50'
                          : 'border-slate-800 bg-slate-900/50 hover:border-slate-700'
                      }`}
                    >
                      <div className="text-sm font-semibold text-white">{algo.name}</div>
                      <div className="text-xs text-slate-400 mt-1">{algo.desc}</div>
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <div className="text-xs font-semibold text-slate-400 mb-2">Supervised - Classification</div>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
                  {[
                    { id: 'logistic_regression', name: 'Logistic Regression', desc: 'Sigmoid & Cross-Entropy' },
                    { id: 'decision_tree_classifier', name: 'Decision Tree Classifier', desc: 'Gini / Entropy CART' },
                    { id: 'knn_classifier', name: 'KNN Classifier', desc: 'Instance neighborhood voting' },
                    { id: 'random_forest_classifier', name: 'Random Forest Classifier', desc: 'Bagged feature ensemble' },
                  ].map((algo) => (
                    <button
                      key={algo.id}
                      onClick={() => setSelectedAlgo(algo.id as AlgorithmId)}
                      className={`text-left p-3.5 rounded-xl border transition-all ${
                        selectedAlgo === algo.id
                          ? 'border-indigo-500 bg-indigo-950/30 ring-1 ring-indigo-500/50'
                          : 'border-slate-800 bg-slate-900/50 hover:border-slate-700'
                      }`}
                    >
                      <div className="text-sm font-semibold text-white">{algo.name}</div>
                      <div className="text-xs text-slate-400 mt-1">{algo.desc}</div>
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <div className="text-xs font-semibold text-slate-400 mb-2">Unsupervised - Clustering</div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  {[
                    { id: 'kmeans_clustering', name: 'K-Means Clustering', desc: "Lloyd's iterative centroid convergence & Voronoi partitions" },
                  ].map((algo) => (
                    <button
                      key={algo.id}
                      onClick={() => setSelectedAlgo(algo.id as AlgorithmId)}
                      className={`text-left p-3.5 rounded-xl border transition-all ${
                        selectedAlgo === algo.id
                          ? 'border-indigo-500 bg-indigo-950/30 ring-1 ring-indigo-500/50'
                          : 'border-slate-800 bg-slate-900/50 hover:border-slate-700'
                      }`}
                    >
                      <div className="text-sm font-semibold text-white">{algo.name}</div>
                      <div className="text-xs text-slate-400 mt-1">{algo.desc}</div>
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Hyperparameter Controls */}
          <div className="border border-slate-800 rounded-xl bg-slate-950 p-6 space-y-6">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <h2 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
                <Sliders className="w-4 h-4 text-indigo-400" />
                Hyperparameter Configuration ({currentCompat.name})
              </h2>
              <span className="text-xs font-mono text-slate-500">Validation: Strict Range</span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
              {/* Linear / Logistic */}
              {(selectedAlgo === 'linear_regression' || selectedAlgo === 'logistic_regression') && (
                <>
                  <div>
                    <div className="flex justify-between text-xs font-medium text-slate-300 mb-1.5">
                      <span>Learning Rate (α)</span>
                      <span className="font-mono text-indigo-400">{hyperparams.learningRate}</span>
                    </div>
                    <input
                      type="range"
                      min="0.001"
                      max="0.5"
                      step="0.005"
                      value={hyperparams.learningRate}
                      onChange={(e) => setHyperparams({ ...hyperparams, learningRate: Number(e.target.value) })}
                      className="w-full accent-indigo-500 cursor-pointer"
                    />
                  </div>

                  <div>
                    <div className="flex justify-between text-xs font-medium text-slate-300 mb-1.5">
                      <span>Epochs / Iterations</span>
                      <span className="font-mono text-indigo-400">{hyperparams.epochs}</span>
                    </div>
                    <input
                      type="range"
                      min="10"
                      max="300"
                      step="10"
                      value={hyperparams.epochs}
                      onChange={(e) => setHyperparams({ ...hyperparams, epochs: Number(e.target.value) })}
                      className="w-full accent-indigo-500 cursor-pointer"
                    />
                  </div>
                </>
              )}

              {selectedAlgo === 'logistic_regression' && (
                <div>
                  <div className="flex justify-between text-xs font-medium text-slate-300 mb-1.5">
                    <span>L2 Regularization Strength (C)</span>
                    <span className="font-mono text-indigo-400">{hyperparams.regularizationC}</span>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="10.0"
                    step="0.1"
                    value={hyperparams.regularizationC}
                    onChange={(e) => setHyperparams({ ...hyperparams, regularizationC: Number(e.target.value) })}
                    className="w-full accent-indigo-500 cursor-pointer"
                  />
                </div>
              )}

              {/* Tree / Forest */}
              {(selectedAlgo.includes('decision_tree') || selectedAlgo.includes('random_forest')) && (
                <>
                  <div>
                    <div className="flex justify-between text-xs font-medium text-slate-300 mb-1.5">
                      <span>Max Depth</span>
                      <span className="font-mono text-indigo-400">{hyperparams.maxDepth}</span>
                    </div>
                    <input
                      type="range"
                      min="1"
                      max="8"
                      value={hyperparams.maxDepth}
                      onChange={(e) => setHyperparams({ ...hyperparams, maxDepth: Number(e.target.value) })}
                      className="w-full accent-indigo-500 cursor-pointer"
                    />
                  </div>

                  <div>
                    <div className="flex justify-between text-xs font-medium text-slate-300 mb-1.5">
                      <span>Min Samples Split</span>
                      <span className="font-mono text-indigo-400">{hyperparams.minSamplesSplit}</span>
                    </div>
                    <input
                      type="range"
                      min="2"
                      max="10"
                      value={hyperparams.minSamplesSplit}
                      onChange={(e) => setHyperparams({ ...hyperparams, minSamplesSplit: Number(e.target.value) })}
                      className="w-full accent-indigo-500 cursor-pointer"
                    />
                  </div>
                </>
              )}

              {selectedAlgo.includes('random_forest') && (
                <div>
                  <div className="flex justify-between text-xs font-medium text-slate-300 mb-1.5">
                    <span>Number of Estimators (Trees)</span>
                    <span className="font-mono text-indigo-400">{hyperparams.nEstimators}</span>
                  </div>
                  <input
                    type="range"
                    min="3"
                    max="25"
                    value={hyperparams.nEstimators}
                    onChange={(e) => setHyperparams({ ...hyperparams, nEstimators: Number(e.target.value) })}
                    className="w-full accent-indigo-500 cursor-pointer"
                  />
                </div>
              )}

              {/* KNN */}
              {selectedAlgo === 'knn_classifier' && (
                <>
                  <div>
                    <div className="flex justify-between text-xs font-medium text-slate-300 mb-1.5">
                      <span>Neighbors (k)</span>
                      <span className="font-mono text-indigo-400">{hyperparams.nNeighbors}</span>
                    </div>
                    <input
                      type="range"
                      min="1"
                      max="15"
                      value={hyperparams.nNeighbors}
                      onChange={(e) => setHyperparams({ ...hyperparams, nNeighbors: Number(e.target.value) })}
                      className="w-full accent-indigo-500 cursor-pointer"
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-slate-300 mb-1.5">Distance Metric</label>
                    <select
                      value={hyperparams.distanceMetric}
                      onChange={(e) => setHyperparams({ ...hyperparams, distanceMetric: e.target.value as any })}
                      className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-200"
                    >
                      <option value="euclidean">Euclidean (L2 Norm)</option>
                      <option value="manhattan">Manhattan (L1 Norm)</option>
                    </select>
                  </div>
                </>
              )}

              {/* K-Means */}
              {selectedAlgo === 'kmeans_clustering' && (
                <>
                  <div>
                    <div className="flex justify-between text-xs font-medium text-slate-300 mb-1.5">
                      <span>Number of Clusters (k)</span>
                      <span className="font-mono text-indigo-400">{hyperparams.nClusters}</span>
                    </div>
                    <input
                      type="range"
                      min="2"
                      max="7"
                      value={hyperparams.nClusters}
                      onChange={(e) => setHyperparams({ ...hyperparams, nClusters: Number(e.target.value) })}
                      className="w-full accent-indigo-500 cursor-pointer"
                    />
                  </div>

                  <div>
                    <div className="flex justify-between text-xs font-medium text-slate-300 mb-1.5">
                      <span>Max Iterations</span>
                      <span className="font-mono text-indigo-400">{hyperparams.maxIterations}</span>
                    </div>
                    <input
                      type="range"
                      min="5"
                      max="40"
                      value={hyperparams.maxIterations}
                      onChange={(e) => setHyperparams({ ...hyperparams, maxIterations: Number(e.target.value) })}
                      className="w-full accent-indigo-500 cursor-pointer"
                    />
                  </div>
                </>
              )}
            </div>
          </div>
        </div>

        {/* Right Col: Active Model Summary Card */}
        <div className="space-y-6">
          <div className="border border-slate-800 rounded-xl bg-slate-950 p-6 space-y-5">
            <h3 className="text-sm font-bold text-white uppercase tracking-wider">
              Model Training Status
            </h3>

            {activeArtifact ? (
              <div className="space-y-4">
                <div className="flex items-center gap-2 text-xs font-medium text-emerald-400">
                  <CheckCircle className="w-4 h-4" />
                  <span>Model fitted &amp; stored in memory</span>
                </div>

                <div className="bg-slate-900/60 p-4 rounded-xl border border-slate-800 space-y-2 text-xs">
                  <div className="text-slate-400">Model Name:</div>
                  <div className="font-semibold text-white">{activeArtifact.name}</div>
                  <div className="text-slate-400 mt-2">Training Duration:</div>
                  <div className="font-mono text-indigo-400 tabular-nums">
                    {activeArtifact.trainingDurationMs} ms
                  </div>
                </div>

                {/* Quick Metrics display */}
                <div className="space-y-2">
                  <div className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                    Held-Out Test Metrics
                  </div>

                  {activeArtifact.metrics.task === 'regression' && (
                    <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                      <div className="bg-slate-900 p-2.5 rounded-lg border border-slate-800">
                        <div className="text-slate-500">R² Score</div>
                        <div className="text-base font-bold text-indigo-400 tabular-nums">
                          {activeArtifact.metrics.metrics.r2}
                        </div>
                      </div>
                      <div className="bg-slate-900 p-2.5 rounded-lg border border-slate-800">
                        <div className="text-slate-500">RMSE</div>
                        <div className="text-base font-bold text-slate-200 tabular-nums">
                          {activeArtifact.metrics.metrics.rmse}
                        </div>
                      </div>
                    </div>
                  )}

                  {activeArtifact.metrics.task === 'classification' && (
                    <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                      <div className="bg-slate-900 p-2.5 rounded-lg border border-slate-800">
                        <div className="text-slate-500">Accuracy</div>
                        <div className="text-base font-bold text-emerald-400 tabular-nums">
                          {(activeArtifact.metrics.metrics.accuracy * 100).toFixed(1)}%
                        </div>
                      </div>
                      <div className="bg-slate-900 p-2.5 rounded-lg border border-slate-800">
                        <div className="text-slate-500">F1-Score</div>
                        <div className="text-base font-bold text-indigo-400 tabular-nums">
                          {activeArtifact.metrics.metrics.f1Score}
                        </div>
                      </div>
                    </div>
                  )}

                  {activeArtifact.metrics.task === 'clustering' && (
                    <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                      <div className="bg-slate-900 p-2.5 rounded-lg border border-slate-800">
                        <div className="text-slate-500">Silhouette</div>
                        <div className="text-base font-bold text-purple-400 tabular-nums">
                          {activeArtifact.metrics.metrics.silhouetteScore}
                        </div>
                      </div>
                      <div className="bg-slate-900 p-2.5 rounded-lg border border-slate-800">
                        <div className="text-slate-500">Inertia (WCSS)</div>
                        <div className="text-base font-bold text-slate-200 tabular-nums">
                          {activeArtifact.metrics.metrics.inertia}
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                <button
                  onClick={onProceedToSimulation}
                  className="w-full flex items-center justify-center gap-2 py-2.5 px-4 text-xs font-semibold text-white bg-indigo-600 hover:bg-indigo-500 rounded-lg transition-colors"
                >
                  Inspect Simulation &amp; Visuals
                  <ArrowRight className="w-3.5 h-3.5" />
                </button>
              </div>
            ) : (
              <div className="text-center py-8 text-xs text-slate-500 space-y-2">
                <Clock className="w-8 h-8 mx-auto text-slate-600 mb-2" />
                <p>No model trained for this session yet.</p>
                <p>Click "Train {currentCompat.name}" above to begin model execution.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
