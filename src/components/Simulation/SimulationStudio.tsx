import React, { useState, useEffect, useRef } from 'react';
import {
  Play,
  Pause,
  RotateCcw,
  Volume2,
  VolumeX,
  Layers,
  Sparkles,
  Info,
  ChevronRight,
  ChevronLeft,
  Sliders,
  CheckCircle,
} from 'lucide-react';
import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  Tooltip,
  CartesianGrid,
  Line,
  ComposedChart,
  BarChart,
  Bar,
  Legend,
} from 'recharts';
import { ModelArtifact, PreprocessedData, TreeNode } from '../../ml/types.ts';
import { computePCA2D } from '../../ml/pca.ts';
import { traceDecisionPath } from '../../ml/algorithms/decisionTree.ts';
import { createSpeechController, SpeechController } from '../../ml/speech.ts';

interface SimulationStudioProps {
  artifact: ModelArtifact;
  preprocessedData: PreprocessedData;
  onNavigateToExperiments: () => void;
}

export const SimulationStudio: React.FC<SimulationStudioProps> = ({
  artifact,
  preprocessedData,
  onNavigateToExperiments,
}) => {
  // Dimensionality projection state
  const featureNames = preprocessedData.featureNames;
  const isHighDim = featureNames.length > 2;
  const [viewMode, setViewMode] = useState<'pca' | 'features'>('features');
  const [featXIdx, setFeatXIdx] = useState<number>(0);
  const [featYIdx, setFeatYIdx] = useState<number>(Math.min(1, featureNames.length - 1));

  // K-Means animation step state
  const [kmeansStep, setKmeansStep] = useState<number>(0);
  const [isPlayingKmeans, setIsPlayingKmeans] = useState<boolean>(false);

  // Decision Tree active sample inspection
  const [selectedSampleIdx, setSelectedSampleIdx] = useState<number>(0);
  const [selectedTreeIdx, setSelectedTreeIdx] = useState<number>(0); // for Random Forest

  // Audio Speech Synthesis state
  const speechCtrlRef = useRef<SpeechController | null>(null);
  const [isSpeaking, setIsSpeaking] = useState<boolean>(false);

  useEffect(() => {
    speechCtrlRef.current = createSpeechController();
    return () => {
      speechCtrlRef.current?.stop();
    };
  }, []);

  // Compute 2D coordinates for scatter
  const trainX = preprocessedData.train.X;
  const trainY = preprocessedData.train.y;
  const testX = preprocessedData.test.X;
  const testY = preprocessedData.test.y;

  const pcaResult = React.useMemo(() => {
    return computePCA2D(trainX);
  }, [trainX]);

  // Handle K-Means Play/Pause interval
  useEffect(() => {
    let timer: any;
    if (isPlayingKmeans && artifact.kmeansModel) {
      const maxSteps = artifact.kmeansModel.iterationCentroids.length - 1;
      timer = setInterval(() => {
        setKmeansStep((prev) => {
          if (prev >= maxSteps) {
            setIsPlayingKmeans(false);
            return prev;
          }
          return prev + 1;
        });
      }, 800);
    }
    return () => clearInterval(timer);
  }, [isPlayingKmeans, artifact.kmeansModel]);

  // Read explanations aloud
  const handleToggleSpeech = () => {
    if (!speechCtrlRef.current) return;
    if (isSpeaking) {
      speechCtrlRef.current.stop();
      setIsSpeaking(false);
    } else {
      const textToRead = artifact.explanations
        .map((exp) => `${exp.title}. ${exp.concept}. ${exp.concreteCalculation}`)
        .join('. ');
      speechCtrlRef.current.speak(textToRead, () => setIsSpeaking(false));
      setIsSpeaking(true);
    }
  };

  // Build scatter points based on projection
  const scatterPoints = React.useMemo(() => {
    const points: any[] = [];
    const n = Math.min(trainX.length, 300);

    for (let i = 0; i < n; i++) {
      let xVal = 0;
      let yVal = 0;

      if (viewMode === 'pca' && isHighDim) {
        xVal = pcaResult.projectedX[i]?.[0] || 0;
        yVal = pcaResult.projectedX[i]?.[1] || 0;
      } else {
        xVal = trainX[i][featXIdx] || 0;
        yVal =
          artifact.taskType === 'regression'
            ? Number(trainY[i]) || 0
            : trainX[i][featYIdx] || 0;
      }

      let category = String(trainY[i] || '0');
      if (artifact.kmeansModel && artifact.kmeansModel.iterationCentroids) {
        // Find closest centroid at current animation step
        const stepCentroids = artifact.kmeansModel.iterationCentroids[kmeansStep] || artifact.kmeansModel.centroids;
        let bestC = 0;
        let minDist = Infinity;
        for (let c = 0; c < stepCentroids.length; c++) {
          let d = 0;
          for (let f = 0; f < trainX[i].length; f++) {
            const diff = trainX[i][f] - stepCentroids[c][f];
            d += diff * diff;
          }
          if (d < minDist) {
            minDist = d;
            bestC = c;
          }
        }
        category = `Cluster ${bestC + 1}`;
      }

      points.push({
        x: Number(xVal.toFixed(2)),
        y: Number(yVal.toFixed(2)),
        category,
        actualTarget: trainY[i],
        rawIdx: i,
      });
    }
    return points;
  }, [
    trainX,
    trainY,
    viewMode,
    featXIdx,
    featYIdx,
    isHighDim,
    pcaResult,
    artifact,
    kmeansStep,
  ]);

  // Calculate regression line points for Linear Regression 2D view
  const regressionLineData = React.useMemo(() => {
    if (artifact.algorithm !== 'linear_regression' || !artifact.linearModel) return [];
    const xs = scatterPoints.map((p) => p.x);
    if (xs.length === 0) return [];
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);

    const w = artifact.linearModel.coefficients[featXIdx] || 0;
    const b = artifact.linearModel.intercept;

    return [
      { x: minX, y: Number((w * minX + b).toFixed(2)) },
      { x: maxX, y: Number((w * maxX + b).toFixed(2)) },
    ];
  }, [artifact, scatterPoints, featXIdx]);

  // Decision Tree active sample path
  const currentTree =
    artifact.treeModel?.root ||
    (artifact.forestModel ? artifact.forestModel.trees[selectedTreeIdx] : null);

  const samplePath = React.useMemo(() => {
    if (!currentTree || !trainX[selectedSampleIdx]) return [];
    return traceDecisionPath(trainX[selectedSampleIdx], currentTree);
  }, [currentTree, trainX, selectedSampleIdx]);

  // Distinct Color Palette for Clusters/Classes
  const categoryColors = [
    '#6366f1', // Indigo
    '#10b981', // Emerald
    '#f59e0b', // Amber
    '#ec4899', // Pink
    '#38bdf8', // Sky
    '#a855f7', // Purple
  ];

  // Group points by category for multi-color scatter
  const groupedPoints = React.useMemo(() => {
    const groups: Record<string, any[]> = {};
    for (const pt of scatterPoints) {
      if (!groups[pt.category]) groups[pt.category] = [];
      groups[pt.category].push(pt);
    }
    return groups;
  }, [scatterPoints]);

  return (
    <div className="space-y-8 max-w-7xl mx-auto py-6 px-4">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-2xl font-bold tracking-tight text-white">04. Interactive Simulation &amp; Viz</h1>
            <span className="text-xs font-mono text-indigo-400 bg-indigo-950/60 px-2 py-0.5 rounded border border-indigo-800/80">
              {artifact.name}
            </span>
          </div>
          <p className="text-sm text-slate-400 mt-1">
            Observe model decisions, explore the feature hyperplane, inspect tree nodes, and trace sample predictions.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={handleToggleSpeech}
            className={`flex items-center gap-2 px-3.5 py-1.5 text-xs font-medium rounded-lg border transition-colors ${
              isSpeaking
                ? 'bg-amber-600/20 border-amber-500 text-amber-300'
                : 'bg-slate-900 border-slate-800 text-slate-300 hover:text-white'
            }`}
          >
            {isSpeaking ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
            <span>{isSpeaking ? 'Mute Narration' : 'Read Aloud (TTS)'}</span>
          </button>

          <button
            onClick={onNavigateToExperiments}
            className="flex items-center gap-2 px-4 py-2 text-xs font-semibold text-white bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors border border-slate-700"
          >
            View in Experiment Tracker
          </button>
        </div>
      </div>

      {/* Projection / Feature Plane Toolbar */}
      <div className="flex items-center justify-between flex-wrap gap-4 bg-slate-950 p-4 rounded-xl border border-slate-800 text-xs">
        <div className="flex items-center gap-3">
          <span className="text-slate-400 font-medium">Coordinate Projection:</span>
          {isHighDim && (
            <div className="flex items-center bg-slate-900 p-1 rounded-lg border border-slate-800">
              <button
                onClick={() => setViewMode('features')}
                className={`px-2.5 py-1 rounded transition-colors ${
                  viewMode === 'features' ? 'bg-indigo-600 text-white font-semibold' : 'text-slate-400'
                }`}
              >
                2 Features
              </button>
              <button
                onClick={() => setViewMode('pca')}
                className={`px-2.5 py-1 rounded transition-colors ${
                  viewMode === 'pca' ? 'bg-indigo-600 text-white font-semibold' : 'text-slate-400'
                }`}
              >
                PCA 2D Projection
              </button>
            </div>
          )}
        </div>

        {viewMode === 'features' ? (
          <div className="flex items-center gap-4 flex-wrap">
            <div className="flex items-center gap-2">
              <span className="text-slate-400">X Axis:</span>
              <select
                value={featXIdx}
                onChange={(e) => setFeatXIdx(Number(e.target.value))}
                className="bg-slate-900 border border-slate-800 rounded px-2.5 py-1 text-slate-200"
              >
                {featureNames.map((f, i) => (
                  <option key={f} value={i}>
                    {f}
                  </option>
                ))}
              </select>
            </div>

            {artifact.taskType !== 'regression' && (
              <div className="flex items-center gap-2">
                <span className="text-slate-400">Y Axis:</span>
                <select
                  value={featYIdx}
                  onChange={(e) => setFeatYIdx(Number(e.target.value))}
                  className="bg-slate-900 border border-slate-800 rounded px-2.5 py-1 text-slate-200"
                >
                  {featureNames.map((f, i) => (
                    <option key={f} value={i}>
                      {f}
                    </option>
                  ))}
                </select>
              </div>
            )}
          </div>
        ) : (
          <div className="font-mono text-indigo-400">
            PCA Explained Variance: PC1: {(pcaResult.explainedVarianceRatio[0] * 100).toFixed(1)}% · PC2: {(pcaResult.explainedVarianceRatio[1] * 100).toFixed(1)}%
          </div>
        )}
      </div>

      {/* Main Algorithm Visualizer Stage */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left 2 Cols: Interactive Canvas / Chart */}
        <div className="lg:col-span-2 border border-slate-800 rounded-xl bg-slate-950 p-6 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <h2 className="text-sm font-bold text-white uppercase tracking-wider">
              {artifact.algorithm === 'linear_regression' && 'Fitted Regression Plane & Residual Line'}
              {artifact.algorithm.includes('decision_tree') && 'Hierarchical Tree Structure & Split Nodes'}
              {artifact.algorithm.includes('random_forest') && 'Ensemble Trees & Feature Importances'}
              {artifact.algorithm === 'kmeans_clustering' && "K-Means Lloyd's Step-by-Step Clustering"}
              {(artifact.algorithm === 'logistic_regression' || artifact.algorithm === 'knn_classifier') &&
                'Classification Feature Plane'}
            </h2>

            {/* K-Means Playback Scrubber */}
            {artifact.kmeansModel && (
              <div className="flex items-center gap-2">
                <button
                  disabled={kmeansStep === 0}
                  onClick={() => setKmeansStep((s) => Math.max(0, s - 1))}
                  className="p-1.5 rounded bg-slate-900 border border-slate-800 text-slate-300 disabled:opacity-40"
                  title="Previous Step"
                >
                  <ChevronLeft className="w-4 h-4" />
                </button>
                <button
                  onClick={() => setIsPlayingKmeans(!isPlayingKmeans)}
                  className="flex items-center gap-1.5 px-3 py-1 text-xs font-semibold bg-indigo-600 hover:bg-indigo-500 text-white rounded"
                >
                  {isPlayingKmeans ? <Pause className="w-3.5 h-3.5 fill-current" /> : <Play className="w-3.5 h-3.5 fill-current" />}
                  <span>Step {kmeansStep + 1} / {artifact.kmeansModel.iterationCentroids.length}</span>
                </button>
                <button
                  disabled={kmeansStep >= artifact.kmeansModel.iterationCentroids.length - 1}
                  onClick={() => setKmeansStep((s) => Math.min(artifact.kmeansModel!.iterationCentroids.length - 1, s + 1))}
                  className="p-1.5 rounded bg-slate-900 border border-slate-800 text-slate-300 disabled:opacity-40"
                  title="Next Step"
                >
                  <ChevronRight className="w-4 h-4" />
                </button>
                <button
                  onClick={() => {
                    setIsPlayingKmeans(false);
                    setKmeansStep(0);
                  }}
                  className="p-1.5 rounded bg-slate-900 border border-slate-800 text-slate-300"
                  title="Reset Iteration"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                </button>
              </div>
            )}
          </div>

          {/* 1. SCATTER / LINE PLOT (Linear Regression, Logistic, KNN, K-Means) */}
          {!artifact.algorithm.includes('decision_tree') && (
            <div className="h-80 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis
                    type="number"
                    dataKey="x"
                    name={viewMode === 'pca' ? 'PC1' : featureNames[featXIdx]}
                    stroke="#64748b"
                    tick={{ fill: '#64748b', fontSize: 11 }}
                  />
                  <YAxis
                    type="number"
                    dataKey="y"
                    name={
                      artifact.taskType === 'regression'
                        ? artifact.targetName
                        : viewMode === 'pca'
                        ? 'PC2'
                        : featureNames[featYIdx]
                    }
                    stroke="#64748b"
                    tick={{ fill: '#64748b', fontSize: 11 }}
                  />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: 8, fontSize: 12 }}
                  />
                  <Legend wrapperStyle={{ fontSize: 11, paddingTop: 10 }} />

                  {/* Scatter Series for each category */}
                  {Object.entries(groupedPoints).map(([cat, pts], idx) => (
                    <Scatter
                      key={cat}
                      name={cat}
                      data={pts}
                      fill={categoryColors[idx % categoryColors.length]}
                    />
                  ))}

                  {/* Fitted Regression Line */}
                  {regressionLineData.length > 0 && (
                    <Line
                      type="monotone"
                      dataKey="y"
                      data={regressionLineData}
                      stroke="#f43f5e"
                      strokeWidth={3}
                      dot={false}
                      name="Fitted OLS Hyperplane"
                    />
                  )}
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* 2. INTERACTIVE DECISION TREE RENDERER */}
          {currentTree && (
            <div className="space-y-4">
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-400">
                  Select sample to highlight active branch traversal path:
                </span>
                <div className="flex items-center gap-2">
                  <span className="font-mono text-indigo-400">Sample #{selectedSampleIdx + 1}</span>
                  <input
                    type="range"
                    min="0"
                    max={Math.min(trainX.length - 1, 40)}
                    value={selectedSampleIdx}
                    onChange={(e) => setSelectedSampleIdx(Number(e.target.value))}
                    className="w-32 accent-indigo-500 cursor-pointer"
                  />
                </div>
              </div>

              {/* Render hierarchical tree boxes */}
              <div className="overflow-x-auto p-4 bg-slate-900/60 rounded-xl border border-slate-800 flex justify-center">
                <TreeHierarchyViewer
                  node={currentTree}
                  activeSample={trainX[selectedSampleIdx]}
                  samplePathNodeIds={samplePath.map((p) => p.nodeId)}
                />
              </div>
            </div>
          )}

          {/* Linear Equation Callout for Linear/Logistic */}
          {artifact.linearModel && (
            <div className="p-3.5 bg-slate-900 rounded-lg border border-slate-800 text-xs font-mono text-indigo-300 flex items-center justify-between">
              <div>
                <span className="text-slate-500 mr-2">Model Equation:</span>
                <strong className="text-white">{artifact.linearModel.equation}</strong>
              </div>
              <div className="text-slate-400">
                Intercept: <strong className="text-white">{artifact.linearModel.intercept}</strong>
              </div>
            </div>
          )}

          {/* Feature Importance Bar Chart for Random Forest */}
          {artifact.forestModel && (
            <div className="space-y-2 mt-4">
              <div className="text-xs font-semibold text-slate-300">
                Random Forest Feature Importance (MDI)
              </div>
              <div className="h-44 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={artifact.forestModel.featureImportances}
                    layout="vertical"
                    margin={{ top: 5, right: 30, left: 60, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis type="number" stroke="#64748b" tick={{ fill: '#64748b', fontSize: 10 }} />
                    <YAxis
                      dataKey="feature"
                      type="category"
                      stroke="#64748b"
                      tick={{ fill: '#94a3b8', fontSize: 10 }}
                    />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: 8, fontSize: 11 }}
                    />
                    <Bar dataKey="importance" fill="#6366f1" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </div>

        {/* Right Col: Written Step-by-Step Explanation & Sample Trace */}
        <div className="space-y-6">
          {/* Sample Trace Inspector for Decision Trees */}
          {samplePath.length > 0 && (
            <div className="border border-slate-800 rounded-xl bg-slate-950 p-5 space-y-3">
              <h3 className="text-xs font-bold text-white uppercase tracking-wider">
                Sample #{selectedSampleIdx + 1} Decision Path Trace
              </h3>
              <div className="space-y-2 text-xs font-mono">
                {samplePath.map((step, idx) => (
                  <div
                    key={idx}
                    className={`p-2 rounded-lg border flex items-center justify-between ${
                      step.satisfied
                        ? 'bg-indigo-950/40 border-indigo-500/60 text-indigo-200'
                        : 'bg-slate-900 border-slate-800 text-slate-400'
                    }`}
                  >
                    <span>{step.rule}</span>
                    <span className="text-[10px] text-slate-500">n={step.samples}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Dynamically Generated Step-by-Step Educational Explanation */}
          <div className="border border-slate-800 rounded-xl bg-slate-950 p-5 space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <h3 className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-indigo-400" />
                Algorithm Step-by-Step Learning Guide
              </h3>
              <span className="text-[10px] text-slate-500 font-mono">Formula Grounded</span>
            </div>

            <div className="space-y-4">
              {artifact.explanations.map((step) => (
                <div key={step.stepNumber} className="space-y-1.5 text-xs">
                  <div className="flex items-center gap-2 font-semibold text-slate-200">
                    <span className="w-5 h-5 rounded-full bg-indigo-950 text-indigo-400 border border-indigo-800 flex items-center justify-center text-[10px] font-mono shrink-0">
                      {step.stepNumber}
                    </span>
                    <span>{step.title}</span>
                  </div>
                  <p className="text-slate-400 leading-relaxed pl-7">{step.concept}</p>
                  {step.formula && (
                    <div className="ml-7 p-2 bg-slate-900/90 rounded border border-slate-800 font-mono text-[11px] text-indigo-300">
                      {step.formula}
                    </div>
                  )}
                  <div className="ml-7 text-[11px] text-slate-500 font-mono">
                    {step.concreteCalculation}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// -------------------------------------------------------------
// Interactive Recursive Tree Component
// -------------------------------------------------------------
const TreeHierarchyViewer: React.FC<{
  node: TreeNode;
  activeSample?: number[];
  samplePathNodeIds: string[];
}> = ({ node, activeSample, samplePathNodeIds }) => {
  const isTraversed = samplePathNodeIds.includes(node.id);

  if (node.isLeaf) {
    return (
      <div
        className={`flex flex-col items-center p-3 rounded-xl border text-xs font-mono transition-all ${
          isTraversed
            ? 'bg-emerald-950/60 border-emerald-500 text-emerald-200 ring-2 ring-emerald-500/30'
            : 'bg-slate-900 border-slate-800 text-slate-300'
        }`}
      >
        <div className="font-bold">Leaf: {String(node.prediction)}</div>
        <div className="text-[10px] text-slate-500 mt-1">samples: {node.samples}</div>
        <div className="text-[10px] text-slate-500">impurity: {node.impurity}</div>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center space-y-3">
      {/* Node box */}
      <div
        className={`p-3 rounded-xl border text-xs font-mono text-center transition-all min-w-36 ${
          isTraversed
            ? 'bg-indigo-950/70 border-indigo-400 text-indigo-100 ring-2 ring-indigo-500/30'
            : 'bg-slate-900 border-slate-800 text-slate-300'
        }`}
      >
        <div className="font-semibold text-white">
          {node.featureName} &le; {node.threshold}
        </div>
        <div className="text-[10px] text-slate-400 mt-1">
          samples: {node.samples} · imp: {node.impurity}
        </div>
      </div>

      {/* Children connector branches */}
      <div className="flex items-start gap-8 pt-2">
        {node.leftChild && (
          <div className="flex flex-col items-center">
            <span className="text-[10px] font-mono text-slate-500 mb-1">True &darr;</span>
            <TreeHierarchyViewer
              node={node.leftChild}
              activeSample={activeSample}
              samplePathNodeIds={samplePathNodeIds}
            />
          </div>
        )}

        {node.rightChild && (
          <div className="flex flex-col items-center">
            <span className="text-[10px] font-mono text-slate-500 mb-1">False &darr;</span>
            <TreeHierarchyViewer
              node={node.rightChild}
              activeSample={activeSample}
              samplePathNodeIds={samplePathNodeIds}
            />
          </div>
        )}
      </div>
    </div>
  );
};
