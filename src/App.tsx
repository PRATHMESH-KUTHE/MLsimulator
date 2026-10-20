import React, { useState, useEffect } from 'react';
import { ActiveTab, Navbar } from './components/Navbar.tsx';
import { DatasetStudio } from './components/DatasetStudio/DatasetStudio.tsx';
import { PreprocessingStudio } from './components/Preprocessing/PreprocessingStudio.tsx';
import { AlgorithmLab } from './components/AlgorithmLab/AlgorithmLab.tsx';
import { SimulationStudio } from './components/Simulation/SimulationStudio.tsx';
import { ExperimentTracker } from './components/Experiments/ExperimentTracker.tsx';
import { ModelRegistry } from './components/ModelRegistry/ModelRegistry.tsx';
import { MLOpsOverview } from './components/MLOpsOverview/MLOpsOverview.tsx';
import { Dataset, Experiment, ModelArtifact, PreprocessedData, PreprocessingConfig } from './ml/types.ts';
import { createIrisDataset } from './ml/datasets.ts';
import { runPreprocessingPipeline } from './ml/preprocessing.ts';
import { trainModel } from './ml/trainer.ts';
import {
  getStoredExperiments,
  saveExperiment,
  getStoredModels,
  saveModelArtifact,
  updateModelStatus,
  clearExperimentHistory,
} from './ml/storage.ts';

export default function App() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('studio');

  // Initialize with classic Iris benchmark
  const [activeDataset, setActiveDataset] = useState<Dataset>(() => createIrisDataset());
  const [selectedTarget, setSelectedTarget] = useState<string>('species');
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>([
    'sepal_length',
    'sepal_width',
    'petal_length',
    'petal_width',
  ]);

  // Preprocessing state
  const [preprocessingConfig, setPreprocessingConfig] = useState<PreprocessingConfig>({
    missingStrategy: 'mean',
    removeDuplicates: true,
    encodingStrategy: 'onehot',
    scalingStrategy: 'standard',
    outlierTreatment: 'none',
    featureSelectionVarianceThreshold: 0,
    testSplitRatio: 0.2,
    randomSeed: 42,
    selectedFeatures: ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
    targetColumn: 'species',
  });

  const [preprocessedData, setPreprocessedData] = useState<PreprocessedData | null>(() => {
    try {
      const initDs = createIrisDataset();
      return runPreprocessingPipeline(
        initDs.data,
        {
          missingStrategy: 'mean',
          removeDuplicates: true,
          encodingStrategy: 'onehot',
          scalingStrategy: 'standard',
          outlierTreatment: 'none',
          featureSelectionVarianceThreshold: 0,
          testSplitRatio: 0.2,
          randomSeed: 42,
          selectedFeatures: ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
          targetColumn: 'species',
        },
        'classification'
      );
    } catch {
      return null;
    }
  });

  // Stored state
  const [experiments, setExperiments] = useState<Experiment[]>(() => getStoredExperiments());
  const [models, setModels] = useState<ModelArtifact[]>(() => getStoredModels());
  const [activeArtifact, setActiveArtifact] = useState<ModelArtifact | null>(null);

  // Train initial baseline model on first load if no models exist
  useEffect(() => {
    if (preprocessedData && models.length === 0) {
      try {
        const { artifact, experiment } = trainModel(
          'decision_tree_classifier',
          preprocessedData,
          activeDataset.name,
          activeDataset.fingerprint,
          { maxDepth: 4, minSamplesSplit: 2 }
        );
        saveModelArtifact(artifact);
        saveExperiment(experiment);
        setActiveArtifact(artifact);
        setModels([artifact]);
        setExperiments([experiment]);
      } catch (err) {
        console.error('Initial baseline training error:', err);
      }
    } else if (models.length > 0 && !activeArtifact) {
      setActiveArtifact(models[0]);
    }
  }, []);

  // Handle dataset selection
  const handleSelectDataset = (dataset: Dataset) => {
    setActiveDataset(dataset);
    const target = dataset.suggestedTarget || '';
    const feats = dataset.columns.filter((c) => c !== target);
    setSelectedTarget(target);
    setSelectedFeatures(feats);

    const newConfig: PreprocessingConfig = {
      ...preprocessingConfig,
      targetColumn: target,
      selectedFeatures: feats,
    };
    setPreprocessingConfig(newConfig);

    try {
      const prep = runPreprocessingPipeline(
        dataset.data,
        newConfig,
        dataset.suggestedTask || 'classification'
      );
      setPreprocessedData(prep);
    } catch (err) {
      console.warn('Initial preprocessing failed for new dataset:', err);
      setPreprocessedData(null);
    }
  };

  // Handle target/feature selection
  const handleUpdateTargetAndFeatures = (target: string, feats: string[]) => {
    setSelectedTarget(target);
    setSelectedFeatures(feats);
    const updated = {
      ...preprocessingConfig,
      targetColumn: target,
      selectedFeatures: feats,
    };
    setPreprocessingConfig(updated);
    try {
      const prep = runPreprocessingPipeline(
        activeDataset.data,
        updated,
        activeDataset.suggestedTask || (target ? 'classification' : 'clustering')
      );
      setPreprocessedData(prep);
    } catch (err) {
      console.warn('Re-preprocessing error:', err);
    }
  };

  // Execute Preprocessing Pipeline
  const handleExecutePipeline = () => {
    try {
      const task = activeDataset.suggestedTask || (selectedTarget ? 'classification' : 'clustering');
      const prep = runPreprocessingPipeline(activeDataset.data, preprocessingConfig, task);
      setPreprocessedData(prep);
    } catch (err: any) {
      alert(`Preprocessing error: ${err.message}`);
    }
  };

  // Handle Model Trained
  const handleModelTrained = (artifact: ModelArtifact, experiment: Experiment) => {
    saveModelArtifact(artifact);
    saveExperiment(experiment);
    setActiveArtifact(artifact);
    setModels(getStoredModels());
    setExperiments(getStoredExperiments());
  };

  // Handle Rerun Experiment
  const handleRerunExperiment = (exp: Experiment) => {
    // Restore config
    setPreprocessingConfig(exp.preprocessingConfig);
    if (preprocessedData) {
      const { artifact, experiment } = trainModel(
        exp.algorithm,
        preprocessedData,
        exp.datasetName,
        exp.datasetFingerprint,
        exp.hyperparameters
      );
      handleModelTrained(artifact, experiment);
      setActiveTab('simulation');
    }
  };

  // Handle Clear Experiments
  const handleClearHistory = () => {
    clearExperimentHistory();
    setExperiments([]);
  };

  // Handle Status change in Registry
  const handleUpdateStatus = (modelId: string, status: 'draft' | 'staging' | 'production') => {
    updateModelStatus(modelId, status);
    setModels(getStoredModels());
    if (activeArtifact && activeArtifact.id === modelId) {
      setActiveArtifact({ ...activeArtifact, status });
    }
  };

  // Trigger Retraining from MLOps Drift Alert
  const handleTriggerRetraining = () => {
    if (!preprocessedData || !activeArtifact) return;
    const { artifact, experiment } = trainModel(
      activeArtifact.algorithm,
      preprocessedData,
      activeArtifact.datasetName,
      activeArtifact.datasetFingerprint,
      activeArtifact.hyperparameters
    );
    artifact.version = '1.1.0';
    artifact.name = `${activeArtifact.name} (Retrained)`;
    handleModelTrained(artifact, experiment);
    setActiveTab('experiments');
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Top Bar adhering to Top Bar Contract */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        datasetName={activeDataset.name}
        hasTrainedModel={activeArtifact !== null}
      />

      {/* Main Workspace Stages */}
      <main className="flex-1 pb-16">
        {activeTab === 'studio' && (
          <DatasetStudio
            activeDataset={activeDataset}
            onSelectDataset={handleSelectDataset}
            onProceedToPreprocessing={() => setActiveTab('preprocessing')}
            onUpdateTargetAndFeatures={handleUpdateTargetAndFeatures}
            selectedTarget={selectedTarget}
            selectedFeatures={selectedFeatures}
          />
        )}

        {activeTab === 'preprocessing' && (
          <PreprocessingStudio
            config={preprocessingConfig}
            onChangeConfig={setPreprocessingConfig}
            preprocessedData={preprocessedData}
            onExecutePipeline={handleExecutePipeline}
            onProceedToLab={() => setActiveTab('lab')}
            taskType={activeDataset.suggestedTask || 'classification'}
          />
        )}

        {activeTab === 'lab' && (
          <AlgorithmLab
            preprocessedData={preprocessedData}
            datasetName={activeDataset.name}
            datasetFingerprint={activeDataset.fingerprint}
            onModelTrained={handleModelTrained}
            onProceedToSimulation={() => setActiveTab('simulation')}
            activeArtifact={activeArtifact}
          />
        )}

        {activeTab === 'simulation' && activeArtifact && preprocessedData && (
          <SimulationStudio
            artifact={activeArtifact}
            preprocessedData={preprocessedData}
            onNavigateToExperiments={() => setActiveTab('experiments')}
          />
        )}

        {activeTab === 'simulation' && (!activeArtifact || !preprocessedData) && (
          <div className="max-w-xl mx-auto my-24 p-8 border border-slate-800 rounded-xl bg-slate-900/60 text-center space-y-4">
            <h2 className="text-lg font-bold text-white">No Model Fitted for Simulation</h2>
            <p className="text-xs text-slate-400">
              Please train a model in the Algorithm Lab first to explore interactive decision trees, regression planes, and Lloyd's K-Means clustering steps.
            </p>
            <button
              onClick={() => setActiveTab('lab')}
              className="px-4 py-2 text-xs font-semibold text-white bg-indigo-600 hover:bg-indigo-500 rounded-lg transition-colors"
            >
              Go to Algorithm Lab
            </button>
          </div>
        )}

        {activeTab === 'experiments' && (
          <ExperimentTracker
            experiments={experiments}
            onRerunExperiment={handleRerunExperiment}
            onClearHistory={handleClearHistory}
            onNavigateToRegistry={() => setActiveTab('registry')}
          />
        )}

        {activeTab === 'registry' && (
          <ModelRegistry
            models={models}
            activeModel={activeArtifact}
            onSelectActiveModel={setActiveArtifact}
            onUpdateStatus={handleUpdateStatus}
            onNavigateToMLOps={() => setActiveTab('mlops')}
          />
        )}

        {activeTab === 'mlops' && (
          <MLOpsOverview
            activeModel={activeArtifact}
            preprocessedData={preprocessedData}
            onTriggerRetraining={handleTriggerRetraining}
            onNavigateToStudio={() => setActiveTab('studio')}
          />
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950 px-6 py-4 text-xs text-slate-500 flex flex-col sm:flex-row items-center justify-between gap-4">
        <div>
          <span>ML Algorithm Simulator &amp; MLOps Studio</span>
          <span className="mx-2">·</span>
          <span>Zero Data Leakage Architecture</span>
        </div>
        <div className="flex items-center gap-4 text-slate-400 font-mono text-[11px]">
          <span>Full-Stack REST Engine</span>
          <span>·</span>
          <span>scikit-learn Compatible Specifications</span>
        </div>
      </footer>
    </div>
  );
}
