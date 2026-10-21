/**
 * Core Machine Learning & MLOps Type Definitions
 */

export type TaskType = 'regression' | 'classification' | 'clustering';

export type AlgorithmId =
  | 'linear_regression'
  | 'logistic_regression'
  | 'decision_tree_classifier'
  | 'decision_tree_regressor'
  | 'knn_classifier'
  | 'random_forest_classifier'
  | 'random_forest_regressor'
  | 'kmeans_clustering';

export interface ColumnMetadata {
  name: string;
  type: 'numerical' | 'categorical';
  missingCount: number;
  uniqueCount: number;
  sampleValues: (string | number)[];
  mean?: number;
  min?: number;
  max?: number;
  std?: number;
}

export interface Dataset {
  id: string;
  name: string;
  source: 'upload' | 'synthetic' | 'sample';
  fingerprint: string;
  version: string;
  createdAt: string;
  columns: string[];
  columnMeta: Record<string, ColumnMetadata>;
  data: Record<string, any>[];
  suggestedTarget?: string;
  suggestedTask?: TaskType;
  description?: string;
}

export interface PreprocessingConfig {
  missingStrategy: 'mean' | 'median' | 'mode' | 'drop';
  removeDuplicates: boolean;
  encodingStrategy: 'onehot' | 'label' | 'none';
  scalingStrategy: 'standard' | 'minmax' | 'none';
  outlierTreatment: 'none' | 'iqr_clip' | 'iqr_remove';
  featureSelectionVarianceThreshold: number; // 0 = keep all
  testSplitRatio: number; // e.g. 0.2
  randomSeed: number;
  selectedFeatures: string[];
  targetColumn: string;
}

export interface PreprocessingSummary {
  originalRowCount: number;
  processedRowCount: number;
  originalColumnCount: number;
  processedColumnCount: number;
  imputedValuesCount: number;
  removedDuplicatesCount: number;
  removedOutliersCount: number;
  encodedFeaturesList: string[];
  featureNames: string[];
  trainSize: number;
  testSize: number;
}

export interface PreprocessedData {
  config: PreprocessingConfig;
  summary: PreprocessingSummary;
  featureNames: string[];
  targetName: string;
  taskType: TaskType;
  classes?: (string | number)[];
  train: {
    X: number[][];
    y: (number | string)[];
    rawIndices: number[];
  };
  test: {
    X: number[][];
    y: (number | string)[];
    rawIndices: number[];
  };
  scalerParams?: {
    means: number[];
    stds: number[];
    mins: number[];
    maxs: number[];
  };
  encoderMappings?: Record<string, Record<string, number>>;
  imputerValues?: Record<string, number | string>;
}

export interface Hyperparameters {
  // Linear / Logistic Regression
  learningRate?: number;
  epochs?: number;
  regularizationC?: number;
  fitIntercept?: boolean;

  // Decision Tree / Random Forest
  criterion?: 'gini' | 'entropy' | 'squared_error';
  maxDepth?: number;
  minSamplesSplit?: number;
  nEstimators?: number; // for Random Forest

  // KNN
  nNeighbors?: number;
  distanceMetric?: 'euclidean' | 'manhattan';
  weights?: 'uniform' | 'distance';

  // K-Means
  nClusters?: number;
  maxIterations?: number;
  initMethod?: 'k-means++' | 'random';
  randomSeed?: number;
}

export interface RegressionMetrics {
  mae: number;
  mse: number;
  rmse: number;
  r2: number;
  trainR2?: number;
}

export interface ClassificationMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  confusionMatrix: {
    labels: string[];
    matrix: number[][]; // [actual][predicted]
  };
  report: Record<
    string,
    {
      precision: number;
      recall: number;
      f1Score: number;
      support: number;
    }
  >;
}

export interface ClusteringMetrics {
  inertia: number;
  silhouetteScore: number;
  clusterCounts: number[];
}

export type ModelMetrics =
  | { task: 'regression'; metrics: RegressionMetrics }
  | { task: 'classification'; metrics: ClassificationMetrics }
  | { task: 'clustering'; metrics: ClusteringMetrics };

export interface TreeNode {
  id: string;
  featureIndex?: number;
  featureName?: string;
  threshold?: number;
  isLeaf: boolean;
  prediction?: number | string;
  impurity?: number;
  samples: number;
  value?: number[] | number; // class counts or mean target
  leftChild?: TreeNode;
  rightChild?: TreeNode;
}

export interface AlgorithmStepExplanation {
  stepNumber: number;
  title: string;
  concept: string;
  formula?: string;
  concreteCalculation: string;
  intermediateValues?: Record<string, any>;
}

export interface ModelArtifact {
  id: string;
  experimentId: string;
  name: string;
  algorithm: AlgorithmId;
  taskType: TaskType;
  datasetFingerprint: string;
  datasetName: string;
  hyperparameters: Hyperparameters;
  featureNames: string[];
  targetName: string;
  classes?: (string | number)[];
  metrics: ModelMetrics;
  trainingDurationMs: number;
  createdAt: string;
  version: string;
  status: 'draft' | 'staging' | 'production';

  // Specific model parameters
  linearModel?: {
    coefficients: number[];
    intercept: number;
    equation: string;
  };
  treeModel?: {
    root: TreeNode;
    maxDepthReached: number;
  };
  forestModel?: {
    trees: TreeNode[];
    featureImportances: { feature: string; importance: number }[];
  };
  knnModel?: {
    trainX: number[][];
    trainY: (number | string)[];
    k: number;
    metric: string;
  };
  kmeansModel?: {
    k: number;
    centroids: number[][];
    iterationCentroids: number[][][]; // Centroids at each iteration
    convergedIn: number;
  };

  // Preprocessing metadata for standalone inference
  preprocessing: {
    scalerType: 'standard' | 'minmax' | 'none';
    scalerParams?: {
      means: number[];
      stds: number[];
      mins: number[];
      maxs: number[];
    };
    encoderMappings?: Record<string, Record<string, number>>;
    imputerValues?: Record<string, number | string>;
    selectedFeatures: string[];
  };

  // Explanations for learning mode
  explanations: AlgorithmStepExplanation[];
}

export interface Experiment {
  id: string;
  timestamp: string;
  name: string;
  algorithm: AlgorithmId;
  taskType: TaskType;
  datasetFingerprint: string;
  datasetName: string;
  hyperparameters: Hyperparameters;
  preprocessingConfig: PreprocessingConfig;
  metrics: ModelMetrics;
  trainingDurationMs: number;
  modelArtifactId: string;
  status: 'completed' | 'failed';
  errorMessage?: string;
}

export interface DriftMetric {
  featureName: string;
  trainingMean: number;
  trainingStd: number;
  inferenceMean: number;
  inferenceStd: number;
  driftDetected: boolean;
  driftScore: number; // e.g. Kolmogorov-Smirnov or z-score divergence
}

// Runtime object exports for Node.js native ESM type-stripping runtime compatibility
export const ColumnMetadata = {};
export const Dataset = {};
export const TaskType = {};
export const AlgorithmId = {};
export const PreprocessingConfig = {};
export const PreprocessingSummary = {};
export const PreprocessedData = {};
export const Hyperparameters = {};
export const RegressionMetrics = {};
export const ClassificationMetrics = {};
export const ClusteringMetrics = {};
export const ModelMetrics = {};
export const TreeNode = {};
export const AlgorithmStepExplanation = {};
export const ModelArtifact = {};
export const Experiment = {};
export const DriftMetric = {};

