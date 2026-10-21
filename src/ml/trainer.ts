import type {
  AlgorithmId,
  Experiment,
  Hyperparameters,
  ModelArtifact,
  ModelMetrics,
  PreprocessedData,
  TaskType,
} from './types.ts';
import { trainLinearRegression } from './algorithms/linearRegression.ts';
import { trainLogisticRegression } from './algorithms/logisticRegression.ts';
import { trainDecisionTree } from './algorithms/decisionTree.ts';
import { trainKNNClassifier } from './algorithms/knn.ts';
import { trainRandomForest } from './algorithms/randomForest.ts';
import { trainKMeans } from './algorithms/kmeans.ts';
import { evaluateClassification, evaluateClustering, evaluateRegression } from './metrics.ts';

export function getAlgorithmCompatibility(algo: AlgorithmId): {
  taskType: TaskType;
  requiresTarget: boolean;
  name: string;
  category: 'Supervised Learning' | 'Unsupervised Learning';
} {
  switch (algo) {
    case 'linear_regression':
      return {
        taskType: 'regression',
        requiresTarget: true,
        name: 'Linear Regression',
        category: 'Supervised Learning',
      };
    case 'decision_tree_regressor':
      return {
        taskType: 'regression',
        requiresTarget: true,
        name: 'Decision Tree Regressor',
        category: 'Supervised Learning',
      };
    case 'random_forest_regressor':
      return {
        taskType: 'regression',
        requiresTarget: true,
        name: 'Random Forest Regressor',
        category: 'Supervised Learning',
      };
    case 'logistic_regression':
      return {
        taskType: 'classification',
        requiresTarget: true,
        name: 'Logistic Regression',
        category: 'Supervised Learning',
      };
    case 'decision_tree_classifier':
      return {
        taskType: 'classification',
        requiresTarget: true,
        name: 'Decision Tree Classifier',
        category: 'Supervised Learning',
      };
    case 'knn_classifier':
      return {
        taskType: 'classification',
        requiresTarget: true,
        name: 'K-Nearest Neighbors (KNN)',
        category: 'Supervised Learning',
      };
    case 'random_forest_classifier':
      return {
        taskType: 'classification',
        requiresTarget: true,
        name: 'Random Forest Classifier',
        category: 'Supervised Learning',
      };
    case 'kmeans_clustering':
      return {
        taskType: 'clustering',
        requiresTarget: false,
        name: 'K-Means Clustering',
        category: 'Unsupervised Learning',
      };
  }
}

export function trainModel(
  algo: AlgorithmId,
  preprocessed: PreprocessedData,
  datasetName: string,
  datasetFingerprint: string,
  hyperparams: Hyperparameters = {}
): { artifact: ModelArtifact; experiment: Experiment } {
  const startTime = performance.now();
  const compat = getAlgorithmCompatibility(algo);

  if (compat.taskType !== preprocessed.taskType) {
    throw new Error(
      `Incompatible algorithm: ${compat.name} is designed for ${compat.taskType} tasks, but the preprocessed data is set up for ${preprocessed.taskType}.`
    );
  }

  const expId = `exp_${Date.now()}_${Math.floor(Math.random() * 1000)}`;
  const artifactId = `model_${Date.now()}_${Math.floor(Math.random() * 1000)}`;

  let metrics: ModelMetrics;
  let linearModel: ModelArtifact['linearModel'];
  let treeModel: ModelArtifact['treeModel'];
  let forestModel: ModelArtifact['forestModel'];
  let knnModel: ModelArtifact['knnModel'];
  let kmeansModel: ModelArtifact['kmeansModel'];
  let explanations: ModelArtifact['explanations'] = [];

  const trainX = preprocessed.train.X;
  const trainY = preprocessed.train.y;
  const testX = preprocessed.test.X;
  const testY = preprocessed.test.y;
  const featureNames = preprocessed.featureNames;

  switch (algo) {
    case 'linear_regression': {
      const res = trainLinearRegression(trainX, trainY, testX, featureNames, hyperparams);
      const regMetrics = evaluateRegression(
        testY.map(Number),
        res.predictions,
        trainY.map(Number),
        res.trainPredictions
      );
      metrics = { task: 'regression', metrics: regMetrics };
      linearModel = {
        coefficients: res.coefficients,
        intercept: res.intercept,
        equation: res.equation,
      };
      explanations = res.explanations;
      break;
    }

    case 'logistic_regression': {
      const res = trainLogisticRegression(trainX, trainY, testX, featureNames, hyperparams);
      const clfMetrics = evaluateClassification(testY, res.predictions, res.classes);
      metrics = { task: 'classification', metrics: clfMetrics };
      linearModel = {
        coefficients: res.weights[0] || [],
        intercept: res.biases[0] || 0,
        equation: `P(y=1) = σ(${res.weights[0]
          ?.map((w, i) => `${w >= 0 && i > 0 ? '+' : ''}${w.toFixed(2)}·${featureNames[i]}`)
          .join(' ')} ${res.biases[0] >= 0 ? '+' : ''}${res.biases[0]?.toFixed(2)})`,
      };
      explanations = res.explanations;
      break;
    }

    case 'decision_tree_classifier': {
      const res = trainDecisionTree(trainX, trainY, testX, featureNames, 'classification', hyperparams);
      const clfMetrics = evaluateClassification(testY, res.predictions, preprocessed.classes);
      metrics = { task: 'classification', metrics: clfMetrics };
      treeModel = {
        root: res.root,
        maxDepthReached: res.maxDepthReached,
      };
      explanations = res.explanations;
      break;
    }

    case 'decision_tree_regressor': {
      const res = trainDecisionTree(trainX, trainY, testX, featureNames, 'regression', hyperparams);
      const regMetrics = evaluateRegression(
        testY.map(Number),
        res.predictions.map(Number),
        trainY.map(Number),
        res.trainPredictions.map(Number)
      );
      metrics = { task: 'regression', metrics: regMetrics };
      treeModel = {
        root: res.root,
        maxDepthReached: res.maxDepthReached,
      };
      explanations = res.explanations;
      break;
    }

    case 'knn_classifier': {
      const res = trainKNNClassifier(trainX, trainY, testX, featureNames, hyperparams);
      const clfMetrics = evaluateClassification(testY, res.predictions, res.classes);
      metrics = { task: 'classification', metrics: clfMetrics };
      knnModel = {
        trainX,
        trainY,
        k: res.k,
        metric: res.metric,
      };
      explanations = res.explanations;
      break;
    }

    case 'random_forest_classifier': {
      const res = trainRandomForest(trainX, trainY, testX, featureNames, 'classification', hyperparams);
      const clfMetrics = evaluateClassification(testY, res.predictions, preprocessed.classes);
      metrics = { task: 'classification', metrics: clfMetrics };
      forestModel = {
        trees: res.trees,
        featureImportances: res.featureImportances,
      };
      explanations = res.explanations;
      break;
    }

    case 'random_forest_regressor': {
      const res = trainRandomForest(trainX, trainY, testX, featureNames, 'regression', hyperparams);
      const regMetrics = evaluateRegression(
        testY.map(Number),
        res.predictions.map(Number),
        trainY.map(Number),
        res.trainPredictions.map(Number)
      );
      metrics = { task: 'regression', metrics: regMetrics };
      forestModel = {
        trees: res.trees,
        featureImportances: res.featureImportances,
      };
      explanations = res.explanations;
      break;
    }

    case 'kmeans_clustering': {
      const res = trainKMeans(trainX, hyperparams);
      const cltMetrics = evaluateClustering(trainX, res.assignments, res.centroids);
      metrics = { task: 'clustering', metrics: cltMetrics };
      kmeansModel = {
        k: res.k,
        centroids: res.centroids,
        iterationCentroids: res.iterationCentroids,
        convergedIn: res.convergedIn,
      };
      explanations = res.explanations;
      break;
    }
  }

  const durationMs = Math.round(performance.now() - startTime);

  const artifact: ModelArtifact = {
    id: artifactId,
    experimentId: expId,
    name: `${compat.name} (${datasetName})`,
    algorithm: algo,
    taskType: compat.taskType,
    datasetFingerprint,
    datasetName,
    hyperparameters: hyperparams,
    featureNames: preprocessed.featureNames,
    targetName: preprocessed.targetName,
    classes: preprocessed.classes,
    metrics,
    trainingDurationMs: durationMs,
    createdAt: new Date().toISOString(),
    version: '1.0.0',
    status: 'staged' as any,
    linearModel,
    treeModel,
    forestModel,
    knnModel,
    kmeansModel,
    preprocessing: {
      scalerType: preprocessed.config.scalingStrategy,
      scalerParams: preprocessed.scalerParams,
      encoderMappings: preprocessed.encoderMappings,
      imputerValues: preprocessed.imputerValues,
      selectedFeatures: preprocessed.config.selectedFeatures,
    },
    explanations,
  };

  const experiment: Experiment = {
    id: expId,
    timestamp: new Date().toISOString(),
    name: `${compat.name} Run`,
    algorithm: algo,
    taskType: compat.taskType,
    datasetFingerprint,
    datasetName,
    hyperparameters: hyperparams,
    preprocessingConfig: preprocessed.config,
    metrics,
    trainingDurationMs: durationMs,
    modelArtifactId: artifactId,
    status: 'completed',
  };

  return { artifact, experiment };
}

/**
 * Predict on single row using saved ModelArtifact
 */
export function predictWithArtifact(
  artifact: ModelArtifact,
  rawInput: Record<string, any>
): { prediction: number | string; probability?: number } {
  // 1. Preprocess raw input according to artifact's saved pipeline
  const p = artifact.preprocessing;
  const originalFeatures = p.selectedFeatures.filter((f) => f !== artifact.targetName);

  // Impute
  const filledInput: Record<string, any> = { ...rawInput };
  if (p.imputerValues) {
    for (const [feat, fillVal] of Object.entries(p.imputerValues)) {
      if (filledInput[feat] === undefined || filledInput[feat] === null || filledInput[feat] === '') {
        filledInput[feat] = fillVal;
      }
    }
  }

  // Vectorize and Encode
  const rawVec: number[] = [];
  for (const feature of originalFeatures) {
    if (p.encoderMappings && p.encoderMappings[feature] && p.scalerType !== undefined) {
      const valStr = String(filledInput[feature] ?? '');
      const cats = Object.keys(p.encoderMappings[feature]);
      for (const cat of cats) {
        rawVec.push(valStr === cat ? 1 : 0);
      }
    } else {
      rawVec.push(Number(filledInput[feature]) || 0);
    }
  }

  // Scale
  let scaledVec = rawVec;
  if (p.scalerType === 'standard' && p.scalerParams) {
    scaledVec = rawVec.map((val, idx) => {
      const m = p.scalerParams!.means[idx] || 0;
      const s = p.scalerParams!.stds[idx] || 1;
      return (val - m) / s;
    });
  } else if (p.scalerType === 'minmax' && p.scalerParams) {
    scaledVec = rawVec.map((val, idx) => {
      const min = p.scalerParams!.mins[idx] || 0;
      const max = p.scalerParams!.maxs[idx] || 1;
      return (val - min) / (max - min || 1);
    });
  }

  // Predict based on algorithm
  if (artifact.linearModel) {
    let val = artifact.linearModel.intercept;
    for (let i = 0; i < scaledVec.length; i++) {
      val += (artifact.linearModel.coefficients[i] || 0) * scaledVec[i];
    }
    if (artifact.taskType === 'regression') {
      return { prediction: Number(val.toFixed(3)) };
    } else {
      const prob = 1 / (1 + Math.exp(-val));
      const predClass = prob >= 0.5 ? artifact.classes?.[0] || '1' : artifact.classes?.[1] || '0';
      return { prediction: predClass, probability: Number(prob.toFixed(3)) };
    }
  }

  if (artifact.treeModel) {
    const walkTree = (node: any): any => {
      if (node.isLeaf || node.featureIndex === undefined) return node.prediction;
      const v = scaledVec[node.featureIndex];
      if (v <= node.threshold) return node.leftChild ? walkTree(node.leftChild) : node.prediction;
      return node.rightChild ? walkTree(node.rightChild) : node.prediction;
    };
    return { prediction: walkTree(artifact.treeModel.root) };
  }

  if (artifact.forestModel) {
    const walkTree = (node: any): any => {
      if (node.isLeaf || node.featureIndex === undefined) return node.prediction;
      const v = scaledVec[node.featureIndex];
      if (v <= node.threshold) return node.leftChild ? walkTree(node.leftChild) : node.prediction;
      return node.rightChild ? walkTree(node.rightChild) : node.prediction;
    };
    const votes = artifact.forestModel.trees.map((t) => walkTree(t));
    if (artifact.taskType === 'regression') {
      const mean = votes.reduce((a: number, b: any) => a + Number(b), 0) / votes.length;
      return { prediction: Number(mean.toFixed(3)) };
    } else {
      const counts: Record<string, number> = {};
      for (const v of votes) counts[String(v)] = (counts[String(v)] || 0) + 1;
      let winning = votes[0];
      let maxC = 0;
      for (const [k, c] of Object.entries(counts)) {
        if (c > maxC) {
          maxC = c;
          winning = k;
        }
      }
      return { prediction: winning, probability: Number((maxC / votes.length).toFixed(3)) };
    }
  }

  if (artifact.kmeansModel) {
    let bestC = 0;
    let minDist = Infinity;
    for (let c = 0; c < artifact.kmeansModel.centroids.length; c++) {
      let d = 0;
      for (let j = 0; j < scaledVec.length; j++) {
        const diff = scaledVec[j] - (artifact.kmeansModel.centroids[c][j] || 0);
        d += diff * diff;
      }
      if (d < minDist) {
        minDist = d;
        bestC = c;
      }
    }
    return { prediction: `Cluster ${bestC}` };
  }

  return { prediction: 'Unknown' };
}
