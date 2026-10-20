import { AlgorithmStepExplanation, Hyperparameters, TreeNode } from '../types.ts';
import { createRNG } from '../datasets.ts';
import { trainDecisionTree } from './decisionTree.ts';

export interface RandomForestResult {
  nEstimators: number;
  trees: TreeNode[];
  predictions: (number | string)[];
  trainPredictions: (number | string)[];
  featureImportances: { feature: string; importance: number }[];
  explanations: AlgorithmStepExplanation[];
}

export function trainRandomForest(
  trainX: number[][],
  trainY: (number | string)[],
  testX: number[][],
  featureNames: string[],
  task: 'classification' | 'regression',
  hyperparams: Hyperparameters = {}
): RandomForestResult {
  const nEstimators = Math.max(3, Math.min(30, hyperparams.nEstimators ?? 10));
  const maxDepth = hyperparams.maxDepth ?? 5;
  const seed = hyperparams.randomSeed ?? 42;
  const rng = createRNG(seed);

  const n = trainX.length;
  const p = trainX[0]?.length || 0;
  const trees: TreeNode[] = [];
  const accumulatedImportances: number[] = new Array(p).fill(0);

  // Train each estimator on a bootstrap sample
  for (let t = 0; t < nEstimators; t++) {
    // 1. Bootstrap sample (sampling with replacement)
    const sampleIndices: number[] = [];
    for (let i = 0; i < n; i++) {
      sampleIndices.push(Math.floor(rng() * n));
    }

    const bX = sampleIndices.map((idx) => trainX[idx]);
    const bY = sampleIndices.map((idx) => trainY[idx]);

    // Train individual CART tree
    const treeResult = trainDecisionTree(bX, bY, testX, featureNames, task, {
      maxDepth,
      minSamplesSplit: 2,
      criterion: hyperparams.criterion,
    });

    trees.push(treeResult.root);

    // Accumulate feature importance
    treeResult.featureImportances.forEach((fi) => {
      const idx = featureNames.indexOf(fi.feature);
      if (idx >= 0) accumulatedImportances[idx] += fi.importance;
    });
  }

  // Aggregate normalized feature importances
  const totalImp = accumulatedImportances.reduce((a, b) => a + b, 0) || 1;
  const featureImportances = featureNames.map((name, idx) => ({
    feature: name,
    importance: Number((accumulatedImportances[idx] / totalImp).toFixed(4)),
  }));
  featureImportances.sort((a, b) => b.importance - a.importance);

  // Helper to predict single row using single tree
  const predictSingleTree = (row: number[], node: TreeNode): number | string => {
    if (node.isLeaf || node.featureIndex === undefined || node.threshold === undefined) {
      return node.prediction!;
    }
    const val = row[node.featureIndex];
    if (val <= node.threshold) {
      return node.leftChild ? predictSingleTree(row, node.leftChild) : node.prediction!;
    } else {
      return node.rightChild ? predictSingleTree(row, node.rightChild) : node.prediction!;
    }
  };

  // Ensemble Aggregate Prediction
  const ensemblePredictRow = (row: number[]): number | string => {
    const votes = trees.map((t) => predictSingleTree(row, t));

    if (task === 'regression') {
      const nums = votes.map(Number);
      const mean = nums.reduce((a, b) => a + b, 0) / nums.length;
      return Number(mean.toFixed(3));
    } else {
      const counts: Record<string, number> = {};
      for (const v of votes) counts[String(v)] = (counts[String(v)] || 0) + 1;
      let winning = votes[0];
      let maxCount = -1;
      for (const [k, count] of Object.entries(counts)) {
        if (count > maxCount) {
          maxCount = count;
          winning = k;
        }
      }
      return winning;
    }
  };

  const predictions = testX.map(ensemblePredictRow);
  const trainPredictions = trainX.map(ensemblePredictRow);

  const explanations: AlgorithmStepExplanation[] = [
    {
      stepNumber: 1,
      title: 'Bootstrap Aggregating (Bagging)',
      concept: 'Random Forest trains an ensemble of de-correlated decision trees, each trained on a bootstrap sample (random sampling with replacement).',
      formula: 'D^{*b} = \\{ (\\mathbf{x}_i, y_i) \\}_{i=1}^N \\quad \\text{drawn uniformly with replacement}',
      concreteCalculation: `Trained ensemble of ${nEstimators} independent trees with max depth ${maxDepth} on ${n} samples.`,
    },
    {
      stepNumber: 2,
      title: 'Variance Reduction via Ensemble Consensus',
      concept: 'Combining predictions across multiple unpruned or shallow trees reduces model variance without increasing bias.',
      formula:
        task === 'classification'
          ? '\\hat{y}_{RF} = \\text{mode} \\{ T_1(\\mathbf{x}), T_2(\\mathbf{x}), \\dots, T_B(\\mathbf{x}) \\}'
          : '\\hat{y}_{RF} = \\frac{1}{B} \\sum_{b=1}^B T_b(\\mathbf{x})',
      concreteCalculation: `Ensemble majority voting / averaging computed over ${nEstimators} trees.`,
    },
    {
      stepNumber: 3,
      title: 'Ensemble Feature Importance (MDI)',
      concept: 'Overall importance of each feature is calculated by averaging the Mean Decrease in Impurity across all trees in the forest.',
      formula: 'Importance_{RF}(f) = \\frac{1}{B} \\sum_{b=1}^B \\text{Importance}_{T_b}(f)',
      concreteCalculation: `Dominant feature across all ${nEstimators} estimators is "${featureImportances[0]?.feature || 'none'}" (${(
        (featureImportances[0]?.importance || 0) * 100
      ).toFixed(1)}%).`,
    },
  ];

  return {
    nEstimators,
    trees,
    predictions,
    trainPredictions,
    featureImportances,
    explanations,
  };
}
