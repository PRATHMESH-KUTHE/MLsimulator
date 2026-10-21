import type { AlgorithmStepExplanation, Hyperparameters, TreeNode } from '../types.ts';

export interface DecisionTreeResult {
  root: TreeNode;
  maxDepthReached: number;
  predictions: (number | string)[];
  trainPredictions: (number | string)[];
  featureImportances: { feature: string; importance: number }[];
  explanations: AlgorithmStepExplanation[];
}

export function trainDecisionTree(
  trainX: number[][],
  trainY: (number | string)[],
  testX: number[][],
  featureNames: string[],
  task: 'classification' | 'regression',
  hyperparams: Hyperparameters = {}
): DecisionTreeResult {
  const maxDepth = hyperparams.maxDepth ?? 5;
  const minSamplesSplit = hyperparams.minSamplesSplit ?? 2;
  const criterion = hyperparams.criterion ?? (task === 'classification' ? 'gini' : 'squared_error');

  const n = trainX.length;
  const p = trainX[0]?.length || 0;
  let nodeCounter = 0;
  let maxDepthReached = 0;
  const rawFeatureImportance: number[] = new Array(p).fill(0);

  // Helper for Classification Impurity
  const calcImpurity = (labels: (number | string)[]): number => {
    if (labels.length === 0) return 0;
    if (task === 'regression') {
      const nums = labels.map(Number);
      const mean = nums.reduce((a, b) => a + b, 0) / nums.length;
      return nums.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / nums.length;
    }

    const counts: Record<string, number> = {};
    for (const l of labels) {
      counts[String(l)] = (counts[String(l)] || 0) + 1;
    }
    const total = labels.length;

    if (criterion === 'entropy') {
      let ent = 0;
      for (const c of Object.values(counts)) {
        const prob = c / total;
        if (prob > 0) ent -= prob * Math.log2(prob);
      }
      return ent;
    } else {
      // Gini impurity: 1 - sum(p_i^2)
      let sumSq = 0;
      for (const c of Object.values(counts)) {
        const prob = c / total;
        sumSq += prob * prob;
      }
      return 1 - sumSq;
    }
  };

  // Helper to determine Leaf Prediction Value
  const getLeafPrediction = (labels: (number | string)[]): { pred: number | string; value: any } => {
    if (task === 'regression') {
      const nums = labels.map(Number);
      const mean = nums.reduce((a, b) => a + b, 0) / (nums.length || 1);
      return { pred: Number(mean.toFixed(3)), value: Number(mean.toFixed(3)) };
    } else {
      const counts: Record<string, number> = {};
      for (const l of labels) counts[String(l)] = (counts[String(l)] || 0) + 1;
      let majority = labels[0];
      let maxCount = -1;
      for (const [k, c] of Object.entries(counts)) {
        if (c > maxCount) {
          maxCount = c;
          majority = k;
        }
      }
      return { pred: majority, value: counts };
    }
  };

  // Recursive Tree Builder
  const buildNode = (indices: number[], depth: number): TreeNode => {
    nodeCounter++;
    const nodeId = `node_${nodeCounter}`;
    if (depth > maxDepthReached) maxDepthReached = depth;

    const nodeSamples = indices.length;
    const nodeLabels = indices.map((idx) => trainY[idx]);
    const currentImpurity = calcImpurity(nodeLabels);
    const { pred, value } = getLeafPrediction(nodeLabels);

    // Check stopping criteria
    if (
      depth >= maxDepth ||
      nodeSamples < minSamplesSplit ||
      currentImpurity <= 1e-7 ||
      new Set(nodeLabels).size <= 1
    ) {
      return {
        id: nodeId,
        isLeaf: true,
        prediction: pred,
        samples: nodeSamples,
        impurity: Number(currentImpurity.toFixed(4)),
        value,
      };
    }

    // Find best split across all features and threshold values
    let bestGain = -Infinity;
    let bestFeature = -1;
    let bestThreshold = 0;
    let bestLeftIndices: number[] = [];
    let bestRightIndices: number[] = [];

    for (let f = 0; f < p; f++) {
      // Collect unique candidate split thresholds
      const vals = Array.from(new Set(indices.map((idx) => trainX[idx][f]))).sort((a, b) => a - b);
      if (vals.length <= 1) continue;

      for (let t = 0; t < vals.length - 1; t++) {
        const threshold = (vals[t] + vals[t + 1]) / 2;
        const left: number[] = [];
        const right: number[] = [];

        for (const idx of indices) {
          if (trainX[idx][f] <= threshold) left.push(idx);
          else right.push(idx);
        }

        if (left.length === 0 || right.length === 0) continue;

        const leftImpurity = calcImpurity(left.map((idx) => trainY[idx]));
        const rightImpurity = calcImpurity(right.map((idx) => trainY[idx]));

        // Information Gain = Impurity(parent) - [N_L/N * Impurity(left) + N_R/N * Impurity(right)]
        const gain =
          currentImpurity -
          ((left.length / nodeSamples) * leftImpurity + (right.length / nodeSamples) * rightImpurity);

        if (gain > bestGain) {
          bestGain = gain;
          bestFeature = f;
          bestThreshold = threshold;
          bestLeftIndices = left;
          bestRightIndices = right;
        }
      }
    }

    // If no positive information gain could be achieved, create leaf
    if (bestGain <= 1e-7 || bestFeature === -1) {
      return {
        id: nodeId,
        isLeaf: true,
        prediction: pred,
        samples: nodeSamples,
        impurity: Number(currentImpurity.toFixed(4)),
        value,
      };
    }

    // Accumulate feature importance (Gain * fraction of samples)
    rawFeatureImportance[bestFeature] += bestGain * (nodeSamples / n);

    // Recursively build children
    const leftChild = buildNode(bestLeftIndices, depth + 1);
    const rightChild = buildNode(bestRightIndices, depth + 1);

    return {
      id: nodeId,
      isLeaf: false,
      featureIndex: bestFeature,
      featureName: featureNames[bestFeature] || `Feature ${bestFeature + 1}`,
      threshold: Number(bestThreshold.toFixed(3)),
      samples: nodeSamples,
      impurity: Number(currentImpurity.toFixed(4)),
      value,
      prediction: pred,
      leftChild,
      rightChild,
    };
  };

  const allIndices = trainX.map((_, i) => i);
  const root = buildNode(allIndices, 0);

  // Calculate normalized feature importance
  const totalImp = rawFeatureImportance.reduce((a, b) => a + b, 0) || 1;
  const featureImportances = featureNames.map((name, idx) => ({
    feature: name,
    importance: Number((rawFeatureImportance[idx] / totalImp).toFixed(4)),
  }));
  featureImportances.sort((a, b) => b.importance - a.importance);

  // Prediction helper
  const predictRow = (row: number[], node: TreeNode): number | string => {
    if (node.isLeaf || node.featureIndex === undefined || node.threshold === undefined) {
      return node.prediction!;
    }
    const val = row[node.featureIndex];
    if (val <= node.threshold) {
      return node.leftChild ? predictRow(row, node.leftChild) : node.prediction!;
    } else {
      return node.rightChild ? predictRow(row, node.rightChild) : node.prediction!;
    }
  };

  const predictions = testX.map((row) => predictRow(row, root));
  const trainPredictions = trainX.map((row) => predictRow(row, root));

  const explanations: AlgorithmStepExplanation[] = [
    {
      stepNumber: 1,
      title: 'Recursive Binary Partitioning',
      concept: 'CART (Classification and Regression Trees) iteratively segments the feature space into orthogonal rectangular regions.',
      formula:
        task === 'classification'
          ? 'Gini(S) = 1 - \\sum_{i=1}^C p_i^2 \\quad | \\quad Entropy(S) = -\\sum_{i=1}^C p_i \\log_2(p_i)'
          : 'MSE(S) = \\frac{1}{|S|} \\sum_{i \\in S} (y_i - \\bar{y})^2',
      concreteCalculation: `Constructed decision tree up to depth ${maxDepthReached} (limit ${maxDepth}) using ${nodeCounter} total nodes.`,
    },
    {
      stepNumber: 2,
      title: 'Information Gain Splitting Rule',
      concept: 'At each internal node, the feature and threshold that maximize impurity reduction (Information Gain) are chosen.',
      formula: '\\Delta I = I(Node) - \\left( \\frac{N_L}{N} I(Left) + \\frac{N_R}{N} I(Right) \\right)',
      concreteCalculation: `Root split chosen: ${root.featureName} <= ${root.threshold} with initial impurity ${root.impurity}.`,
    },
    {
      stepNumber: 3,
      title: 'Feature Importance Evaluation',
      concept: 'Features are ranked by total weighted impurity reduction across all split points in the tree.',
      formula: 'Importance(f) = \\sum_{nodes \\in splits(f)} \\frac{N_{node}}{N_{total}} \\Delta I_{node}',
      concreteCalculation: `Top explanatory feature is "${featureImportances[0]?.feature || 'none'}" contributing ${(
        (featureImportances[0]?.importance || 0) * 100
      ).toFixed(1)}% of total impurity decrease.`,
    },
  ];

  return {
    root,
    maxDepthReached,
    predictions,
    trainPredictions,
    featureImportances,
    explanations,
  };
}

/**
 * Traces a specific input sample through the tree to return the exact decision path.
 */
export function traceDecisionPath(
  row: number[],
  root: TreeNode
): { nodeId: string; rule: string; satisfied: boolean; impurity: number; samples: number }[] {
  const path: { nodeId: string; rule: string; satisfied: boolean; impurity: number; samples: number }[] = [];

  let current: TreeNode | undefined = root;

  while (current) {
    if (current.isLeaf) {
      path.push({
        nodeId: current.id,
        rule: `Leaf Prediction: ${current.prediction}`,
        satisfied: true,
        impurity: current.impurity || 0,
        samples: current.samples,
      });
      break;
    }

    const featIdx: number = current.featureIndex ?? 0;
    const threshold: number = current.threshold ?? 0;
    const val: number = row[featIdx] ?? 0;
    const isLeft: boolean = val <= threshold;

    path.push({
      nodeId: current.id,
      rule: `${current.featureName} (${val.toFixed(2)}) <= ${threshold.toFixed(2)}`,
      satisfied: isLeft,
      impurity: current.impurity || 0,
      samples: current.samples,
    });

    current = isLeft ? current.leftChild : current.rightChild;
  }

  return path;
}
