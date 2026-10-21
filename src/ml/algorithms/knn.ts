import type { AlgorithmStepExplanation, Hyperparameters } from '../types.ts';

export interface KNNResult {
  k: number;
  metric: string;
  weights: string;
  predictions: (string | number)[];
  probabilities: number[][];
  classes: string[];
  explanations: AlgorithmStepExplanation[];
}

export function trainKNNClassifier(
  trainX: number[][],
  trainY: (number | string)[],
  testX: number[][],
  featureNames: string[],
  hyperparams: Hyperparameters = {}
): KNNResult {
  const k = Math.max(1, Math.min(trainX.length, hyperparams.nNeighbors ?? 5));
  const metric = hyperparams.distanceMetric ?? 'euclidean';
  const weightMode = hyperparams.weights ?? 'uniform';

  const classes = Array.from(new Set(trainY.map(String))).sort();

  // Helper distance function
  const computeDistance = (a: number[], b: number[]): number => {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      const diff = Math.abs(a[i] - b[i]);
      sum += metric === 'manhattan' ? diff : diff * diff;
    }
    return metric === 'manhattan' ? sum : Math.sqrt(sum);
  };

  const predictSample = (query: number[]): { pred: string | number; probs: number[] } => {
    // Calculate distance to all training samples
    const distances: { index: number; dist: number; label: string }[] = [];
    for (let i = 0; i < trainX.length; i++) {
      const dist = computeDistance(query, trainX[i]);
      distances.push({ index: i, dist, label: String(trainY[i]) });
    }

    distances.sort((a, b) => a.dist - b.dist);
    const kNeighbors = distances.slice(0, k);

    // Vote tally
    const votes: Record<string, number> = {};
    for (const c of classes) votes[c] = 0;

    for (const neighbor of kNeighbors) {
      const weight = weightMode === 'distance' ? 1 / (neighbor.dist + 1e-5) : 1;
      votes[neighbor.label] = (votes[neighbor.label] || 0) + weight;
    }

    let totalVoteWeight = 0;
    let winningClass = classes[0];
    let maxVote = -1;

    for (const [c, v] of Object.entries(votes)) {
      totalVoteWeight += v;
      if (v > maxVote) {
        maxVote = v;
        winningClass = c;
      }
    }

    const probs = classes.map((c) =>
      Number((totalVoteWeight > 0 ? votes[c] / totalVoteWeight : 1 / classes.length).toFixed(3))
    );

    return { pred: winningClass, probs };
  };

  const testResults = testX.map(predictSample);
  const predictions = testResults.map((r) => r.pred);
  const probabilities = testResults.map((r) => r.probs);

  const explanations: AlgorithmStepExplanation[] = [
    {
      stepNumber: 1,
      title: 'Non-Parametric Instance-Based Learning',
      concept: 'KNN is a lazy learner storing the entire training set rather than fitting an explicit parameterized equation.',
      formula: 'd(\\mathbf{x}, \\mathbf{z}) = \\left( \\sum_{j=1}^p |x_j - z_j|^q \\right)^{1/q} \\quad (q=2 \\text{ Euclidean}, q=1 \\text{ Manhattan})',
      concreteCalculation: `Using k = ${k} nearest neighbors under ${metric} distance and ${weightMode} voting weighting.`,
    },
    {
      stepNumber: 2,
      title: 'Neighborhood Query & Distance Computation',
      concept: 'For any query vector, distances to all training points are computed and sorted to select the top k closest vectors.',
      formula: 'N_k(\\mathbf{x}) = \\text{arg\\,min}_{S \\subset D, |S|=k} \\sum_{\\mathbf{z} \\in S} d(\\mathbf{x}, \\mathbf{z})',
      concreteCalculation: `Evaluating neighborhood across ${trainX.length} reference points and ${classes.length} class candidates.`,
    },
    {
      stepNumber: 3,
      title: 'Majority Voting & Posterior Probability',
      concept: 'Classes cast votes weighted uniformly or inversely proportional to distance. Highest total score wins the classification.',
      formula: 'P(y = c | \\mathbf{x}) = \\frac{\\sum_{i \\in N_k} w_i \\cdot \\mathbb{I}(y_i = c)}{\\sum_{i \\in N_k} w_i}',
      concreteCalculation: `Produced predictions for ${testX.length} test samples across target classes [${classes.join(', ')}].`,
    },
  ];

  return {
    k,
    metric,
    weights: weightMode,
    predictions,
    probabilities,
    classes,
    explanations,
  };
}
