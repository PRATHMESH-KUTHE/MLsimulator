import type { AlgorithmStepExplanation, Hyperparameters } from '../types.ts';

export interface LogisticRegressionResult {
  weights: number[][]; // One vector per class (for OvR)
  biases: number[];
  classes: string[];
  predictions: string[];
  probabilities: number[][];
  stepTrace: {
    epoch: number;
    loss: number;
  }[];
  explanations: AlgorithmStepExplanation[];
}

export function sigmoid(z: number): number {
  if (z >= 40) return 1.0;
  if (z <= -40) return 0.0;
  return 1.0 / (1.0 + Math.exp(-z));
}

export function trainLogisticRegression(
  trainX: number[][],
  trainY: (number | string)[],
  testX: number[][],
  featureNames: string[],
  hyperparams: Hyperparameters = {}
): LogisticRegressionResult {
  const n = trainX.length;
  const p = trainX[0]?.length || 0;
  const classes = Array.from(new Set(trainY.map(String))).sort();
  const nClasses = classes.length;

  const lr = hyperparams.learningRate ?? 0.1;
  const epochs = hyperparams.epochs ?? 100;
  const C = hyperparams.regularizationC ?? 1.0; // Inverse of regularization strength (1/lambda)
  const l2Penalty = 1 / (C * n);

  const allWeights: number[][] = [];
  const allBiases: number[] = [];
  const totalLossHistory: number[] = new Array(epochs).fill(0);

  // Train binary classifier for each class (One-vs-Rest for multiclass)
  for (let c = 0; c < (nClasses === 2 ? 1 : nClasses); c++) {
    const targetClass = classes[c];
    // Binary labels: 1 if matches target class, 0 otherwise
    const yBin = trainY.map((y) => (String(y) === targetClass ? 1 : 0));

    let w = new Array(p).fill(0);
    let b = 0;

    for (let ep = 0; ep < epochs; ep++) {
      let gradB = 0;
      const gradW = new Array(p).fill(0);
      let loss = 0;

      for (let i = 0; i < n; i++) {
        let z = b;
        for (let j = 0; j < p; j++) {
          z += w[j] * trainX[i][j];
        }
        const yPredProb = sigmoid(z);
        const err = yPredProb - yBin[i];

        gradB += err;
        for (let j = 0; j < p; j++) {
          gradW[j] += err * trainX[i][j];
        }

        // Cross-entropy log loss
        const eps = 1e-12;
        loss += -(yBin[i] * Math.log(yPredProb + eps) + (1 - yBin[i]) * Math.log(1 - yPredProb + eps));
      }

      // Add L2 penalty to gradients
      for (let j = 0; j < p; j++) {
        gradW[j] = gradW[j] / n + l2Penalty * w[j];
        w[j] -= lr * gradW[j];
      }
      b -= (lr * gradB) / n;

      totalLossHistory[ep] += loss / n;
    }

    allWeights.push(w);
    allBiases.push(b);
  }

  // Handle binary case where single boundary separates class 0 and class 1
  if (nClasses === 2) {
    // Class 0 weights are -w, Class 1 weights are +w
    allWeights.push(allWeights[0].map((v) => -v));
    allBiases.push(-allBiases[0]);
  }

  // Predict on test set
  const predictProbabilities = (row: number[]): number[] => {
    if (nClasses === 2) {
      let z = allBiases[0];
      for (let j = 0; j < p; j++) {
        z += allWeights[0][j] * row[j];
      }
      const p1 = sigmoid(z);
      return [Number(p1.toFixed(4)), Number((1 - p1).toFixed(4))];
    } else {
      // Softmax over OvR logits
      const logits: number[] = [];
      for (let c = 0; c < nClasses; c++) {
        let z = allBiases[c];
        for (let j = 0; j < p; j++) {
          z += allWeights[c][j] * row[j];
        }
        logits.push(z);
      }
      const maxLogit = Math.max(...logits);
      const exps = logits.map((l) => Math.exp(l - maxLogit));
      const sumExp = exps.reduce((a, b) => a + b, 0);
      return exps.map((e) => Number((e / sumExp).toFixed(4)));
    }
  };

  const probabilities = testX.map(predictProbabilities);
  const predictions = probabilities.map((probs) => {
    let maxIdx = 0;
    for (let c = 1; c < probs.length; c++) {
      if (probs[c] > probs[maxIdx]) maxIdx = c;
    }
    return classes[maxIdx];
  });

  // Step trace
  const traceSamples = Math.min(epochs, 20);
  const stepInterval = Math.max(1, Math.floor(epochs / traceSamples));
  const stepTrace = [];
  for (let ep = 0; ep < epochs; ep += stepInterval) {
    stepTrace.push({
      epoch: ep + 1,
      loss: Number((totalLossHistory[ep] / allWeights.length).toFixed(4)),
    });
  }

  const explanations: AlgorithmStepExplanation[] = [
    {
      stepNumber: 1,
      title: 'Log-Odds & Sigmoid Activation',
      concept: 'Logistic regression maps a linear combination of features to a probability between 0 and 1 via the logistic sigmoid function.',
      formula: 'P(y = 1 | \\mathbf{x}) = \\sigma(\\mathbf{w}^T \\mathbf{x} + b) = \\frac{1}{1 + e^{-(\\mathbf{w}^T \\mathbf{x} + b)}}',
      concreteCalculation: `Trained ${nClasses} classes over ${n} samples with learning rate ${lr} and L2 parameter C = ${C}.`,
    },
    {
      stepNumber: 2,
      title: 'Log Loss (Binary Cross-Entropy) Optimization',
      concept: 'Minimizes the negative log-likelihood of correct class probabilities using gradient descent.',
      formula: 'J(\\mathbf{w}, b) = -\\frac{1}{N} \\sum_{i=1}^N [y_i \\log(\\hat{p}_i) + (1 - y_i) \\log(1 - \\hat{p}_i)] + \\frac{1}{2C} \\|\\mathbf{w}\\|^2',
      concreteCalculation: `Initial loss settled from ${totalLossHistory[0]?.toFixed(3)} down to ${totalLossHistory[epochs - 1]?.toFixed(3)}.`,
    },
    {
      stepNumber: 3,
      title: 'Decision Boundary Separation',
      concept: 'The classification boundary lies where P(y=1) = 0.5, corresponding to linear hyperplanes where z = w^T x + b = 0.',
      formula: '\\mathbf{w}^T \\mathbf{x} + b = 0',
      concreteCalculation: `Class predictions decided by argmax over normalized softmax probabilities across classes [${classes.join(', ')}].`,
    },
  ];

  return {
    weights: allWeights.map((w) => w.map((v) => Number(v.toFixed(4)))),
    biases: allBiases.map((b) => Number(b.toFixed(4))),
    classes,
    predictions,
    probabilities,
    stepTrace,
    explanations,
  };
}
