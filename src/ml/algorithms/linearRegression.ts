import { AlgorithmStepExplanation, Hyperparameters } from '../types.ts';

export interface LinearRegressionResult {
  coefficients: number[];
  intercept: number;
  equation: string;
  predictions: number[];
  trainPredictions: number[];
  stepTrace: {
    epoch: number;
    weights: number[];
    bias: number;
    cost: number;
  }[];
  explanations: AlgorithmStepExplanation[];
}

export function trainLinearRegression(
  trainX: number[][],
  trainY: (number | string)[],
  testX: number[][],
  featureNames: string[],
  hyperparams: Hyperparameters = {}
): LinearRegressionResult {
  const yTrain = trainY.map(Number);
  const n = trainX.length;
  const p = trainX[0]?.length || 0;

  const lr = hyperparams.learningRate ?? 0.05;
  const epochs = hyperparams.epochs ?? 100;
  const fitIntercept = hyperparams.fitIntercept ?? true;

  // 1. Solve using Ridge / OLS Normal Equation (Exact analytical solution)
  // X_augmented = [1, x1, x2...] if fitIntercept
  const lambdaRidge = 1e-4; // Numerical stability
  const cols = fitIntercept ? p + 1 : p;

  // Compute X^T * X and X^T * y
  const XtX: number[][] = Array.from({ length: cols }, () => new Array(cols).fill(0));
  const Xty: number[] = new Array(cols).fill(0);

  for (let i = 0; i < n; i++) {
    const row = fitIntercept ? [1, ...trainX[i]] : trainX[i];
    const yi = yTrain[i];
    for (let r = 0; r < cols; r++) {
      Xty[r] += row[r] * yi;
      for (let c = 0; c < cols; c++) {
        XtX[r][c] += row[r] * row[c];
      }
    }
  }

  // Add L2 ridge regularization to diagonal
  for (let r = (fitIntercept ? 1 : 0); r < cols; r++) {
    XtX[r][r] += lambdaRidge;
  }

  // Solve XtX * theta = Xty using Gaussian elimination with partial pivoting
  const A = XtX.map((r) => [...r]);
  const b = [...Xty];

  for (let i = 0; i < cols; i++) {
    let maxRow = i;
    for (let k = i + 1; k < cols; k++) {
      if (Math.abs(A[k][i]) > Math.abs(A[maxRow][i])) {
        maxRow = k;
      }
    }
    [A[i], A[maxRow]] = [A[maxRow], A[i]];
    [b[i], b[maxRow]] = [b[maxRow], b[i]];

    const pivot = A[i][i] || 1e-9;
    for (let k = i + 1; k < cols; k++) {
      const factor = A[k][i] / pivot;
      for (let j = i; j < cols; j++) {
        A[k][j] -= factor * A[i][j];
      }
      b[k] -= factor * b[i];
    }
  }

  const theta: number[] = new Array(cols).fill(0);
  for (let i = cols - 1; i >= 0; i--) {
    let sum = b[i];
    for (let j = i + 1; j < cols; j++) {
      sum -= A[i][j] * theta[j];
    }
    theta[i] = sum / (A[i][i] || 1e-9);
  }

  const intercept = fitIntercept ? theta[0] : 0;
  const coefficients = fitIntercept ? theta.slice(1) : theta;

  // 2. Generate Gradient Descent Step Trace for Educational Animation
  const stepTrace: LinearRegressionResult['stepTrace'] = [];
  let wTrace = new Array(p).fill(0);
  let bTrace = 0;
  const traceSamples = Math.min(epochs, 25);
  const epochStep = Math.max(1, Math.floor(epochs / traceSamples));

  for (let ep = 1; ep <= epochs; ep++) {
    let gradB = 0;
    const gradW = new Array(p).fill(0);
    let cost = 0;

    for (let i = 0; i < n; i++) {
      let pred = bTrace;
      for (let j = 0; j < p; j++) {
        pred += wTrace[j] * trainX[i][j];
      }
      const err = pred - yTrain[i];
      cost += err * err;
      gradB += err;
      for (let j = 0; j < p; j++) {
        gradW[j] += err * trainX[i][j];
      }
    }

    cost = cost / (2 * n);
    if (fitIntercept) bTrace -= (lr * gradB) / n;
    for (let j = 0; j < p; j++) {
      wTrace[j] -= (lr * gradW[j]) / n;
    }

    if (ep % epochStep === 0 || ep === epochs) {
      stepTrace.push({
        epoch: ep,
        weights: wTrace.map((w) => Number(w.toFixed(4))),
        bias: Number(bTrace.toFixed(4)),
        cost: Number(cost.toFixed(4)),
      });
    }
  }

  // 3. Inference helpers
  const predictRow = (row: number[]): number => {
    let val = intercept;
    for (let j = 0; j < row.length; j++) {
      val += (coefficients[j] || 0) * row[j];
    }
    return Number(val.toFixed(3));
  };

  const predictions = testX.map(predictRow);
  const trainPredictions = trainX.map(predictRow);

  // Formulate human-readable equation
  const equationTerms = coefficients
    .map((c, i) => `${c >= 0 && i > 0 ? '+ ' : ''}${c.toFixed(3)} × ${featureNames[i] || `X${i + 1}`}`)
    .join(' ');
  const equation = `ŷ = ${equationTerms} ${intercept >= 0 ? '+ ' : '- '}${Math.abs(intercept).toFixed(3)}`;

  // Educational Explanations
  const explanations: AlgorithmStepExplanation[] = [
    {
      stepNumber: 1,
      title: 'Problem Formulation & Objective',
      concept: 'Linear Regression finds the optimal linear hyperplane that minimizes the Sum of Squared Residuals (SSR).',
      formula: 'SSR = \\sum_{i=1}^{N} (y_i - (\\mathbf{w}^T \\mathbf{x}_i + b))^2',
      concreteCalculation: `Fitting ${p} features over ${n} training samples to minimize average squared distance to the line.`,
      intermediateValues: { sampleCount: n, featureCount: p },
    },
    {
      stepNumber: 2,
      title: 'Ordinary Least Squares (OLS) Solution',
      concept: 'The global analytical minimum is obtained via the Normal Equation setting the gradient of SSR to zero.',
      formula: '\\mathbf{\\hat{\\theta}} = (\\mathbf{X}^T \\mathbf{X} + \\lambda \\mathbf{I})^{-1} \\mathbf{X}^T \\mathbf{y}',
      concreteCalculation: `Computed intercept b = ${intercept.toFixed(3)}, primary weight w1 = ${(coefficients[0] || 0).toFixed(3)}.`,
      intermediateValues: { coefficients, intercept },
    },
    {
      stepNumber: 3,
      title: 'Residual Analysis & Prediction',
      concept: 'Predictions are generated through linear combination. The vertical discrepancy between actual y and predicted ŷ represents the residual error.',
      formula: 'e_i = y_i - \\hat{y}_i',
      concreteCalculation: `Model produced equation: ${equation}`,
    },
  ];

  return {
    coefficients: coefficients.map((c) => Number(c.toFixed(4))),
    intercept: Number(intercept.toFixed(4)),
    equation,
    predictions,
    trainPredictions,
    stepTrace,
    explanations,
  };
}
