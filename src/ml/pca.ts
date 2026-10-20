/**
 * Principal Component Analysis (PCA)
 * Computes 2D projection of multidimensional vectors using Power Iteration on Covariance Matrix.
 */

export interface PCAResult {
  projectedX: [number, number][];
  components: number[][]; // 2 principal component eigenvectors
  explainedVarianceRatio: [number, number];
  mean: number[];
}

export function computePCA2D(X: number[][]): PCAResult {
  const n = X.length;
  if (n === 0) {
    return {
      projectedX: [],
      components: [[], []],
      explainedVarianceRatio: [0, 0],
      mean: [],
    };
  }

  const p = X[0].length;
  if (p === 1) {
    return {
      projectedX: X.map((row) => [row[0], 0]),
      components: [[1], [0]],
      explainedVarianceRatio: [1.0, 0.0],
      mean: [X.reduce((a, b) => a + b[0], 0) / n],
    };
  }

  // 1. Mean-center the data
  const mean: number[] = new Array(p).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < p; j++) {
      mean[j] += X[i][j];
    }
  }
  for (let j = 0; j < p; j++) {
    mean[j] /= n;
  }

  const centered: number[][] = X.map((row) => row.map((val, j) => val - mean[j]));

  // 2. Compute Covariance Matrix (p x p)
  const cov: number[][] = Array.from({ length: p }, () => new Array(p).fill(0));
  for (let i = 0; i < p; i++) {
    for (let j = 0; j < p; j++) {
      let sum = 0;
      for (let k = 0; k < n; k++) {
        sum += centered[k][i] * centered[k][j];
      }
      cov[i][j] = sum / (n > 1 ? n - 1 : 1);
    }
  }

  // Total variance (trace of covariance matrix)
  let totalVariance = 0;
  for (let i = 0; i < p; i++) {
    totalVariance += cov[i][i];
  }
  if (totalVariance <= 0) totalVariance = 1e-9;

  // Helper for power iteration to find dominant eigenvector and eigenvalue
  const powerIteration = (
    matrix: number[][],
    deflateVector?: number[]
  ): { vector: number[]; eigenvalue: number } => {
    let v: number[] = new Array(p).fill(0).map((_, i) => (i % 2 === 0 ? 1 : 0.5));
    // Normalize initial vector
    let norm = Math.sqrt(v.reduce((a, b) => a + b * b, 0)) || 1;
    v = v.map((x) => x / norm);

    const maxIters = 60;
    for (let iter = 0; iter < maxIters; iter++) {
      // w = matrix * v
      let w: number[] = new Array(p).fill(0);
      for (let i = 0; i < p; i++) {
        for (let j = 0; j < p; j++) {
          w[i] += matrix[i][j] * v[j];
        }
      }

      // If deflating against previous eigenvector, project out
      if (deflateVector) {
        const dot = w.reduce((acc, val, idx) => acc + val * deflateVector[idx], 0);
        w = w.map((val, idx) => val - dot * deflateVector[idx]);
      }

      norm = Math.sqrt(w.reduce((a, b) => a + b * b, 0));
      if (norm < 1e-9) break;
      v = w.map((x) => x / norm);
    }

    // Rayleigh quotient for eigenvalue: lambda = v^T * matrix * v
    let lambda = 0;
    for (let i = 0; i < p; i++) {
      let rowDot = 0;
      for (let j = 0; j < p; j++) {
        rowDot += matrix[i][j] * v[j];
      }
      lambda += v[i] * rowDot;
    }

    return { vector: v, eigenvalue: Math.max(0, lambda) };
  };

  const pc1 = powerIteration(cov);
  const pc2 = powerIteration(cov, pc1.vector);

  const evr1 = Math.min(1.0, Math.max(0, pc1.eigenvalue / totalVariance));
  const evr2 = Math.min(1.0 - evr1, Math.max(0, pc2.eigenvalue / totalVariance));

  // 3. Project data onto PC1 and PC2
  const projectedX: [number, number][] = centered.map((row) => {
    let p1 = 0;
    let p2 = 0;
    for (let j = 0; j < p; j++) {
      p1 += row[j] * pc1.vector[j];
      p2 += row[j] * pc2.vector[j];
    }
    return [Number(p1.toFixed(3)), Number(p2.toFixed(3))];
  });

  return {
    projectedX,
    components: [pc1.vector, pc2.vector],
    explainedVarianceRatio: [Number(evr1.toFixed(3)), Number(evr2.toFixed(3))],
    mean,
  };
}
