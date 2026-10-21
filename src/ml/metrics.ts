import type { ClassificationMetrics, ClusteringMetrics, RegressionMetrics } from './types.ts';

// -------------------------------------------------------------
// Regression Metrics
// -------------------------------------------------------------
export function evaluateRegression(
  actual: number[],
  predicted: number[],
  trainActual?: number[],
  trainPredicted?: number[]
): RegressionMetrics {
  const n = actual.length;
  if (n === 0) {
    return { mae: 0, mse: 0, rmse: 0, r2: 0 };
  }

  let absErrorSum = 0;
  let sqErrorSum = 0;
  const meanActual = actual.reduce((a, b) => a + b, 0) / n;
  let totSumSquares = 0;

  for (let i = 0; i < n; i++) {
    const err = actual[i] - predicted[i];
    absErrorSum += Math.abs(err);
    sqErrorSum += err * err;
    const diffMean = actual[i] - meanActual;
    totSumSquares += diffMean * diffMean;
  }

  const mae = absErrorSum / n;
  const mse = sqErrorSum / n;
  const rmse = Math.sqrt(mse);
  const r2 = totSumSquares > 1e-9 ? Math.max(-2, 1 - sqErrorSum / totSumSquares) : 0;

  let trainR2: number | undefined;
  if (trainActual && trainPredicted && trainActual.length > 0) {
    const trainN = trainActual.length;
    const trainMean = trainActual.reduce((a, b) => a + b, 0) / trainN;
    let trTot = 0;
    let trSq = 0;
    for (let i = 0; i < trainN; i++) {
      const e = trainActual[i] - trainPredicted[i];
      trSq += e * e;
      const d = trainActual[i] - trainMean;
      trTot += d * d;
    }
    trainR2 = trTot > 1e-9 ? Math.max(-2, 1 - trSq / trTot) : 0;
  }

  return {
    mae: Number(mae.toFixed(4)),
    mse: Number(mse.toFixed(4)),
    rmse: Number(rmse.toFixed(4)),
    r2: Number(r2.toFixed(4)),
    trainR2: trainR2 !== undefined ? Number(trainR2.toFixed(4)) : undefined,
  };
}

// -------------------------------------------------------------
// Classification Metrics
// -------------------------------------------------------------
export function evaluateClassification(
  actual: (string | number)[],
  predicted: (string | number)[],
  classList?: (string | number)[]
): ClassificationMetrics {
  const n = actual.length;
  if (n === 0) {
    return {
      accuracy: 0,
      precision: 0,
      recall: 0,
      f1Score: 0,
      confusionMatrix: { labels: [], matrix: [] },
      report: {},
    };
  }

  const labels = classList
    ? classList.map(String)
    : Array.from(new Set([...actual, ...predicted].map(String))).sort();

  // Initialize confusion matrix
  const matrix: number[][] = Array.from({ length: labels.length }, () =>
    new Array(labels.length).fill(0)
  );

  let correct = 0;
  for (let i = 0; i < n; i++) {
    const actStr = String(actual[i]);
    const predStr = String(predicted[i]);
    const actIdx = labels.indexOf(actStr);
    const predIdx = labels.indexOf(predStr);

    if (actIdx >= 0 && predIdx >= 0) {
      matrix[actIdx][predIdx]++;
    }
    if (actStr === predStr) {
      correct++;
    }
  }

  const accuracy = correct / n;
  const report: ClassificationMetrics['report'] = {};
  let totalPrecision = 0;
  let totalRecall = 0;
  let totalF1 = 0;

  for (let i = 0; i < labels.length; i++) {
    const label = labels[i];
    const tp = matrix[i][i];
    let fn = 0;
    for (let j = 0; j < labels.length; j++) {
      if (j !== i) fn += matrix[i][j];
    }
    let fp = 0;
    for (let j = 0; j < labels.length; j++) {
      if (j !== i) fp += matrix[j][i];
    }

    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
    const support = tp + fn;

    report[label] = {
      precision: Number(precision.toFixed(3)),
      recall: Number(recall.toFixed(3)),
      f1Score: Number(f1.toFixed(3)),
      support,
    };

    totalPrecision += precision;
    totalRecall += recall;
    totalF1 += f1;
  }

  const macroPrecision = totalPrecision / (labels.length || 1);
  const macroRecall = totalRecall / (labels.length || 1);
  const macroF1 = totalF1 / (labels.length || 1);

  return {
    accuracy: Number(accuracy.toFixed(4)),
    precision: Number(macroPrecision.toFixed(4)),
    recall: Number(macroRecall.toFixed(4)),
    f1Score: Number(macroF1.toFixed(4)),
    confusionMatrix: {
      labels,
      matrix,
    },
    report,
  };
}

// -------------------------------------------------------------
// Clustering Metrics
// -------------------------------------------------------------
export function evaluateClustering(
  X: number[][],
  labels: number[],
  centroids: number[][]
): ClusteringMetrics {
  const n = X.length;
  const k = centroids.length;

  if (n === 0 || k === 0) {
    return { inertia: 0, silhouetteScore: 0, clusterCounts: [] };
  }

  // 1. Inertia (Within-cluster sum of squares)
  let inertia = 0;
  const clusterCounts: number[] = new Array(k).fill(0);

  for (let i = 0; i < n; i++) {
    const cluster = labels[i];
    clusterCounts[cluster] = (clusterCounts[cluster] || 0) + 1;
    const center = centroids[cluster];
    if (center) {
      let distSq = 0;
      for (let j = 0; j < X[i].length; j++) {
        const d = X[i][j] - center[j];
        distSq += d * d;
      }
      inertia += distSq;
    }
  }

  // 2. Silhouette Score calculation (subsample if dataset is large for latency budget)
  const maxSilhouetteSamples = Math.min(n, 200);
  const stride = Math.max(1, Math.floor(n / maxSilhouetteSamples));
  let silhouetteTotal = 0;
  let sampledCount = 0;

  for (let i = 0; i < n; i += stride) {
    const cOwn = labels[i];
    if (clusterCounts[cOwn] <= 1) continue; // single point cluster

    // a(i): mean intra-cluster distance
    let aSum = 0;
    let aCount = 0;

    // b(i): minimum mean distance to points in another cluster
    const bSums: number[] = new Array(k).fill(0);
    const bCounts: number[] = new Array(k).fill(0);

    for (let j = 0; j < n; j += stride) {
      if (i === j) continue;
      let dist = 0;
      for (let f = 0; f < X[i].length; f++) {
        const diff = X[i][f] - X[j][f];
        dist += diff * diff;
      }
      dist = Math.sqrt(dist);

      if (labels[j] === cOwn) {
        aSum += dist;
        aCount++;
      } else {
        const cOther = labels[j];
        bSums[cOther] += dist;
        bCounts[cOther]++;
      }
    }

    const a_i = aCount > 0 ? aSum / aCount : 0;
    let b_i = Infinity;
    for (let c = 0; c < k; c++) {
      if (c !== cOwn && bCounts[c] > 0) {
        const meanDistOther = bSums[c] / bCounts[c];
        if (meanDistOther < b_i) b_i = meanDistOther;
      }
    }
    if (b_i === Infinity) b_i = 0;

    const max_ab = Math.max(a_i, b_i);
    const s_i = max_ab > 0 ? (b_i - a_i) / max_ab : 0;

    silhouetteTotal += s_i;
    sampledCount++;
  }

  const avgSilhouette = sampledCount > 0 ? silhouetteTotal / sampledCount : 0;

  return {
    inertia: Number(inertia.toFixed(2)),
    silhouetteScore: Number(avgSilhouette.toFixed(4)),
    clusterCounts,
  };
}
