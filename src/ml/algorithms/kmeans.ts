import type { AlgorithmStepExplanation, Hyperparameters } from '../types.ts';
import { createRNG } from '../datasets.ts';

export interface KMeansResult {
  k: number;
  centroids: number[][];
  iterationCentroids: number[][][]; // Centroids at each step t
  assignments: number[];
  iterationAssignments: number[][]; // Assignments at each step t
  convergedIn: number;
  inertia: number;
  explanations: AlgorithmStepExplanation[];
}

export function trainKMeans(
  X: number[][],
  hyperparams: Hyperparameters = {}
): KMeansResult {
  const k = Math.max(2, Math.min(8, hyperparams.nClusters ?? 3));
  const maxIters = hyperparams.maxIterations ?? 20;
  const seed = hyperparams.randomSeed ?? 42;
  const rng = createRNG(seed);

  const n = X.length;
  const p = X[0]?.length || 0;

  if (n < k) {
    throw new Error(`Sample count (${n}) is smaller than number of clusters k (${k}).`);
  }

  // 1. K-Means++ Initialization
  const centroids: number[][] = [];
  // First center chosen uniformly at random
  const firstIdx = Math.floor(rng() * n);
  centroids.push([...X[firstIdx]]);

  const distSq = (a: number[], b: number[]): number => {
    let sum = 0;
    for (let j = 0; j < a.length; j++) {
      const d = a[j] - b[j];
      sum += d * d;
    }
    return sum;
  };

  for (let c = 1; c < k; c++) {
    // Distance from each point to nearest chosen centroid
    const minDistsSq: number[] = [];
    let sumDistsSq = 0;

    for (let i = 0; i < n; i++) {
      let minDist = Infinity;
      for (let j = 0; j < centroids.length; j++) {
        const d = distSq(X[i], centroids[j]);
        if (d < minDist) minDist = d;
      }
      minDistsSq.push(minDist);
      sumDistsSq += minDist;
    }

    // Weighted random selection proportional to D(x)^2
    let r = rng() * sumDistsSq;
    let chosenIdx = 0;
    for (let i = 0; i < n; i++) {
      r -= minDistsSq[i];
      if (r <= 0) {
        chosenIdx = i;
        break;
      }
    }
    centroids.push([...X[chosenIdx]]);
  }

  // Centroid history across iterations for animation
  const iterationCentroids: number[][][] = [centroids.map((c) => [...c])];
  const iterationAssignments: number[][] = [];

  let currentCentroids = centroids.map((c) => [...c]);
  let currentAssignments: number[] = new Array(n).fill(0);
  let convergedIn = maxIters;

  // 2. Lloyd's Iteration Loop
  for (let iter = 0; iter < maxIters; iter++) {
    // Assignment Step: assign each point to nearest centroid
    let hasChanged = false;
    const newAssignments: number[] = [];

    for (let i = 0; i < n; i++) {
      let minDist = Infinity;
      let bestCluster = 0;
      for (let c = 0; c < k; c++) {
        const d = distSq(X[i], currentCentroids[c]);
        if (d < minDist) {
          minDist = d;
          bestCluster = c;
        }
      }
      if (newAssignments.length === 0 || bestCluster !== currentAssignments[i]) {
        hasChanged = true;
      }
      newAssignments.push(bestCluster);
    }

    currentAssignments = newAssignments;
    iterationAssignments.push([...currentAssignments]);

    // Update Step: recompute centroid positions as mean of points
    const sums: number[][] = Array.from({ length: k }, () => new Array(p).fill(0));
    const counts: number[] = new Array(k).fill(0);

    for (let i = 0; i < n; i++) {
      const c = currentAssignments[i];
      counts[c]++;
      for (let j = 0; j < p; j++) {
        sums[c][j] += X[i][j];
      }
    }

    let maxShift = 0;
    const nextCentroids: number[][] = [];

    for (let c = 0; c < k; c++) {
      if (counts[c] === 0) {
        // Re-seed orphan cluster to random point
        const randPt = X[Math.floor(rng() * n)];
        nextCentroids.push([...randPt]);
      } else {
        const newCenter = sums[c].map((s) => s / counts[c]);
        const shift = Math.sqrt(distSq(currentCentroids[c], newCenter));
        if (shift > maxShift) maxShift = shift;
        nextCentroids.push(newCenter);
      }
    }

    currentCentroids = nextCentroids;
    iterationCentroids.push(currentCentroids.map((c) => [...c]));

    if (maxShift < 1e-4) {
      convergedIn = iter + 1;
      break;
    }
  }

  // Final Inertia
  let inertia = 0;
  for (let i = 0; i < n; i++) {
    const c = currentAssignments[i];
    inertia += distSq(X[i], currentCentroids[c]);
  }

  const explanations: AlgorithmStepExplanation[] = [
    {
      stepNumber: 1,
      title: 'K-Means++ Initialization',
      concept: 'Centroids are initialized sequentially using distance-proportional probabilities to spread initial seeds widely across the dataset.',
      formula: 'P(\\mathbf{x}) = \\frac{D(\\mathbf{x})^2}{\\sum_{\\mathbf{x}^\\prime} D(\\mathbf{x}^\\prime)^2}',
      concreteCalculation: `Seeded ${k} initial centroids from ${n} data points with random seed ${seed}.`,
      intermediateValues: { initialCentroids: iterationCentroids[0] },
    },
    {
      stepNumber: 2,
      title: 'Voronoi Partition Assignment Step',
      concept: 'Every observation is assigned to its closest centroid in Euclidean space, generating Voronoi cells.',
      formula: 'S_i^{(t)} = \\{ \\mathbf{x}_p : \\|\\mathbf{x}_p - \\mathbf{m}_i^{(t)}\\|^2 \\le \\|\\mathbf{x}_p - \\mathbf{m}_j^{(t)}\\|^2 \\quad \\forall j \\}',
      concreteCalculation: `Partitioned ${n} points into ${k} clusters over ${convergedIn} iterations until centroid shift reached < 10^-4.`,
    },
    {
      stepNumber: 3,
      title: 'Center of Mass Update & Convergence',
      concept: 'Centroids move to the barycenter (mean) of all points belonging to their cluster, iteratively minimizing Within-Cluster Sum of Squares.',
      formula: '\\mathbf{m}_i^{(t+1)} = \\frac{1}{|S_i^{(t)}|} \\sum_{\\mathbf{x}_j \\in S_i^{(t)}} \\mathbf{x}_j',
      concreteCalculation: `Final Within-Cluster Sum of Squares (Inertia) = ${inertia.toFixed(2)}.`,
    },
  ];

  return {
    k,
    centroids: currentCentroids.map((c) => c.map((v) => Number(v.toFixed(3)))),
    iterationCentroids: iterationCentroids.map((step) =>
      step.map((c) => c.map((v) => Number(v.toFixed(3))))
    ),
    assignments: currentAssignments,
    iterationAssignments,
    convergedIn,
    inertia: Number(inertia.toFixed(2)),
    explanations,
  };
}
