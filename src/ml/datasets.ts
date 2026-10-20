import { ColumnMetadata, Dataset, TaskType } from './types.ts';

// Deterministic Pseudo-Random Number Generator (Mulberry32)
export function createRNG(seed: number = 42) {
  let s = Math.floor(seed) >>> 0;
  return function () {
    s |= 0;
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Standard normal distribution generator (Box-Muller transform)
export function normalRandom(rng: () => number, mean: number = 0, std: number = 1): number {
  let u1 = rng();
  let u2 = rng();
  while (u1 === 0) u1 = rng();
  while (u2 === 0) u2 = rng();
  const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  return z0 * std + mean;
}

// Simple deterministic hash for dataset fingerprinting
export function computeFingerprint(data: Record<string, any>[], columns: string[]): string {
  let str = columns.join(',');
  const sampleRows = Math.min(data.length, 50);
  for (let i = 0; i < sampleRows; i++) {
    const row = data[i];
    str += '|' + columns.map((col) => String(row[col] ?? '')).join(':');
  }
  str += `_rows=${data.length}_cols=${columns.length}`;

  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = (hash << 5) - hash + char;
    hash |= 0;
  }
  return 'fp_' + Math.abs(hash).toString(16) + '_' + data.length;
}

export function inspectColumns(data: Record<string, any>[], columns: string[]): Record<string, ColumnMetadata> {
  const meta: Record<string, ColumnMetadata> = {};

  for (const col of columns) {
    let missingCount = 0;
    const values: any[] = [];
    const numValues: number[] = [];

    for (const row of data) {
      const val = row[col];
      if (val === null || val === undefined || val === '' || Number.isNaN(val)) {
        missingCount++;
      } else {
        values.push(val);
        const num = Number(val);
        if (!isNaN(num) && typeof val !== 'boolean') {
          numValues.push(num);
        }
      }
    }

    const uniqueSet = new Set(values);
    // If >85% of non-missing values are valid numbers, classify as numerical
    const isNumerical = values.length > 0 && numValues.length / values.length >= 0.85;

    let mean: number | undefined;
    let min: number | undefined;
    let max: number | undefined;
    let std: number | undefined;

    if (isNumerical && numValues.length > 0) {
      min = Math.min(...numValues);
      max = Math.max(...numValues);
      const sum = numValues.reduce((a, b) => a + b, 0);
      mean = sum / numValues.length;
      const variance = numValues.reduce((a, b) => a + Math.pow(b - mean!, 2), 0) / (numValues.length || 1);
      std = Math.sqrt(variance);
    }

    meta[col] = {
      name: col,
      type: isNumerical ? 'numerical' : 'categorical',
      missingCount,
      uniqueCount: uniqueSet.size,
      sampleValues: Array.from(uniqueSet).slice(0, 5),
      mean: mean !== undefined ? Number(mean.toFixed(3)) : undefined,
      min: min !== undefined ? Number(min.toFixed(3)) : undefined,
      max: max !== undefined ? Number(max.toFixed(3)) : undefined,
      std: std !== undefined ? Number(std.toFixed(3)) : undefined,
    };
  }

  return meta;
}

// Robust CSV Parser
export function parseCSV(csvContent: string): { columns: string[]; data: Record<string, any>[] } {
  const lines = csvContent
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 0);

  if (lines.length < 2) {
    throw new Error('CSV file must contain at least a header row and one data row.');
  }

  // Helper to split row by comma taking quotes into account
  const parseRow = (line: string): string[] => {
    const fields: string[] = [];
    let current = '';
    let inQuotes = false;
    for (let i = 0; i < line.length; i++) {
      const char = line[i];
      if (char === '"' || char === "'") {
        if (inQuotes && line[i + 1] === char) {
          current += char;
          i++; // escaped quote
        } else {
          inQuotes = !inQuotes;
        }
      } else if (char === ',' && !inQuotes) {
        fields.push(current.trim());
        current = '';
      } else {
        current += char;
      }
    }
    fields.push(current.trim());
    return fields;
  };

  const headers = parseRow(lines[0]).map((h) => h.replace(/^["']|["']$/g, '').trim());
  if (new Set(headers).size !== headers.length) {
    throw new Error('CSV contains duplicate column headers.');
  }

  const data: Record<string, any>[] = [];

  for (let r = 1; r < lines.length; r++) {
    const vals = parseRow(lines[r]);
    if (vals.length === 1 && vals[0] === '') continue; // skip blank line

    const rowObj: Record<string, any> = {};
    for (let c = 0; c < headers.length; c++) {
      const colName = headers[c];
      const rawVal = vals[c] !== undefined ? vals[c].replace(/^["']|["']$/g, '').trim() : '';

      if (rawVal === '' || rawVal.toLowerCase() === 'nan' || rawVal.toLowerCase() === 'null') {
        rowObj[colName] = null;
      } else {
        const numVal = Number(rawVal);
        rowObj[colName] = !isNaN(numVal) && !isNaN(parseFloat(rawVal)) ? numVal : rawVal;
      }
    }
    data.push(rowObj);
  }

  return { columns: headers, data };
}

// -------------------------------------------------------------
// Built-in Datasets
// -------------------------------------------------------------

export function createIrisDataset(): Dataset {
  const rawIris = [
    [5.1, 3.5, 1.4, 0.2, 'setosa'],
    [4.9, 3.0, 1.4, 0.2, 'setosa'],
    [4.7, 3.2, 1.3, 0.2, 'setosa'],
    [4.6, 3.1, 1.5, 0.2, 'setosa'],
    [5.0, 3.6, 1.4, 0.2, 'setosa'],
    [5.4, 3.9, 1.7, 0.4, 'setosa'],
    [4.6, 3.4, 1.4, 0.3, 'setosa'],
    [5.0, 3.4, 1.5, 0.2, 'setosa'],
    [4.4, 2.9, 1.4, 0.2, 'setosa'],
    [4.9, 3.1, 1.5, 0.1, 'setosa'],
    [5.4, 3.7, 1.5, 0.2, 'setosa'],
    [4.8, 3.4, 1.6, 0.2, 'setosa'],
    [4.8, 3.0, 1.4, 0.1, 'setosa'],
    [4.3, 3.0, 1.1, 0.1, 'setosa'],
    [5.8, 4.0, 1.2, 0.2, 'setosa'],
    [5.7, 4.4, 1.5, 0.4, 'setosa'],
    [5.4, 3.9, 1.3, 0.4, 'setosa'],
    [5.1, 3.5, 1.4, 0.3, 'setosa'],
    [5.7, 3.8, 1.7, 0.3, 'setosa'],
    [5.1, 3.8, 1.5, 0.3, 'setosa'],
    [5.4, 3.4, 1.7, 0.2, 'setosa'],
    [5.1, 3.7, 1.5, 0.4, 'setosa'],
    [4.6, 3.6, 1.0, 0.2, 'setosa'],
    [5.1, 3.3, 1.7, 0.5, 'setosa'],
    [4.8, 3.4, 1.9, 0.2, 'setosa'],
    [5.0, 3.0, 1.6, 0.2, 'setosa'],
    [5.0, 3.4, 1.6, 0.4, 'setosa'],
    [5.2, 3.5, 1.5, 0.2, 'setosa'],
    [5.2, 3.4, 1.4, 0.2, 'setosa'],
    [4.7, 3.2, 1.6, 0.2, 'setosa'],
    [4.8, 3.1, 1.6, 0.2, 'setosa'],
    [5.4, 3.4, 1.5, 0.4, 'setosa'],
    [5.2, 4.1, 1.5, 0.1, 'setosa'],
    [5.5, 4.2, 1.4, 0.2, 'setosa'],
    [4.9, 3.1, 1.5, 0.2, 'setosa'],
    [5.0, 3.2, 1.2, 0.2, 'setosa'],
    [5.5, 3.5, 1.3, 0.2, 'setosa'],
    [4.9, 3.6, 1.4, 0.1, 'setosa'],
    [4.4, 3.0, 1.3, 0.2, 'setosa'],
    [5.1, 3.4, 1.5, 0.2, 'setosa'],
    [5.0, 3.5, 1.3, 0.3, 'setosa'],
    [4.5, 2.3, 1.3, 0.3, 'setosa'],
    [4.4, 3.2, 1.3, 0.2, 'setosa'],
    [5.0, 3.5, 1.6, 0.6, 'setosa'],
    [5.1, 3.8, 1.9, 0.4, 'setosa'],
    [4.8, 3.0, 1.4, 0.3, 'setosa'],
    [5.1, 3.8, 1.6, 0.2, 'setosa'],
    [4.6, 3.2, 1.4, 0.2, 'setosa'],
    [5.3, 3.7, 1.5, 0.2, 'setosa'],
    [5.0, 3.3, 1.4, 0.2, 'setosa'],
    [7.0, 3.2, 4.7, 1.4, 'versicolor'],
    [6.4, 3.2, 4.5, 1.5, 'versicolor'],
    [6.9, 3.1, 4.9, 1.5, 'versicolor'],
    [5.5, 2.3, 4.0, 1.3, 'versicolor'],
    [6.5, 2.8, 4.6, 1.5, 'versicolor'],
    [5.7, 2.8, 4.5, 1.3, 'versicolor'],
    [6.3, 3.3, 4.7, 1.6, 'versicolor'],
    [4.9, 2.4, 3.3, 1.0, 'versicolor'],
    [6.6, 2.9, 4.6, 1.3, 'versicolor'],
    [5.2, 2.7, 3.9, 1.4, 'versicolor'],
    [5.0, 2.0, 3.5, 1.0, 'versicolor'],
    [5.9, 3.0, 4.2, 1.5, 'versicolor'],
    [6.0, 2.2, 4.0, 1.0, 'versicolor'],
    [6.1, 2.9, 4.7, 1.4, 'versicolor'],
    [5.6, 2.9, 3.6, 1.3, 'versicolor'],
    [6.7, 3.1, 4.4, 1.4, 'versicolor'],
    [5.6, 3.0, 4.5, 1.5, 'versicolor'],
    [5.8, 2.7, 4.1, 1.0, 'versicolor'],
    [6.2, 2.2, 4.5, 1.5, 'versicolor'],
    [5.6, 2.5, 3.9, 1.1, 'versicolor'],
    [5.9, 3.2, 4.8, 1.8, 'versicolor'],
    [6.1, 2.8, 4.0, 1.3, 'versicolor'],
    [6.3, 2.5, 4.9, 1.5, 'versicolor'],
    [6.1, 2.8, 4.7, 1.2, 'versicolor'],
    [6.4, 2.9, 4.3, 1.3, 'versicolor'],
    [6.6, 3.0, 4.4, 1.4, 'versicolor'],
    [6.8, 2.8, 4.8, 1.4, 'versicolor'],
    [6.7, 3.0, 5.0, 1.7, 'versicolor'],
    [6.0, 2.9, 4.5, 1.5, 'versicolor'],
    [5.7, 2.6, 3.5, 1.0, 'versicolor'],
    [5.5, 2.4, 3.8, 1.1, 'versicolor'],
    [5.5, 2.4, 3.7, 1.0, 'versicolor'],
    [5.8, 2.7, 3.9, 1.2, 'versicolor'],
    [6.0, 2.7, 5.1, 1.6, 'versicolor'],
    [5.4, 3.0, 4.5, 1.5, 'versicolor'],
    [6.0, 3.4, 4.5, 1.6, 'versicolor'],
    [6.7, 3.1, 4.7, 1.5, 'versicolor'],
    [6.3, 2.3, 4.4, 1.3, 'versicolor'],
    [5.6, 3.0, 4.1, 1.3, 'versicolor'],
    [5.5, 2.5, 4.0, 1.3, 'versicolor'],
    [5.5, 2.6, 4.4, 1.2, 'versicolor'],
    [6.1, 3.0, 4.6, 1.4, 'versicolor'],
    [5.8, 2.6, 4.0, 1.2, 'versicolor'],
    [5.0, 2.3, 3.3, 1.0, 'versicolor'],
    [5.6, 2.7, 4.2, 1.3, 'versicolor'],
    [5.7, 3.0, 4.2, 1.2, 'versicolor'],
    [5.7, 2.9, 4.2, 1.3, 'versicolor'],
    [6.2, 2.9, 4.3, 1.3, 'versicolor'],
    [5.1, 2.5, 3.0, 1.1, 'versicolor'],
    [5.7, 2.8, 4.1, 1.3, 'versicolor'],
    [6.3, 3.3, 6.0, 2.5, 'virginica'],
    [5.8, 2.7, 5.1, 1.9, 'virginica'],
    [7.1, 3.0, 5.9, 2.1, 'virginica'],
    [6.3, 2.9, 5.6, 1.8, 'virginica'],
    [6.5, 3.0, 5.8, 2.2, 'virginica'],
    [7.6, 3.0, 6.6, 2.1, 'virginica'],
    [4.9, 2.5, 4.5, 1.7, 'virginica'],
    [7.3, 2.9, 6.3, 1.8, 'virginica'],
    [6.7, 2.5, 5.8, 1.8, 'virginica'],
    [7.2, 3.6, 6.1, 2.5, 'virginica'],
    [6.5, 3.2, 5.1, 2.0, 'virginica'],
    [6.4, 2.7, 5.3, 1.9, 'virginica'],
    [6.8, 3.0, 5.5, 2.1, 'virginica'],
    [5.7, 2.5, 5.0, 2.0, 'virginica'],
    [5.8, 2.8, 5.1, 2.4, 'virginica'],
    [6.4, 3.2, 5.3, 2.3, 'virginica'],
    [6.5, 3.0, 5.5, 1.8, 'virginica'],
    [7.7, 3.8, 6.7, 2.2, 'virginica'],
    [7.7, 2.6, 6.9, 2.3, 'virginica'],
    [6.0, 2.2, 5.0, 1.5, 'virginica'],
    [6.9, 3.2, 5.7, 2.3, 'virginica'],
    [5.6, 2.8, 4.9, 2.0, 'virginica'],
    [7.7, 2.8, 6.7, 2.0, 'virginica'],
    [6.3, 2.7, 4.9, 1.8, 'virginica'],
    [6.7, 3.3, 5.7, 2.1, 'virginica'],
    [7.2, 3.2, 6.0, 1.8, 'virginica'],
    [6.2, 2.8, 4.8, 1.8, 'virginica'],
    [6.1, 3.0, 4.9, 1.8, 'virginica'],
    [6.4, 2.8, 5.6, 2.1, 'virginica'],
    [7.2, 3.0, 5.8, 1.6, 'virginica'],
    [7.4, 2.8, 6.1, 1.9, 'virginica'],
    [7.9, 3.8, 6.4, 2.0, 'virginica'],
    [6.4, 2.8, 5.6, 2.2, 'virginica'],
    [6.3, 2.8, 5.1, 1.5, 'virginica'],
    [6.1, 2.6, 5.6, 1.4, 'virginica'],
    [7.7, 3.0, 6.1, 2.3, 'virginica'],
    [6.3, 3.4, 5.6, 2.4, 'virginica'],
    [6.4, 3.1, 5.5, 1.8, 'virginica'],
    [6.0, 3.0, 4.8, 1.8, 'virginica'],
    [6.9, 3.1, 5.4, 2.1, 'virginica'],
    [6.7, 3.1, 5.6, 2.4, 'virginica'],
    [6.9, 3.1, 5.1, 2.3, 'virginica'],
    [5.8, 2.7, 5.1, 1.9, 'virginica'],
    [6.8, 3.2, 5.9, 2.3, 'virginica'],
    [6.7, 3.3, 5.7, 2.5, 'virginica'],
    [6.7, 3.0, 5.2, 2.3, 'virginica'],
    [6.3, 2.5, 5.0, 1.9, 'virginica'],
    [6.5, 3.0, 5.2, 2.0, 'virginica'],
    [6.2, 3.4, 5.4, 2.3, 'virginica'],
    [5.9, 3.0, 5.1, 1.8, 'virginica'],
  ];

  const columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'];
  const data = rawIris.map((row) => ({
    sepal_length: row[0],
    sepal_width: row[1],
    petal_length: row[2],
    petal_width: row[3],
    species: row[4],
  }));

  return {
    id: 'sample-iris',
    name: 'Iris Flower Species',
    source: 'sample',
    fingerprint: computeFingerprint(data, columns),
    version: '1.0.0',
    createdAt: new Date().toISOString(),
    columns,
    columnMeta: inspectColumns(data, columns),
    data,
    suggestedTarget: 'species',
    suggestedTask: 'classification',
    description: 'Classic 150-sample benchmark dataset for classification with 4 morphological measurements.',
  };
}

export function createDiabetesDataset(): Dataset {
  const rng = createRNG(101);
  const data: Record<string, any>[] = [];
  const columns = ['age', 'bmi', 'blood_pressure', 'total_cholesterol', 'glucose', 'progression_index'];

  for (let i = 0; i < 150; i++) {
    const age = Math.round(normalRandom(rng, 48, 13));
    const bmi = Number(normalRandom(rng, 26.5, 4.4).toFixed(1));
    const bp = Math.round(normalRandom(rng, 94, 14));
    const chol = Math.round(normalRandom(rng, 189, 35));
    const glucose = Math.round(normalRandom(rng, 91, 11));

    // True underlying physiological relationship + observational noise
    const noise = normalRandom(rng, 0, 15);
    const progression = Math.max(
      25,
      Math.min(
        340,
        Math.round(0.8 * age + 3.2 * (bmi - 20) + 0.6 * bp + 0.3 * (chol - 150) + 1.1 * (glucose - 80) + noise)
      )
    );

    data.push({
      age,
      bmi,
      blood_pressure: bp,
      total_cholesterol: chol,
      glucose,
      progression_index: progression,
    });
  }

  return {
    id: 'sample-diabetes',
    name: 'Diabetes Disease Progression',
    source: 'sample',
    fingerprint: computeFingerprint(data, columns),
    version: '1.0.0',
    createdAt: new Date().toISOString(),
    columns,
    columnMeta: inspectColumns(data, columns),
    data,
    suggestedTarget: 'progression_index',
    suggestedTask: 'regression',
    description: 'Physiological measurements used to predict one-year disease progression in diabetes patients.',
  };
}

export function createMallCustomersDataset(): Dataset {
  const rng = createRNG(303);
  const data: Record<string, any>[] = [];
  const columns = ['age', 'annual_income_k', 'spending_score_1_100', 'gender'];

  // 4 distinct customer archetypes for K-Means clustering
  const archetypes = [
    { meanIncome: 25, meanSpend: 78, meanAge: 25, label: 'Young High Spenders' },
    { meanIncome: 28, meanSpend: 21, meanAge: 45, label: 'Frugal Budget Shoppers' },
    { meanIncome: 86, meanSpend: 82, meanAge: 32, label: 'Affluent VIP Spenders' },
    { meanIncome: 88, meanSpend: 17, meanAge: 52, label: 'High Income Savers' },
    { meanIncome: 55, meanSpend: 50, meanAge: 39, label: 'Balanced Mainstream' },
  ];

  for (let i = 0; i < 160; i++) {
    const arch = archetypes[i % archetypes.length];
    const income = Math.max(15, Math.round(normalRandom(rng, arch.meanIncome, 6)));
    const spend = Math.max(1, Math.min(99, Math.round(normalRandom(rng, arch.meanSpend, 8))));
    const age = Math.max(18, Math.min(70, Math.round(normalRandom(rng, arch.meanAge, 5))));
    const gender = rng() > 0.5 ? 'Female' : 'Male';

    data.push({
      age,
      annual_income_k: income,
      spending_score_1_100: spend,
      gender,
    });
  }

  return {
    id: 'sample-mall',
    name: 'Customer Mall Segmentation',
    source: 'sample',
    fingerprint: computeFingerprint(data, columns),
    version: '1.0.0',
    createdAt: new Date().toISOString(),
    columns,
    columnMeta: inspectColumns(data, columns),
    data,
    suggestedTask: 'clustering',
    description: 'Income, spending habit, and age data for unsupervised customer cluster profiling.',
  };
}

// -------------------------------------------------------------
// Synthetic Dataset Generators
// -------------------------------------------------------------

export function generateSyntheticRegression(options: {
  samples: number;
  features: number;
  noise: number;
  seed: number;
}): Dataset {
  const { samples, features, noise, seed } = options;
  const rng = createRNG(seed);
  const data: Record<string, any>[] = [];
  const columns: string[] = [];

  for (let f = 0; f < features; f++) {
    columns.push(`feature_${f + 1}`);
  }
  columns.push('target');

  // Random ground truth coefficients
  const trueWeights: number[] = [];
  for (let f = 0; f < features; f++) {
    trueWeights.push((rng() * 4 - 2) * 5); // between -10 and +10
  }
  const trueBias = (rng() * 2 - 1) * 10;

  for (let i = 0; i < samples; i++) {
    const row: Record<string, any> = {};
    let y = trueBias;

    for (let f = 0; f < features; f++) {
      const val = Number(normalRandom(rng, 0, 1.5).toFixed(2));
      row[`feature_${f + 1}`] = val;
      y += val * trueWeights[f];
    }

    const err = normalRandom(rng, 0, noise);
    row['target'] = Number((y + err).toFixed(2));
    data.push(row);
  }

  return {
    id: `synthetic-reg-${Date.now()}`,
    name: `Synthetic Regression (${samples} rows, ${features} features)`,
    source: 'synthetic',
    fingerprint: computeFingerprint(data, columns),
    version: '1.0.0',
    createdAt: new Date().toISOString(),
    columns,
    columnMeta: inspectColumns(data, columns),
    data,
    suggestedTarget: 'target',
    suggestedTask: 'regression',
    description: `Configured synthetic linear regression model with noise level ${noise} and seed ${seed}.`,
  };
}

export function generateSyntheticClassification(options: {
  samples: number;
  features: number;
  nClasses: number;
  classSeparation: number;
  noise: number;
  seed: number;
}): Dataset {
  const { samples, features, nClasses, classSeparation, noise, seed } = options;
  const rng = createRNG(seed);
  const data: Record<string, any>[] = [];
  const columns: string[] = [];

  for (let f = 0; f < features; f++) {
    columns.push(`feature_${f + 1}`);
  }
  columns.push('label');

  // Create class cluster centroids distributed around a hypersphere
  const classCenters: number[][] = [];
  for (let c = 0; c < nClasses; c++) {
    const angle = (2 * Math.PI * c) / nClasses;
    const center: number[] = [];
    for (let f = 0; f < features; f++) {
      if (f === 0) center.push(Math.cos(angle) * classSeparation * 2.5);
      else if (f === 1) center.push(Math.sin(angle) * classSeparation * 2.5);
      else center.push((rng() * 2 - 1) * classSeparation);
    }
    classCenters.push(center);
  }

  for (let i = 0; i < samples; i++) {
    const classIdx = i % nClasses;
    const center = classCenters[classIdx];
    const row: Record<string, any> = {};

    for (let f = 0; f < features; f++) {
      const val = center[f] + normalRandom(rng, 0, 1.0 + noise * 0.8);
      row[`feature_${f + 1}`] = Number(val.toFixed(2));
    }

    row['label'] = `Class_${String.fromCharCode(65 + classIdx)}`;
    data.push(row);
  }

  return {
    id: `synthetic-clf-${Date.now()}`,
    name: `Synthetic Classification (${samples} rows, ${nClasses} classes)`,
    source: 'synthetic',
    fingerprint: computeFingerprint(data, columns),
    version: '1.0.0',
    createdAt: new Date().toISOString(),
    columns,
    columnMeta: inspectColumns(data, columns),
    data,
    suggestedTarget: 'label',
    suggestedTask: 'classification',
    description: `Configured synthetic classification dataset with ${nClasses} classes, separation factor ${classSeparation}.`,
  };
}

export function generateSyntheticClustering(options: {
  samples: number;
  features: number;
  nClusters: number;
  clusterStd: number;
  seed: number;
}): Dataset {
  const { samples, features, nClusters, clusterStd, seed } = options;
  const rng = createRNG(seed);
  const data: Record<string, any>[] = [];
  const columns: string[] = [];

  for (let f = 0; f < features; f++) {
    columns.push(`feature_${f + 1}`);
  }

  // Create cluster centers
  const centers: number[][] = [];
  for (let k = 0; k < nClusters; k++) {
    const angle = (2 * Math.PI * k) / nClusters;
    const center: number[] = [];
    for (let f = 0; f < features; f++) {
      if (f === 0) center.push(Math.cos(angle) * 6);
      else if (f === 1) center.push(Math.sin(angle) * 6);
      else center.push((rng() * 2 - 1) * 4);
    }
    centers.push(center);
  }

  for (let i = 0; i < samples; i++) {
    const kIdx = i % nClusters;
    const center = centers[kIdx];
    const row: Record<string, any> = {};

    for (let f = 0; f < features; f++) {
      const val = center[f] + normalRandom(rng, 0, clusterStd);
      row[`feature_${f + 1}`] = Number(val.toFixed(2));
    }
    data.push(row);
  }

  return {
    id: `synthetic-clt-${Date.now()}`,
    name: `Synthetic Clustering (${samples} points, ${nClusters} blobs)`,
    source: 'synthetic',
    fingerprint: computeFingerprint(data, columns),
    version: '1.0.0',
    createdAt: new Date().toISOString(),
    columns,
    columnMeta: inspectColumns(data, columns),
    data,
    suggestedTask: 'clustering',
    description: `Generated ${nClusters} distinct clusters with cluster std deviation ${clusterStd}.`,
  };
}
