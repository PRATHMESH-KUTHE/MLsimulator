import type { PreprocessedData, PreprocessingConfig, PreprocessingSummary, TaskType } from './types.ts';
import { createRNG } from './datasets.ts';

export function runPreprocessingPipeline(
  rawData: Record<string, any>[],
  config: PreprocessingConfig,
  taskType: TaskType
): PreprocessedData {
  const originalRowCount = rawData.length;
  const originalColumnCount = Object.keys(rawData[0] || {}).length;

  let currentData = rawData.map((r) => ({ ...r }));
  let removedDuplicatesCount = 0;
  let imputedValuesCount = 0;
  let removedOutliersCount = 0;

  // 1. Duplicate Row Removal
  if (config.removeDuplicates) {
    const seen = new Set<string>();
    const deduplicated: Record<string, any>[] = [];
    for (const row of currentData) {
      const key = JSON.stringify(row);
      if (!seen.has(key)) {
        seen.add(key);
        deduplicated.push(row);
      } else {
        removedDuplicatesCount++;
      }
    }
    currentData = deduplicated;
  }

  // 2. Separate Target (if supervised) and check rows
  const targetCol = config.targetColumn;
  if (taskType !== 'clustering') {
    if (!targetCol) {
      throw new Error('A target column must be specified for supervised learning tasks.');
    }
    // Drop rows with missing target values
    const beforeCount = currentData.length;
    currentData = currentData.filter(
      (r) => r[targetCol] !== null && r[targetCol] !== undefined && r[targetCol] !== ''
    );
    if (currentData.length < 5) {
      throw new Error('Too few valid data samples remaining after filtering missing target rows.');
    }
  }

  // 3. Train/Test Split BEFORE fitting any learned transformations (prevent data leakage!)
  const rng = createRNG(config.randomSeed);
  const indices = currentData.map((_, i) => i);

  // Fisher-Yates shuffle with deterministic RNG
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }

  const testCount =
    taskType === 'clustering'
      ? 0
      : Math.max(1, Math.min(Math.floor(indices.length * config.testSplitRatio), indices.length - 2));
  const trainIndices = indices.slice(testCount);
  const testIndices = indices.slice(0, testCount);

  let trainRows = trainIndices.map((idx) => ({ ...currentData[idx] }));
  let testRows = testIndices.map((idx) => ({ ...currentData[idx] }));

  // 4. Missing Value Imputation (Fitted on Train Only)
  const imputerValues: Record<string, number | string> = {};
  const selectedFeatures = config.selectedFeatures.filter((f) => f !== targetCol);

  for (const feature of selectedFeatures) {
    const trainVals = trainRows
      .map((r) => r[feature])
      .filter((v) => v !== null && v !== undefined && v !== '' && !Number.isNaN(v));

    const isNum = trainVals.length > 0 && trainVals.every((v) => typeof v === 'number');

    if (config.missingStrategy === 'drop') {
      // Handled via filtering
      trainRows = trainRows.filter(
        (r) => r[feature] !== null && r[feature] !== undefined && r[feature] !== ''
      );
      testRows = testRows.filter(
        (r) => r[feature] !== null && r[feature] !== undefined && r[feature] !== ''
      );
    } else if (isNum) {
      const numbers = trainVals as number[];
      if (numbers.length === 0) {
        imputerValues[feature] = 0;
      } else if (config.missingStrategy === 'mean') {
        imputerValues[feature] = numbers.reduce((a, b) => a + b, 0) / numbers.length;
      } else if (config.missingStrategy === 'median') {
        const sorted = [...numbers].sort((a, b) => a - b);
        const mid = Math.floor(sorted.length / 2);
        imputerValues[feature] =
          sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
      } else {
        // Mode (most frequent)
        const freq: Record<number, number> = {};
        for (const n of numbers) freq[n] = (freq[n] || 0) + 1;
        let bestVal = numbers[0];
        let maxFreq = 0;
        for (const [vStr, count] of Object.entries(freq)) {
          if (count > maxFreq) {
            maxFreq = count;
            bestVal = Number(vStr);
          }
        }
        imputerValues[feature] = bestVal;
      }
    } else {
      // Categorical Mode
      const freq: Record<string, number> = {};
      for (const val of trainVals) {
        const s = String(val);
        freq[s] = (freq[s] || 0) + 1;
      }
      let bestVal = trainVals[0] !== undefined ? String(trainVals[0]) : 'Unknown';
      let maxFreq = 0;
      for (const [k, count] of Object.entries(freq)) {
        if (count > maxFreq) {
          maxFreq = count;
          bestVal = k;
        }
      }
      imputerValues[feature] = bestVal;
    }

    // Apply imputation to train and test
    if (config.missingStrategy !== 'drop') {
      const fillVal = imputerValues[feature];
      for (const r of trainRows) {
        if (r[feature] === null || r[feature] === undefined || r[feature] === '' || Number.isNaN(r[feature])) {
          r[feature] = fillVal;
          imputedValuesCount++;
        }
      }
      for (const r of testRows) {
        if (r[feature] === null || r[feature] === undefined || r[feature] === '' || Number.isNaN(r[feature])) {
          r[feature] = fillVal;
          imputedValuesCount++;
        }
      }
    }
  }

  // 5. Outlier Detection & Treatment (IQR rule on numerical features, fitted on train)
  if (config.outlierTreatment !== 'none') {
    for (const feature of selectedFeatures) {
      const trainNums = trainRows
        .map((r) => Number(r[feature]))
        .filter((v) => !isNaN(v))
        .sort((a, b) => a - b);

      if (trainNums.length >= 8) {
        const q1 = trainNums[Math.floor(trainNums.length * 0.25)];
        const q3 = trainNums[Math.floor(trainNums.length * 0.75)];
        const iqr = q3 - q1;
        const lowerBound = q1 - 1.5 * iqr;
        const upperBound = q3 + 1.5 * iqr;

        if (config.outlierTreatment === 'iqr_clip') {
          for (const r of trainRows) {
            const v = Number(r[feature]);
            if (!isNaN(v)) {
              if (v < lowerBound) {
                r[feature] = lowerBound;
                removedOutliersCount++;
              } else if (v > upperBound) {
                r[feature] = upperBound;
                removedOutliersCount++;
              }
            }
          }
          for (const r of testRows) {
            const v = Number(r[feature]);
            if (!isNaN(v)) {
              if (v < lowerBound) r[feature] = lowerBound;
              else if (v > upperBound) r[feature] = upperBound;
            }
          }
        }
      }
    }
  }

  // 6. Categorical Encoding (OneHot or Label, fitted on train)
  const encoderMappings: Record<string, Record<string, number>> = {};
  const encodedFeaturesList: string[] = [];
  const transformedFeatureNames: string[] = [];

  for (const feature of selectedFeatures) {
    const isCat = trainRows.some(
      (r) => typeof r[feature] === 'string' && isNaN(Number(r[feature]))
    );

    if (isCat && config.encodingStrategy === 'onehot') {
      encodedFeaturesList.push(feature);
      const uniqueCats = Array.from(new Set(trainRows.map((r) => String(r[feature])))).sort();
      encoderMappings[feature] = {};
      uniqueCats.forEach((cat, idx) => {
        encoderMappings[feature][cat] = idx;
        transformedFeatureNames.push(`${feature}_${cat}`);
      });
    } else if (isCat && config.encodingStrategy === 'label') {
      encodedFeaturesList.push(feature);
      const uniqueCats = Array.from(new Set(trainRows.map((r) => String(r[feature])))).sort();
      encoderMappings[feature] = {};
      uniqueCats.forEach((cat, idx) => {
        encoderMappings[feature][cat] = idx;
      });
      transformedFeatureNames.push(feature);
    } else {
      transformedFeatureNames.push(feature);
    }
  }

  // Helper to transform row object into numerical vector
  const vectorizeRow = (row: Record<string, any>): number[] => {
    const vec: number[] = [];
    for (const feature of selectedFeatures) {
      if (encoderMappings[feature] && config.encodingStrategy === 'onehot') {
        const valStr = String(row[feature]);
        const categories = Object.keys(encoderMappings[feature]);
        for (const cat of categories) {
          vec.push(valStr === cat ? 1 : 0);
        }
      } else if (encoderMappings[feature] && config.encodingStrategy === 'label') {
        const valStr = String(row[feature]);
        vec.push(encoderMappings[feature][valStr] ?? -1);
      } else {
        vec.push(Number(row[feature]) || 0);
      }
    }
    return vec;
  };

  let trainX = trainRows.map(vectorizeRow);
  let testX = testRows.map(vectorizeRow);

  // 7. Feature Scaling (Fitted on Train split only!)
  let scalerParams: PreprocessedData['scalerParams'] = undefined;
  if (config.scalingStrategy !== 'none' && trainX.length > 0) {
    const nFeatures = transformedFeatureNames.length;
    const means: number[] = new Array(nFeatures).fill(0);
    const stds: number[] = new Array(nFeatures).fill(0);
    const mins: number[] = new Array(nFeatures).fill(Infinity);
    const maxs: number[] = new Array(nFeatures).fill(-Infinity);

    for (let f = 0; f < nFeatures; f++) {
      const colVals = trainX.map((r) => r[f]);
      const sum = colVals.reduce((a, b) => a + b, 0);
      const mean = sum / colVals.length;
      means[f] = mean;

      const variance = colVals.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / (colVals.length || 1);
      stds[f] = Math.sqrt(variance) || 1e-7;

      mins[f] = Math.min(...colVals);
      maxs[f] = Math.max(...colVals);
      if (mins[f] === maxs[f]) maxs[f] += 1e-7;
    }

    scalerParams = { means, stds, mins, maxs };

    const scaleVector = (vec: number[]): number[] => {
      return vec.map((val, f) => {
        if (config.scalingStrategy === 'standard') {
          return (val - means[f]) / stds[f];
        } else {
          return (val - mins[f]) / (maxs[f] - mins[f]);
        }
      });
    };

    trainX = trainX.map(scaleVector);
    testX = testX.map(scaleVector);
  }

  // Extract Targets
  let trainY: (number | string)[] = [];
  let testY: (number | string)[] = [];
  let classes: (string | number)[] | undefined = undefined;

  if (taskType !== 'clustering') {
    trainY = trainRows.map((r) => r[targetCol]);
    testY = testRows.map((r) => r[targetCol]);

    if (taskType === 'classification') {
      const uniqueClasses = Array.from(new Set(trainY)).sort();
      classes = uniqueClasses;
    }
  }

  const summary: PreprocessingSummary = {
    originalRowCount,
    processedRowCount: trainRows.length + testRows.length,
    originalColumnCount,
    processedColumnCount: transformedFeatureNames.length + (taskType !== 'clustering' ? 1 : 0),
    imputedValuesCount,
    removedDuplicatesCount,
    removedOutliersCount,
    encodedFeaturesList,
    featureNames: transformedFeatureNames,
    trainSize: trainRows.length,
    testSize: testRows.length,
  };

  return {
    config,
    summary,
    featureNames: transformedFeatureNames,
    targetName: targetCol,
    taskType,
    classes,
    train: {
      X: trainX,
      y: trainY,
      rawIndices: trainIndices,
    },
    test: {
      X: testX,
      y: testY,
      rawIndices: testIndices,
    },
    scalerParams,
    encoderMappings,
    imputerValues,
  };
}
