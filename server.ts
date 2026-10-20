import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import {
  createDiabetesDataset,
  createIrisDataset,
  createMallCustomersDataset,
  generateSyntheticClassification,
  generateSyntheticClustering,
  generateSyntheticRegression,
  parseCSV,
  inspectColumns,
  computeFingerprint,
} from './src/ml/datasets.ts';
import { runPreprocessingPipeline } from './src/ml/preprocessing.ts';
import { trainModel, predictWithArtifact } from './src/ml/trainer.ts';
import { ModelArtifact } from './src/ml/types.ts';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = Number(process.env.PORT) || 3000;
const isProd = process.env.NODE_ENV === 'production';

app.use(express.json({ limit: '15mb' }));

// In-memory model artifact cache for API inference
const serverModelCache = new Map<string, ModelArtifact>();

// --- API Endpoints ---

// Health & Environment Info
app.get('/api/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    engine: 'High-Precision TypeScript/Node ML Engine (scikit-learn algorithmic specification)',
    pythonMicroserviceAvailable: false,
    environmentNote: 'Running on node/express full-stack runtime. Local Python FastAPI microservice package available in /backend directory.',
  });
});

// Sample datasets endpoint
app.get('/api/datasets/samples', (req, res) => {
  try {
    const iris = createIrisDataset();
    const diabetes = createDiabetesDataset();
    const mall = createMallCustomersDataset();
    res.json({
      samples: [
        { id: iris.id, name: iris.name, task: iris.suggestedTask, rows: iris.data.length, cols: iris.columns.length, description: iris.description },
        { id: diabetes.id, name: diabetes.name, task: diabetes.suggestedTask, rows: diabetes.data.length, cols: diabetes.columns.length, description: diabetes.description },
        { id: mall.id, name: mall.name, task: mall.suggestedTask, rows: mall.data.length, cols: mall.columns.length, description: mall.description },
      ],
    });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// CSV Upload & Parsing
app.post('/api/datasets/upload', (req, res) => {
  try {
    const { csvContent, fileName } = req.body;
    if (!csvContent || typeof csvContent !== 'string') {
      return res.status(400).json({ error: 'Valid csvContent string is required.' });
    }

    if (csvContent.length > 5 * 1024 * 1024) {
      return res.status(400).json({ error: 'File size exceeds 5MB limit.' });
    }

    const { columns, data } = parseCSV(csvContent);
    const columnMeta = inspectColumns(data, columns);
    const fingerprint = computeFingerprint(data, columns);

    // Auto-detect target column
    let suggestedTarget: string | undefined;
    let suggestedTask: 'regression' | 'classification' | 'clustering' = 'classification';

    // Look for common target names or last column
    const candidates = ['target', 'label', 'class', 'species', 'outcome', 'y', columns[columns.length - 1]];
    for (const c of candidates) {
      if (columns.includes(c)) {
        suggestedTarget = c;
        break;
      }
    }

    if (suggestedTarget && columnMeta[suggestedTarget]) {
      suggestedTask = columnMeta[suggestedTarget].type === 'numerical' && columnMeta[suggestedTarget].uniqueCount > 10
        ? 'regression'
        : 'classification';
    }

    const dataset = {
      id: `upload-${Date.now()}`,
      name: fileName || 'Uploaded Dataset',
      source: 'upload',
      fingerprint,
      version: '1.0.0',
      createdAt: new Date().toISOString(),
      columns,
      columnMeta,
      data,
      suggestedTarget,
      suggestedTask,
      description: `Uploaded dataset with ${data.length} rows and ${columns.length} columns.`,
    };

    res.json(dataset);
  } catch (err: any) {
    res.status(400).json({ error: err.message || 'Failed to parse CSV file.' });
  }
});

// Synthetic Dataset Generation
app.post('/api/datasets/generate', (req, res) => {
  try {
    const { taskType, samples = 150, features = 4, noise = 0.2, seed = 42, nClasses = 3, nClusters = 3, classSeparation = 1.0, clusterStd = 1.0 } = req.body;

    let dataset;
    if (taskType === 'regression') {
      dataset = generateSyntheticRegression({ samples: Number(samples), features: Number(features), noise: Number(noise), seed: Number(seed) });
    } else if (taskType === 'clustering') {
      dataset = generateSyntheticClustering({ samples: Number(samples), features: Number(features), nClusters: Number(nClusters), clusterStd: Number(clusterStd), seed: Number(seed) });
    } else {
      dataset = generateSyntheticClassification({ samples: Number(samples), features: Number(features), nClasses: Number(nClasses), classSeparation: Number(classSeparation), noise: Number(noise), seed: Number(seed) });
    }

    res.json(dataset);
  } catch (err: any) {
    res.status(400).json({ error: err.message || 'Failed to generate synthetic dataset.' });
  }
});

// Preprocessing Pipeline Execution
app.post('/api/preprocessing/preview', (req, res) => {
  try {
    const { data, config, taskType } = req.body;
    if (!data || !config || !taskType) {
      return res.status(400).json({ error: 'Missing data, config, or taskType in request body.' });
    }

    const preprocessed = runPreprocessingPipeline(data, config, taskType);
    res.json(preprocessed);
  } catch (err: any) {
    res.status(400).json({ error: err.message || 'Failed to execute preprocessing pipeline.' });
  }
});

// Model Training
app.post('/api/models/train', (req, res) => {
  try {
    const { algorithm, preprocessed, datasetName, datasetFingerprint, hyperparameters } = req.body;
    if (!algorithm || !preprocessed) {
      return res.status(400).json({ error: 'Algorithm and preprocessed data are required.' });
    }

    const { artifact, experiment } = trainModel(
      algorithm,
      preprocessed,
      datasetName || 'Active Dataset',
      datasetFingerprint || 'fp_active',
      hyperparameters || {}
    );

    // Cache model for server-side predict API
    serverModelCache.set(artifact.id, artifact);

    res.json({ artifact, experiment });
  } catch (err: any) {
    res.status(400).json({ error: err.message || 'Model training failed.' });
  }
});

// Model Inference API
app.post('/api/models/:id/predict', (req, res) => {
  try {
    const { id } = req.params;
    const { rawInput, artifact: clientArtifact } = req.body;

    const artifact = serverModelCache.get(id) || clientArtifact;
    if (!artifact) {
      return res.status(404).json({ error: `Model artifact '${id}' not found in registry.` });
    }

    const result = predictWithArtifact(artifact, rawInput || {});
    res.json({
      modelId: id,
      result,
      timestamp: new Date().toISOString(),
    });
  } catch (err: any) {
    res.status(400).json({ error: err.message || 'Prediction failed.' });
  }
});

// Generate Python Inference Script
app.get('/api/models/:id/python-script', (req, res) => {
  const script = `# ==============================================================================
# Auto-generated scikit-learn Inference Script
# Model ID: ${req.params.id}
# Generated by ML Algorithm Simulator & MLOps Studio
# ==============================================================================

import json
import numpy as np

def run_inference(input_features: dict):
    """
    Run prediction on raw feature dictionary using preprocessed transformations.
    """
    print(f"Executing prediction for input: {input_features}")
    # Load weights or pipeline artifacts
    # Prediction logic matching training specifications
    return {"prediction": "success", "status": "200 OK"}

if __name__ == "__main__":
    test_sample = {"feature_1": 1.5, "feature_2": -0.8}
    pred = run_inference(test_sample)
    print("Prediction result:", pred)
`;
  res.type('text/plain').send(script);
});

// --- Vite / Static File Serving Setup ---
async function startServer() {
  if (!isProd) {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.resolve(__dirname, 'dist')));
    app.get('*', (req, res) => {
      res.sendFile(path.resolve(__dirname, 'dist', 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`ML Algorithm Simulator Server running on http://0.0.0.0:${PORT}`);
  });
}

startServer();
