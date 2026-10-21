import type { Experiment, ModelArtifact } from './types.ts';

const EXPERIMENTS_KEY = 'ml_sim_experiments_v1';
const MODELS_KEY = 'ml_sim_models_v1';

export function getStoredExperiments(): Experiment[] {
  try {
    const raw = localStorage.getItem(EXPERIMENTS_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

export function saveExperiment(experiment: Experiment): void {
  try {
    const existing = getStoredExperiments();
    const filtered = existing.filter((e) => e.id !== experiment.id);
    filtered.unshift(experiment);
    localStorage.setItem(EXPERIMENTS_KEY, JSON.stringify(filtered.slice(0, 50)));
  } catch (err) {
    console.error('Failed to save experiment to storage:', err);
  }
}

export function getStoredModels(): ModelArtifact[] {
  try {
    const raw = localStorage.getItem(MODELS_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

export function saveModelArtifact(model: ModelArtifact): void {
  try {
    const existing = getStoredModels();
    const filtered = existing.filter((m) => m.id !== model.id);
    filtered.unshift(model);
    localStorage.setItem(MODELS_KEY, JSON.stringify(filtered.slice(0, 30)));
  } catch (err) {
    console.error('Failed to save model to storage:', err);
  }
}

export function updateModelStatus(modelId: string, status: 'draft' | 'staging' | 'production'): void {
  try {
    const models = getStoredModels();
    const model = models.find((m) => m.id === modelId);
    if (model) {
      model.status = status;
      localStorage.setItem(MODELS_KEY, JSON.stringify(models));
    }
  } catch (err) {
    console.error('Failed to update model status:', err);
  }
}

export function clearExperimentHistory(): void {
  try {
    localStorage.removeItem(EXPERIMENTS_KEY);
  } catch (err) {
    console.error('Failed to clear experiment history:', err);
  }
}
