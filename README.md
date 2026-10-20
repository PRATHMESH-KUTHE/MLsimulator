# ML Algorithm Simulator & MLOps Studio

An educational, interactive machine learning simulation platform and end-to-end MLOps lifecycle studio.

## Features

1. **Dataset Studio**:
   - Curated benchmarks: Iris Flower (Classification), Diabetes Progression (Regression), Mall Customer Segments (Clustering).
   - Configurable Synthetic Generators: `make_regression`, `make_classification`, and `make_blobs` with noise, feature count, sample count, and random seed.
   - CSV File Upload: Drag-and-drop parsing, column type inference (numerical vs categorical), missing value detection, and target selection.

2. **Data Preprocessing & Leak Prevention**:
   - Imputation: Mean, Median, Mode, or Row Removal.
   - Outlier treatment: IQR 1.5× Winsorizing.
   - Categorical Encoding: OneHotEncoder and LabelEncoder.
   - Feature Scaling: StandardScaler and MinMaxScaler.
   - **Zero Data Leakage Contract**: Scalers and encoders fit strictly on the training partition and are frozen for the test partition and production inference.
   - Train/Test Split slider with deterministic seeds.

3. **Algorithm Lab & Real Model Training**:
   - **Linear Regression**: Ordinary Least Squares (Normal Equation) & Gradient Descent step trace.
   - **Logistic Regression**: Sigmoid activation, Binary Cross-Entropy log loss, and One-vs-Rest multiclass.
   - **Decision Tree Classifier & Regressor**: CART recursive partitioning with Gini impurity, Entropy, and MSE variance reduction.
   - **K-Nearest Neighbors (KNN)**: Euclidean & Manhattan distance with distance-weighted majority voting.
   - **Random Forest Classifier & Regressor**: Bootstrap aggregating (bagging) with feature importance (Mean Decrease in Impurity).
   - **K-Means Clustering**: Lloyd's algorithm with K-Means++ initialization, Voronoi partitioning, and step-by-step centroid convergence.

4. **Interactive Simulation & Visualization**:
   - Linear Regression 2D scatter with fitted line, equation card, and residual error bars.
   - Hierarchical SVG Decision Tree with interactive sample branch path tracing.
   - K-Means step playback scrubber (Play, Pause, Step Forward, Reset) with centroid movement trails.
   - PCA 2D projection for high-dimensional feature spaces.
   - Text-to-Speech (TTS) audio narration for learning concepts using the Web Speech Synthesis API.

5. **MLOps Lifecycle**:
   - **Data Versioning**: Fingerprint hashing and schema tracking.
   - **Experiment Tracking**: Side-by-side run comparison, metrics delta, and 1-click reproducibility rerun.
   - **Model Registry**: Staging / Production status flags, schema contracts, and artifact export (`model_artifact.json`).
   - **Live Inference Studio**: Dynamic form for real-time model scoring and downloadable Python inference script.
   - **Covariate Drift Simulator**: Compares reference training statistics against production query streams with automated drift detection alerts and 1-click retraining.

## Running Locally

### Full-Stack (Node.js & Express)
```bash
npm install
npm run dev
```
Runs on `http://localhost:3000`.

### Python Backend (Optional FastAPI Microservice)
```bash
cd backend
pip install -r requirements.txt
python main.py
```
Runs on `http://localhost:8000`.
