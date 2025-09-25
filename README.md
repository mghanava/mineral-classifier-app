# Mineral Deposit Classification

This project provides a web-based dashboard for a DVC pipeline that classifies mineral deposits. The dashboard, built with Streamlit, allows users to interact with and visualize the pipeline's stages.

## Demo
[![Demo Video](https://img.youtube.com/vi/TZHFRYZLA9k/maxresdefault.jpg)](https://youtu.be/TZHFRYZLA9k)

*Demo showing the interactive dashboard for mineral deposit classification*

## Core Concepts and Workflow

This project implements an online learning pipeline for mineral deposit classification using graph neural networks. The workflow is designed to simulate a real-world scenario where new data becomes available over time, and the model needs to adapt and generalize. The key stages are:

1.  **Bootstrap Stage**:
    -   Initial data samples are generated (`src/stages/generate_base_data.py`).
    -   A graph is constructed from these samples using methods such as k-nearest neighbors (KNN), a distance threshold, or a distance percentile (`src/utilities/data_utils.py`). The constructed graph is then split for cross-validation into multiple train-validation graphs, a test graph for evaluation, and a final graph for model calibration, ensuring no data leakage through nodes or edges.   

2.  **Training and Evaluation**:
    -   The pipeline supports different graph neural network architectures (GCN and GAT) for training (`src/stages/train.py`) on the constructed graph (`src/models`).
    -   The trained model is evaluated (`src/stages/evaluate.py`), and its output probabilities are calibrated using various techniques (such as temperature scaling, isotonic regression, Platt scaling, beta calibration, or Dirichlet calibration) to ensure they are reliable and not overconfident.

3.  **Online Learning Cycles**:
    -   **New Data Simulation**: New, unlabeled data samples are introduced to simulate incoming data streams (`src/stages/generate_pred_data.py`).
    -   **Prediction**: The calibrated model is used to predict the labels of the new samples (`src/stages/predict.py`).
    -   **Drift Analysis**: A detailed drift analysis is performed (`src/stages/analyze_drift.py`) to detect any domain shift between the original data and the new samples. The approach includes:
        -   Comparing feature distributions one-by-one (marginal distributions).
        -   Analyzing feature interactions using a mutual information matrix.
        -   Visualizing the data structure in lower dimensions using PCA and KernelPCA.
        -   Conducting formal statistical tests using permutation techniques (Maximum Mean Discrepancy, Energy Distance, and Wasserstein Distance) to quantify any detected drift.
    -   **Data Integration**: The new, labeled samples are combined with the original bootstrap data (`src/stages/prepare_next_cycle_data.py`).
    -   This cycle of training, prediction, and integration allows the model to continuously learn and adapt.

4.  **Performance Analysis**:
    -   Finally, `src/stages/analyze_cycles_performance.py` is used to analyze and compare the model's performance across different learning cycles, tracking its improvement and generalization capabilities over time.

## Installation

Follow these steps for a manual setup.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/mghanava/mineral-classifier-app.git
    cd mineral-classifier-app
    ```

2.  **Build and run the container:**
    Use Docker Compose to build the image and start the service. The command passes your host user's ID to the container to prevent file permission errors.
    ```bash
    DOCKER_BUILDKIT=1 UID=$(id -u) GID=$(id -g) docker-compose up -d --build
    ```

### Usage and Pipeline Execution

The dashboard provides an interactive way to configure and run the DVC pipeline.

-   **Streamlit Dashboard:** Once the container is running, open your web browser and navigate to `http://localhost:8501`.
-   **Development Shell:** To access an interactive shell inside the running container, use:
    ```bash
    docker-compose exec mineral_classifier bash
    ```

#### Workflow

1.  **Configure Parameters**: Navigate to the **Parameters** tab. Here, you can view and modify all pipeline settings from `params.yaml`. This includes parameters for data generation, model architecture, training, evaluation, and drift analysis.

2.  **Generate Pipeline File**: After making your desired changes, press the **Generate dvc.yaml** button. This crucial step runs the `setup_dvc.py` script to create the `dvc.yaml` pipeline file based on your new configuration.

3.  **Run the Pipeline**:
    -   **Full Pipeline Run**: To execute the entire pipeline, use the main "Run Pipeline" feature in the dashboard, which runs `dvc repro`.
    -   **Targeted Stage Run**: Each pipeline stage (e.g., Evaluate, Predict, Analyze Drift) has its own tab. If you only change parameters affecting a specific stage, you can navigate to that stage's tab and run it individually. DVC intelligently detects what has changed and will only re-execute the necessary parts of the pipeline.
    
    For example, to experiment with a different calibration technique, you would change the `evaluate` parameters in the **Parameters** tab and save changes, and then go to the **Evaluation** tab to run just that stage.

### DVC Pipeline

This project uses [DVC (Data Version Control)](https://dvc.org/) to create a reproducible machine learning pipeline. The entire workflow is defined in `dvc.yaml` as a series of connected stages. Each stage includes:
-   `cmd`: The command to execute.
-   `deps`: Dependencies, such as source code and input data.
-   `params`: Parameters from `params.yaml` that affect the stage.
-   `outs`: Outputs, such as data files or models.

The pipeline is structured as a series of cycles to facilitate online learning. It begins with a `bootstrap` stage to generate initial data, followed by repeating cycles of training, evaluation, prediction, drift analysis, and data integration. The final stage, `analyze_cycles_performance`, evaluates the model's performance across all cycles.

To run the pipeline in the development shell, first run `python setup_dvc.py` to generate the `dvc.yaml` file based on the number of cycles provided in `params.yaml`. Then, run the pipeline with:
```bash
dvc repro
```
DVC will automatically track dependencies and run only the stages that have changed.

### Stopping the Application

To stop and remove the container and attached volumes, run:

```bash
    docker-compose down -v
```

### Project Structure

```
├── app.py                    # The Streamlit dashboard application
├── setup_dvc.py              # DVC setup script
├── entrypoint.sh             # Sets correct user permissions at container startup
├── docker-compose.yaml       # Docker Compose configuration
├── Dockerfile                # Dockerfile for the application image
├── dvc.yaml                  # DVC pipeline definition
├── params.yaml               # Parameters for the DVC pipeline
├── pyproject.toml            # Python dependencies and ruff formatter setup
└── src/                      # Source code for the pipeline stages
    ├── models/
    ├── stages/
    └── utilities/
```