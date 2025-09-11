# Mineral Deposit Classification

This project provides a web-based dashboard for a DVC pipeline that classifies mineral deposits. The dashboard, built with Streamlit, allows users to interact with and visualize the pipeline's stages, from data generation to model evaluation and prediction.

## Prerequisites

Before you begin, ensure you have the following installed on your system:

-   **Docker:** [Get Docker](https://docs.docker.com/get-docker/)
-   **Docker Compose:** [Install Docker Compose](https://docs.docker.com/compose/install/)
-   **NVIDIA Drivers & NVIDIA Container Toolkit:** Required for GPU support.
    -   [Install NVIDIA Drivers](https://www.nvidia.com/Download/index.aspx)
    -   [Install NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Getting Started

Follow these steps to get the project up and running:

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd mineral_deposit_classification
    ```

2.  **Build and run the container:**
    Use Docker Compose to build the image and start the service in detached mode:
    ```bash
    docker-compose up -d --build
    ```
    This command will start the Streamlit application, which will be accessible in your web browser.

## Usage

### Accessing the Application

-   **Streamlit Dashboard:** Once the container is running, open your web browser and navigate to `http://localhost:8501`.
-   **Development Shell:** To access an interactive shell inside the running container for debugging or running DVC commands directly, use the following command:
    ```bash
    docker-compose exec mineral_classifier bash
    ```

### Stopping the Application

To stop and remove the container and attached volumes, run:
```bash
docker-compose down -v
```

## Project Structure

```
├── app.py                    # The Streamlit dashboard application
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
