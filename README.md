# Mineral Deposit Classification

This project provides a web-based dashboard for a DVC pipeline that classifies mineral deposits. The dashboard, built with Streamlit, allows users to interact with and visualize the pipeline's stages.

## Demo
[![Demo Video](https://img.youtube.com/vi/TZHFRYZLA9k/maxresdefault.jpg)](https://youtu.be/TZHFRYZLA9k)

*Demo showing the interactive dashboard for mineral deposit classification*

## Manual Installation

Follow these steps for a manual setup.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/mghanava/mineral-classifier-app.git
    cd mineral-classifier-app
    ```

2.  **Build and run the container:**
    A `run.sh` script is provided to simplify the setup process. It builds the container and sets the correct file permissions for DVC to prevent errors.

    First, make the script executable:
    ```bash
    chmod +x run.sh
    ```

    Now, run the script:
    ```bash
    ./run.sh
    ```

### Usage

-   **Streamlit Dashboard:** Once the container is running, open your web browser and navigate to `http://localhost:8501`.
-   **Development Shell:** To access an interactive shell inside the running container, use:
    ```bash
    docker-compose exec mineral_classifier bash
    ```

### Stopping the Application

To stop and remove the container and attached volumes, run:

```bash
    docker-compose down -v
```

### Project Structure

```
├── app.py                    # The Streamlit dashboard application
├── setup_dvc.py              # DVC setup script
├── run.sh                    # bash file to build and run container
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