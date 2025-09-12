# Mineral Deposit Classification

This project provides a web-based dashboard for a DVC pipeline that classifies mineral deposits. The dashboard, built with Streamlit, allows users to interact with and visualize the pipeline's stages.

![App Demo GIF](https://raw.githubusercontent.com/mghanava/mineral-deposit-classification/main/docs/app-demo.gif)

## 🚀 Quickstart Demo

This is the fastest way to get the demo up and running on your machine.

1.  **Ensure Prerequisites:** Make sure you have [Docker](https://docs.docker.com/get-docker/), [Docker Compose](https://docs.docker.com/compose/install/), and [DVC](https://dvc.org/doc/install) installed. If you have a GPU, ensure the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) is also set up.

2.  **Run the Demo Script:**
    Clone the repository, `cd` into it, and run the `demo.sh` script:
    ```bash
    git clone https://github.com/mghanava/mineral-deposit-classification.git
    cd mineral-deposit-classification
    ./demo.sh
    ```
    The script will automatically pull the data, build the Docker container, and open the application in your default web browser.

3.  **Stop the Application:**
    When you're finished, shut down the container with:
    ```bash
    docker-compose down
    ```

---

## Manual Installation

Follow these steps for a manual setup.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/mghanava/mineral-deposit-classification.git
    cd mineral-deposit-classification
    ```

2.  **Pull DVC Data:**
    ```bash
    dvc pull -f
    ```

3.  **Build and run the container:**
    Use Docker Compose to build the image and start the service in detached mode:
    ```bash
    docker-compose up -d --build
    ```

### Usage

-   **Streamlit Dashboard:** Once the container is running, open your web browser and navigate to `http://localhost:8501`.
-   **Development Shell:** To access an interactive shell inside the running container, use:
    ```bash
    docker-compose exec mineral_classifier bash
    ```

### Project Structure

```
├── app.py                    # The Streamlit dashboard application
├── demo.sh                   # One-click demo script
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