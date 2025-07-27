FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel
# set the working directory in the container
WORKDIR /app
# install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# copy python requirements file first to leverage docker cache
COPY requirements.txt .
# install python dependencies and DVC
RUN pip install --no-cache-dir -r requirements.txt
# copy the rest of the application code
COPY . .
# expose the port the app runs on
EXPOSE 8000
# set python path to include the src directory
ENV PYTHONPATH=/app:$PYTHONPATH
# Creates a non-root user and adds permission to access the /app folder
RUN useradd -u 1000 appuser && chown -R appuser /app
RUN mkdir -p /home/appuser && chown appuser /home/appuser
USER appuser
# command to run when the container starts
CMD ["dvc", "exp", "save"]




