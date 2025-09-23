#!/bin/bash
echo "Starting Docker with BuildKit..."
export DOCKER_BUILDKIT=1
docker-compose up -d --build
echo "Fixing .dvc permissions..."
sudo chown -R 1000:1000 .dvc/
docker-compose restart
echo "Done!"