echo "Fixing .dvc permissions..."
sudo chmod -R 777 .dvc/

echo "Starting Docker with BuildKit..."
export DOCKER_BUILDKIT=1
docker-compose up -d --build

echo "Done!"