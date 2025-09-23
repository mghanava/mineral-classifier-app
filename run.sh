export UID=$(id -u)
export GID=$(id -g)
export DOCKER_BUILDKIT=1
docker-compose up -d --build