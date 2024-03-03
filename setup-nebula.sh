echo "Installing NebulaGraph"
git clone -b v3.5.0 https://github.com/vesoft-inc/nebula-docker-compose.git

echo "Starting NebulaGraph"
cd nebula-docker-compose/
docker-compose up -d