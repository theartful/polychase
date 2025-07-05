#/bin/bash

SCRIPT="$(readlink -f "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT")"

cd "${SCRIPT_DIR}"
docker buildx build -f ./docker/Dockerfile.linux . -t polychase

docker run --name polychase -d -i -t polychase /bin/sh
docker cp polychase:/work/polychase-src/wheelhouse/. ./blender_addon/wheels

docker container stop polychase
docker container rm polychase
