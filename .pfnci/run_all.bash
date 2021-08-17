#!/bin/bash

# Set up some useful variables.
docker_url="asia.gcr.io/pfn-public-ci/pynif3d"
container_name="flexci"

# Grant access to the Docker registry.
service docker stop
mount -t tmpfs tmpfs /var/lib/docker -o size=100%
service docker start
gcloud auth configure-docker

# Create container and copy the current source code.
docker pull "$docker_url"
docker create --name="$container_name" -it "$docker_url"
docker start "$container_name"
docker cp . "$container_name":/tmp/"$container_name"

# Run the tests and capture the exit code.
docker exec -e PYTHONPATH=/tmp/"$container_name"/.pfnci \
  -e CI_JOB_URL="$CI_JOB_URL" \
  -e CI_JOB_ID="$CI_JOB_ID" \
  -e CI_PROJECT_NAME="$CI_PROJECT_NAME" \
  -w /tmp/"$container_name" \
  "$container_name" \
  bash -c "bash .pfnci/checks.bash"
exit_code=$?

echo "docker exit_code is $exit_code"

# Clean-up.
docker stop "$container_name"
docker rm "$container_name"

# Return the exit code, so that the job is marked as "Failed" on the CI's page, in case of failure.
exit "$exit_code"