#!/bin/bash

# Make sure we aren't running...
echo "Bringing Docker down..."
docker compose down

# Build image
echo "Docker building..."
docker build -t bigfishradio .

# Up time
echo "PRODUCTION ENVIRONMENT ENABLED!"
docker compose -f docker-compose.yml up -d

