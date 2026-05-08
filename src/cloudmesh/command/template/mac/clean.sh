#!/bin/bash

# Configuration - Match these to your launch script
CONTAINER_NAME="open-webui"
VOLUME_NAME="open-webui"

echo "------------------------------------------------"
echo "WARNING: This will PERMANENTLY DELETE all:"
echo "1. User accounts and passwords"
echo "2. Chat histories"
echo "3. Uploaded documents and RAG data"
echo "------------------------------------------------"
read -p "Are you sure you want to wipe everything? (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Stopping and removing container: $CONTAINER_NAME..."
    docker stop $CONTAINER_NAME 2>/dev/null
    docker rm $CONTAINER_NAME 2>/dev/null

    echo "Deleting persistent volume: $VOLUME_NAME..."
    docker volume rm $VOLUME_NAME 2>/dev/null

    echo "Success! The next time you run your launch script, it will be a fresh install."
else
    echo "Reset cancelled."
fi
