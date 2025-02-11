#!/bin/bash

# Define the source directory
SOURCE_DIR="root"

# Get the parent directory of the source
PARENT_DIR=$(dirname "$(realpath "$SOURCE_DIR")")

# Loop to create 8 copies
for i in {1..8}; do
    DEST_DIR="${PARENT_DIR}/root_${i}"
    rsync -a "$SOURCE_DIR/" "$DEST_DIR/"
    echo "Copied $SOURCE_DIR to $DEST_DIR using rsync"
done

echo "All copies completed successfully!"