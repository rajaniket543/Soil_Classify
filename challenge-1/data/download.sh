#!/bin/bash

COMPETITION="soil-classification"
TARGET_DIR="./data"

echo "Downloading competition data: $COMPETITION"
mkdir -p "$TARGET_DIR"
kaggle competitions download -c "$COMPETITION" -p "$TARGET_DIR" --force
echo "Unzipping files..."
unzip -o "$TARGET_DIR"/*.zip -d "$TARGET_DIR"
echo "Download and unzip complete."
