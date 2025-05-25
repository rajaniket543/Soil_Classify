#!/bin/bash

# Correct Kaggle competition name
COMPETITION="soil-classification-part-2"
TARGET_DIR="./data"

echo "📥 Downloading competition data: $COMPETITION"
mkdir -p "$TARGET_DIR"

# Download using the Kaggle CLI
kaggle competitions download -c "$COMPETITION" -p "$TARGET_DIR" --force

# Unzip the downloaded ZIP files
echo "📦 Unzipping files..."
unzip -o "$TARGET_DIR"/*.zip -d "$TARGET_DIR"

echo "✅ Download and unzip complete. Data saved to $TARGET_DIR"
