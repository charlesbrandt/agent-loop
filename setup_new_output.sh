#!/bin/bash

# Configuration
BARN_DIR="barn"
OUTPUT_DIR="output"
SUBDIRS=("planning" "stories" "src" "tests" "logs")

# Function to get the project name from project_config.yaml
get_project_name() {
    grep "project_name:" project_config.yaml | awk -F': ' '{print $2}' | sed 's/"//g' | awk '{print $1}' | tr '[:upper:]' '[:lower:]'
}

PROJECT_NAME_WORD=$(get_project_name)
if [ -z "$PROJECT_NAME_WORD" ]; then
    PROJECT_NAME_WORD="default"
fi

# 1. Archive the old output directory
if [ -d "$OUTPUT_DIR" ]; then
    echo "Archiving existing '$OUTPUT_DIR' directory..."
    mkdir -p "$BARN_DIR"

    DATE=$(date +"%Y-%m-%d")
    BASE_ARCHIVE_NAME="${OUTPUT_DIR}-${DATE}-${PROJECT_NAME_WORD}"
    ARCHIVE_DIR_NAME="${BASE_ARCHIVE_NAME}"
    COUNTER=1
    
    # Check if a directory with the same date and project name word already exists
    while [ -d "$BARN_DIR/$ARCHIVE_DIR_NAME" ]; do
      ARCHIVE_DIR_NAME="${BASE_ARCHIVE_NAME}_$(date +"%H-%M-%S")"
      sleep 1 # Ensure unique timestamp if operations are very fast
    done
    
    mv "$OUTPUT_DIR" "$BARN_DIR/$ARCHIVE_DIR_NAME"
    echo "Archived to '$BARN_DIR/$ARCHIVE_DIR_NAME'"

    # Copy project_config.yaml and templates/prompts.yaml to the archived directory for reference
    echo "Copying configuration files to archived directory..."
    cp project_config.yaml "$BARN_DIR/$ARCHIVE_DIR_NAME/"
    cp templates/prompts.yaml "$BARN_DIR/$ARCHIVE_DIR_NAME/"
    echo "Configuration files copied."
fi

# 2. Create the new output directory and subdirectories
echo "Creating new '$OUTPUT_DIR' directory..."
mkdir -p "$OUTPUT_DIR"
for subdir in "${SUBDIRS[@]}"; do
    mkdir -p "$OUTPUT_DIR/$subdir"
done
echo "Created subdirectories: ${SUBDIRS[*]}"

# 3. Initialize Git repository
echo "Initializing Git repository in '$OUTPUT_DIR'..."
git -C "$OUTPUT_DIR" init -b main
echo "Git repository initialized with default branch 'main'."

# 4. Create a .gitignore file
echo "Creating .gitignore..."
cat <<EOL > "$OUTPUT_DIR/.gitignore"
# Logs
logs/
*.log

# IDE files
.vscode/
.idea/

# Python cache
__pycache__/
*.pyc
EOL
echo "'.gitignore' created."

# 5. Create a placeholder README.md
echo "Creating placeholder README.md..."
cat <<EOL > "$OUTPUT_DIR/README.md"
# TODO
EOL
echo "Placeholder README.md created."

echo "Setup complete. New output directory is ready at '$OUTPUT_DIR'."
