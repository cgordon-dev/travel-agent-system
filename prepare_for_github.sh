#!/bin/bash

# Script to prepare the TravelGraph project for GitHub upload
# This script ensures that sensitive credentials are protected

echo "Preparing TravelGraph project for GitHub upload..."

# Ensure .gitignore is set up
if [ ! -f .gitignore ]; then
    echo "ERROR: .gitignore file is missing!"
    exit 1
fi

# Create necessary directories if they don't exist
mkdir -p data rl_data rl_models logs
mkdir -p docs/images

# Check if .env file exists and ensure it's not tracked
if [ -f .env ]; then
    echo "Checking .env file safety..."
    if grep -q "your_" .env || grep -q "change_me" .env; then
        echo "✅ .env file appears to have placeholder credentials"
    else
        echo "⚠️  WARNING: .env file may contain real credentials!"
        echo "    Please ensure all sensitive credentials are replaced with placeholders"
        echo "    or remove them before uploading to GitHub."
    fi
    
    # Check if .env.example exists
    if [ ! -f .env.example ]; then
        echo "Creating .env.example from .env with placeholders..."
        cat .env | sed 's/=.*/=your_value_here/g' > .env.example
    fi
else
    echo "No .env file found. Creating one from .env.example..."
    if [ -f .env.example ]; then
        cp .env.example .env
    else
        echo "ERROR: Neither .env nor .env.example exists!"
        exit 1
    fi
fi

# Check for credentials in Python files
echo "Scanning Python files for potential API keys or credentials..."
grep -r "api_key\|password\|secret\|token" --include="*.py" . | grep -v "your_\|example\|placeholder"
if [ $? -eq 0 ]; then
    echo "⚠️  WARNING: Possible credentials found in Python files"
    echo "    Please review the lines above and ensure they are not real credentials"
else
    echo "✅ No obvious credentials found in Python files"
fi

# Check for large files that shouldn't be in git
echo "Checking for large files..."
find . -type f -size +10M | grep -v "node_modules\|venv\|env\|.git"
if [ $? -eq 0 ]; then
    echo "⚠️  WARNING: Large files found that might not belong in git"
    echo "    Consider adding them to .gitignore or removing them"
else
    echo "✅ No large files found"
fi

# Ensure demo/placeholder data is in place
echo "Setting up placeholder data directories..."
touch data/.gitkeep
touch rl_data/.gitkeep
touch rl_models/.gitkeep
touch logs/.gitkeep

# Final checks
echo "Running final checks..."

# Check README exists
if [ ! -f README.md ]; then
    echo "ERROR: README.md is missing!"
    exit 1
else
    echo "✅ README.md exists"
fi

# Check LICENSE exists
if [ ! -f LICENSE ]; then
    echo "⚠️  WARNING: LICENSE file is missing!"
    echo "    Consider adding an open source license to your project"
else
    echo "✅ LICENSE exists"
fi

# Check requirements.txt exists
if [ ! -f requirements.txt ]; then
    echo "ERROR: requirements.txt is missing!"
    exit 1
else
    echo "✅ requirements.txt exists"
fi

echo ""
echo "======================= SUMMARY ======================="
echo "Project appears ready for GitHub upload."
echo ""
echo "Before uploading, please verify:"
echo "1. All API keys and credentials have been removed or replaced with placeholders"
echo "2. .env is in your .gitignore file"
echo "3. No sensitive or large files will be uploaded"
echo "4. Documentation accurately reflects your project"
echo ""
echo "To upload to GitHub:"
echo "1. Create a new repository on GitHub"
echo "2. Initialize git in this directory if not already done"
echo "3. Add all files and commit"
echo "4. Add the GitHub repository as remote"
echo "5. Push to GitHub"
echo ""
echo "Example:"
echo "  git init"
echo "  git add ."
echo "  git commit -m \"Initial commit\""
echo "  git remote add origin https://github.com/yourusername/travel-agent-team.git"
echo "  git push -u origin main"
echo "========================================================="

# Make the script executable
chmod +x prepare_for_github.sh

echo ""
echo "This script is now executable. Run it before uploading to GitHub."