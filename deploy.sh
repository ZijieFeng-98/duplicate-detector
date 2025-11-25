#!/bin/bash
# Quick deployment script for Streamlit Cloud

echo "🚀 Streamlit Cloud Deployment Helper"
echo "====================================="
echo ""

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "❌ Error: Not a git repository"
    echo "   Run: git init"
    exit 1
fi

# Check if files are committed
if [ -n "$(git status --porcelain)" ]; then
    echo "⚠️  Warning: You have uncommitted changes"
    echo ""
    echo "Files to commit:"
    git status --short
    echo ""
    read -p "Commit these changes? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git add .
        read -p "Commit message: " commit_msg
        git commit -m "${commit_msg:-Ready for deployment}"
    else
        echo "❌ Please commit changes first"
        exit 1
    fi
fi

# Check required files
echo "✅ Checking required files..."
missing=0

files=(
    "streamlit_app.py"
    "requirements.txt"
    "ai_pdf_panel_duplicate_check_AUTO.py"
    ".streamlit/config.toml"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file (MISSING)"
        missing=1
    fi
done

if [ $missing -eq 1 ]; then
    echo ""
    echo "❌ Some required files are missing"
    exit 1
fi

# Check if remote is set
if ! git remote | grep -q "origin"; then
    echo ""
    echo "⚠️  No 'origin' remote found"
    read -p "Add GitHub remote? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -p "GitHub repository URL: " repo_url
        git remote add origin "$repo_url"
    fi
fi

# Show current status
echo ""
echo "📊 Current Status:"
echo "  Branch: $(git branch --show-current)"
echo "  Remote: $(git remote get-url origin 2>/dev/null || echo 'Not set')"
echo "  Last commit: $(git log -1 --oneline)"

# Push to GitHub
echo ""
read -p "Push to GitHub? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📤 Pushing to GitHub..."
    git push origin main || git push origin master
    echo ""
    echo "✅ Pushed successfully!"
    echo ""
    echo "🎯 Next Steps:"
    echo "  1. Go to https://share.streamlit.io/"
    echo "  2. Click 'New app'"
    echo "  3. Select your repository"
    echo "  4. Main file: streamlit_app.py"
    echo "  5. Click 'Deploy'"
    echo ""
    echo "Your app will be live in ~3-5 minutes! 🚀"
else
    echo "Skipped push. Run 'git push origin main' when ready."
fi

