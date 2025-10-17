#!/bin/bash

# Deploy to GitHub Script
# Run this AFTER creating your GitHub repository

echo "════════════════════════════════════════════════════════════════════"
echo "  🚀 DEPLOY TO GITHUB"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# Prompt for GitHub repository URL
echo "📝 Please enter your GitHub repository URL:"
echo "   (Example: https://github.com/YOUR_USERNAME/Streamlit_Duplicate_Detector.git)"
echo ""
read -p "Repository URL: " REPO_URL

if [ -z "$REPO_URL" ]; then
    echo "❌ Error: Repository URL cannot be empty"
    exit 1
fi

echo ""
echo "📂 Adding remote repository..."
git remote add origin "$REPO_URL" 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Remote added successfully"
else
    echo "⚠️  Remote might already exist, checking..."
    git remote set-url origin "$REPO_URL"
    echo "✅ Remote URL updated"
fi

echo ""
echo "📤 Pushing to GitHub..."
git branch -M main
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo "  ✅ SUCCESS! Your code is now on GitHub!"
    echo "════════════════════════════════════════════════════════════════════"
    echo ""
    echo "🌐 View your repository at:"
    echo "   ${REPO_URL%.git}"
    echo ""
    echo "📋 NEXT STEP: Deploy to Streamlit Cloud"
    echo "   1. Go to: https://share.streamlit.io/"
    echo "   2. Click 'New app'"
    echo "   3. Select your repository: ${REPO_URL%.git}"
    echo "   4. Main file: streamlit_app.py"
    echo "   5. Click 'Deploy!'"
    echo ""
    echo "════════════════════════════════════════════════════════════════════"
else
    echo ""
    echo "❌ Error: Failed to push to GitHub"
    echo "   Please check your GitHub URL and try again"
    echo "   Run: ./deploy_to_github.sh"
fi

