#!/bin/bash

# OllamaDiffuser Release Script
# Usage: ./scripts/release.sh v1.0.0

set -e

VERSION=$1

if [ -z "$VERSION" ]; then
    echo "❌ Error: Version required"
    echo "Usage: $0 v1.0.0"
    exit 1
fi

# Remove 'v' prefix if present
VERSION_NUMBER=${VERSION#v}

echo "🚀 Starting release process for version $VERSION"

# Check if we're on main branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo "⚠️  Warning: Not on main branch (currently on $CURRENT_BRANCH)"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo "❌ Error: Uncommitted changes detected"
    echo "Please commit or stash your changes first"
    exit 1
fi

# Update version in setup.py
echo "📝 Updating version in setup.py..."
sed -i.bak "s/version=\"[^\"]*\"/version=\"$VERSION_NUMBER\"/" setup.py
rm setup.py.bak

# Update CHANGELOG.md
echo "📝 Updating CHANGELOG.md..."
TODAY=$(date +%Y-%m-%d)
sed -i.bak "s/## \[Unreleased\]/## [Unreleased]\n\n## [$VERSION_NUMBER] - $TODAY/" CHANGELOG.md
rm CHANGELOG.md.bak

# Commit version changes
echo "💾 Committing version changes..."
git add setup.py CHANGELOG.md
git commit -m "Prepare release $VERSION"

# Create and push tag
echo "🏷️  Creating tag $VERSION..."
git tag -a "$VERSION" -m "Release $VERSION"

echo "⬆️  Pushing changes and tag..."
git push origin main
git push origin "$VERSION"

# Build package
echo "📦 Building package..."
./publish_to_pypi.sh

echo "✅ Release $VERSION prepared successfully!"
echo ""
echo "Next steps:"
echo "1. Go to GitHub and create a release from tag $VERSION"
echo "2. Upload to PyPI: python -m twine upload dist/*"
echo "3. Build and push Docker image (optional)"
echo ""
echo "GitHub release URL:"
echo "https://github.com/ollamadiffuser/ollamadiffuser/releases/new?tag=$VERSION" 