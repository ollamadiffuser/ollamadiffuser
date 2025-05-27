# 📦 Publishing Guide for OllamaDiffuser

This guide covers how to publish and distribute OllamaDiffuser through various channels.

## 🐍 PyPI (Python Package Index)

### Prerequisites
1. Create accounts on:
   - [PyPI](https://pypi.org/account/register/)
   - [TestPyPI](https://test.pypi.org/account/register/) (for testing)

2. Install publishing tools:
```bash
pip install build twine
```

### Publishing Steps

#### 1. Prepare for Release
```bash
# Update version in setup.py
# Update CHANGELOG.md
# Commit changes
git add .
git commit -m "Prepare for release v1.0.0"
```

#### 2. Build Package
```bash
# Use the provided script
./publish_to_pypi.sh

# Or manually:
rm -rf build/ dist/ *.egg-info/
python -m build
python -m twine check dist/*
```

#### 3. Test Upload (Optional)
```bash
python -m twine upload --repository testpypi dist/*
# Test install: pip install -i https://test.pypi.org/simple/ ollamadiffuser
```

#### 4. Upload to PyPI
```bash
python -m twine upload dist/*
```

### API Token Setup
1. Go to PyPI Account Settings → API tokens
2. Create token with scope for your project
3. Use token as password with username `__token__`

## 🐙 GitHub Release

### Manual Release
1. Go to your GitHub repository
2. Click "Releases" → "Create a new release"
3. Tag version: `v1.0.0`
4. Release title: `Release v1.0.0`
5. Describe changes and attach files

### Automated Release
The project includes GitHub Actions workflow (`.github/workflows/release.yml`):

1. Set up PyPI API token in GitHub Secrets:
   - Go to repository Settings → Secrets → Actions
   - Add `PYPI_API_TOKEN` with your PyPI token

2. Create and push a tag:
```bash
git tag v1.0.0
git push origin v1.0.0
```

This will automatically:
- Create GitHub release
- Build and publish to PyPI

## 🐳 Docker Publishing

### Build and Test Locally
```bash
# Build image
docker build -t ollamadiffuser:latest .

# Test run
docker run -p 8001:8001 ollamadiffuser:latest

# Or use docker-compose
docker-compose up
```

### Publish to Docker Hub
```bash
# Tag for Docker Hub
docker tag ollamadiffuser:latest yourusername/ollamadiffuser:latest
docker tag ollamadiffuser:latest yourusername/ollamadiffuser:v1.0.0

# Push to Docker Hub
docker push yourusername/ollamadiffuser:latest
docker push yourusername/ollamadiffuser:v1.0.0
```

### GitHub Container Registry
```bash
# Login to GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# Tag and push
docker tag ollamadiffuser:latest ghcr.io/ollamadiffuser/ollamadiffuser:latest
docker push ghcr.io/ollamadiffuser/ollamadiffuser:latest
```

## 📚 Documentation Publishing

### GitHub Pages
1. Enable GitHub Pages in repository settings
2. Use `docs/` folder or `gh-pages` branch
3. Documentation will be available at: `https://ollamadiffuser.github.io/ollamadiffuser/`

### Read the Docs
1. Connect your GitHub repository to [Read the Docs](https://readthedocs.org/)
2. Configure build settings
3. Documentation builds automatically on commits

## 🔄 Version Management

### Semantic Versioning
Follow [SemVer](https://semver.org/):
- `MAJOR.MINOR.PATCH`
- `1.0.0` → `1.0.1` (patch: bug fixes)
- `1.0.0` → `1.1.0` (minor: new features)
- `1.0.0` → `2.0.0` (major: breaking changes)

### Release Checklist
- [ ] Update version in `setup.py`
- [ ] Update `CHANGELOG.md`
- [ ] Run tests
- [ ] Build and test package locally
- [ ] Create git tag
- [ ] Push to GitHub
- [ ] Verify automated release
- [ ] Test installation from PyPI
- [ ] Update documentation

## 🚀 Quick Publish Commands

```bash
# Complete release process
./scripts/release.sh v1.0.0

# Or step by step:
git tag v1.0.0
git push origin v1.0.0
./publish_to_pypi.sh
```

## 📋 Distribution Channels Summary

| Channel | Command | Users Install With |
|---------|---------|-------------------|
| PyPI | `twine upload dist/*` | `pip install ollamadiffuser` |
| GitHub | Create release | Download from releases |
| Docker Hub | `docker push` | `docker run yourusername/ollamadiffuser` |
| Conda | Submit to conda-forge | `conda install ollamadiffuser` |

## 🔐 Security Notes

- Never commit API tokens to git
- Use GitHub Secrets for CI/CD
- Enable 2FA on all publishing accounts
- Regularly rotate API tokens
- Sign releases with GPG (optional)

## 📞 Support

For publishing issues:
- Check [PyPI Help](https://pypi.org/help/)
- Review [GitHub Actions docs](https://docs.github.com/en/actions)
- See [Docker Hub docs](https://docs.docker.com/docker-hub/) 