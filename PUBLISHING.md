# Publishing to PyPI

This project uses GitHub Actions with PyPI Trusted Publishing (OIDC) for secure, token-free releases.

## One-Time Setup on PyPI

1. Go to https://pypi.org/manage/account/publishing/
2. Add a new pending publisher with these values:

| Field | Value |
|-------|-------|
| **PyPI Project Name** | `pgsql-mcp` |
| **Owner** | `surajmandalcell` |
| **Repository name** | `pgsql-mcp` |
| **Workflow name** | `pypi-publish.yml` |
| **Environment name** | `pypi` |

3. In your GitHub repository, create the environment:
   - Go to Settings → Environments → New environment
   - Name: `pypi`
   - Optionally add protection rules (required reviewers, etc.)

## Publishing a Release

### Option 1: GitHub Release (Recommended)

```bash
# 1. Update version in pyproject.toml
# 2. Commit and push
git add pyproject.toml
git commit -m "Bump version to X.Y.Z"
git push

# 3. Create and push a tag
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push --tags

# 4. Create GitHub release (triggers the workflow)
gh release create vX.Y.Z --title "pgsql-mcp vX.Y.Z" --notes "Release notes here"
```

### Option 2: Manual Workflow Dispatch

1. Go to Actions → "Publish to PyPI" → Run workflow
2. Click "Run workflow"

### Option 3: Local Build and Upload (Manual)

```bash
# Build
uv build

# Upload (requires PyPI API token)
uv publish
# or
twine upload dist/*
```
