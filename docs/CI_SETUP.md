# CI/CD Setup Instructions

The GitHub Actions workflow file is stored at `ci/test.yml` in this repository.
To activate CI, you need to copy it to `.github/workflows/test.yml`.

## Why Manual Setup Is Required

GitHub requires a PAT with the **`workflow` scope** to create or modify files
under `.github/workflows/`. The repository was pushed with a PAT that has
`repo` scope only.

## One-Time Setup (2 minutes)

### Option A: GitHub Web UI (Easiest)

1. Go to https://github.com/theblackhat55/spx_algo
2. Click **"Add file" → "Create new file"**
3. In the filename box type: `.github/workflows/test.yml`
4. Open https://github.com/theblackhat55/spx_algo/blob/main/ci/test.yml
5. Click the **Raw** button, copy all content
6. Paste into the new file editor
7. Click **"Commit new file"** → commit directly to `main`

### Option B: New PAT with `workflow` Scope

1. Go to https://github.com/settings/tokens/new
2. Check scopes: **`repo`** ✅  **`workflow`** ✅
3. Click **"Generate token"** and copy it
4. Run:
   ```bash
   git clone https://github.com/theblackhat55/spx_algo.git
   cd spx_algo
   mkdir -p .github/workflows
   cp ci/test.yml .github/workflows/test.yml
   git add .github/workflows/test.yml
   git commit -m "ci: activate GitHub Actions workflow"
   git push  # use the new token with workflow scope
   ```

### Option C: GitHub CLI

```bash
gh auth login  # use your GitHub credentials
gh repo clone theblackhat55/spx_algo
cd spx_algo
mkdir -p .github/workflows
cp ci/test.yml .github/workflows/test.yml
git add .github/workflows/test.yml
git commit -m "ci: activate GitHub Actions workflow"
git push
```

## What the Workflow Does

The CI pipeline (`.github/workflows/test.yml`) runs on every push to `main`/`develop`:

| Step | Description |
|------|-------------|
| Python 3.11 setup | With pip cache for fast installs |
| System deps | `gcc`, `g++`, `libgomp1`, `libopenblas-dev` |
| Leakage gate | `pytest tests/test_no_leakage.py` — must pass first |
| Full test suite | 380 tests with `--cov-fail-under=80` |
| Coverage upload | `coverage.xml` artifact saved |

## Expected CI Badge

After activating, add to your README:

```markdown
![CI](https://github.com/theblackhat55/spx_algo/actions/workflows/test.yml/badge.svg)
```
