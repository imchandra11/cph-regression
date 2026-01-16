# Publishing Guide for cph-regression

This guide explains how to publish the `cph-regression` package to PyPI.

## Prerequisites

1. Create accounts on:
   - [PyPI](https://pypi.org/account/register/) (for production releases)
   - [TestPyPI](https://test.pypi.org/account/register/) (for testing releases)

2. Install required tools:
```bash
pip install build twine
```

## Building the Package

1. **Clean previous builds:**
```bash
rm -rf dist/ build/ *.egg-info
```

2. **Build the package:**
```bash
python -m build
```

This will create:
- `dist/cph-regression-0.1.0.tar.gz` (source distribution)
- `dist/cph_regression-0.1.0-py3-none-any.whl` (wheel distribution)

## Testing the Package (TestPyPI)

1. **Upload to TestPyPI:**
```bash
python -m twine upload --repository testpypi dist/*
```

You'll be prompted for your TestPyPI username and password.

2. **Test installation from TestPyPI:**
```bash
pip install --index-url https://test.pypi.org/simple/ cph-regression
```

3. **Test the command:**
```bash
cph-regression --help
```

## Publishing to PyPI

Once you've tested on TestPyPI:

1. **Upload to PyPI:**
```bash
python -m twine upload dist/*
```

You'll be prompted for your PyPI username and password.

2. **Verify installation:**
```bash
pip install cph-regression
cph-regression --help
```

## Updating the Version

Before publishing a new version:

1. Update version in:
   - `pyproject.toml` (version field)
   - `setup.py` (version field)
   - `cph_regression/__init__.py` (__version__)

2. Update `CHANGELOG.md` (if you have one)

3. Commit and tag the release:
```bash
git add .
git commit -m "Release version 0.1.0"
git tag v0.1.0
git push origin main --tags
```

## Using API Tokens (Recommended)

Instead of using passwords, you can use API tokens:

1. Go to PyPI account settings
2. Create an API token
3. Use it with twine:
```bash
python -m twine upload --username __token__ --password <your-token> dist/*
```

## Troubleshooting

- **"File already exists"**: The version already exists on PyPI. Update the version number.
- **"Invalid distribution"**: Check that all required files are included in MANIFEST.in
- **"Missing dependencies"**: Verify all dependencies are listed in pyproject.toml
