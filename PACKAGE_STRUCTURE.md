# Package Structure and Migration Guide

## New Package Structure

After packaging, your project has the following structure:

```
cph-regression/
├── cph_regression/              # Main package (installable via pip)
│   ├── __init__.py
│   ├── __main__.py              # Allows: python -m cph_regression
│   ├── cli.py                   # CLI entry point (cph-regression command)
│   └── regression/              # Regression module
│       ├── __init__.py
│       ├── cli.py
│       ├── main.py
│       ├── mainfittest.py
│       ├── dataset.py
│       ├── datamodule.py
│       ├── modelfactory.py
│       ├── modelmodule.py
│       └── callbacks.py
├── Regression/                  # Original module (kept for backward compatibility)
├── GemstonePricePrediction/     # Example project
├── pyproject.toml               # Modern Python packaging config
├── setup.py                     # Fallback setup script
├── LICENSE                      # MIT License
├── README.md                    # Package README for PyPI
├── MANIFEST.in                  # Files to include in package
├── requirements.txt             # Dependencies
└── .gitignore                   # Updated for packaging
```

## Import Changes

### Before (Local Development)
```python
from Regression.modelmodule import ModelModuleRGS
from Regression.datamodule import DataModuleRGS
```

### After (Installed Package)
```python
from cph_regression.regression.modelmodule import ModelModuleRGS
from cph_regression.regression.datamodule import DataModuleRGS
```

Or use the top-level imports:
```python
from cph_regression import ModelModuleRGS, DataModuleRGS
```

## Config File Changes

### Before
```yaml
model:
  class_path: Regression.modelmodule.ModelModuleRGS
data:
  class_path: Regression.datamodule.DataModuleRGS
callbacks:
  - class_path: Regression.callbacks.ONNXExportCallback
```

### After
```yaml
model:
  class_path: cph_regression.regression.modelmodule.ModelModuleRGS
data:
  class_path: cph_regression.regression.datamodule.DataModuleRGS
callbacks:
  - class_path: cph_regression.regression.callbacks.ONNXExportCallback
```

## Usage After Installation

### Installation
```bash
pip install cph-regression
```

### Command Line Usage
```bash
# Default: fit + test
cph-regression --config config.yaml

# Training only
cph-regression fit --config config.yaml

# Testing only
cph-regression test --config config.yaml
```

### Python Usage
```python
from cph_regression import ModelModuleRGS, DataModuleRGS
from cph_regression.regression.cli import RGSLightningCLI

# Use in your code
cli = RGSLightningCLI()
```

## Local Development

For local development, you can install the package in editable mode:

```bash
pip install -e .
```

This allows you to make changes to the code without reinstalling.

## Backward Compatibility

The original `Regression/` folder is kept for backward compatibility. However, for new projects, use the installed package imports.

## Next Steps

1. **Update your config files** to use the new import paths
2. **Test locally** by installing in editable mode: `pip install -e .`
3. **Test the CLI**: `cph-regression --config GemstonePricePrediction/configs/gemstone.yaml`
4. **Publish to PyPI** following the guide in `PUBLISHING.md`
