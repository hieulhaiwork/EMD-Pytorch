# Contributing to EMD PyTorch

Thank you for your interest in contributing to this project! We welcome contributions from the community.

## Development Setup

1. Fork the repository
2. Clone your fork: `git clone https://github.com/hieulhaiwork/EMD-Pytorch.git`
3. Create a development environment:
   ```bash
   conda create -n emd-dev python=3.9
   conda activate emd-dev
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```
4. Install in development mode: `pip install -e .`

## Making Changes

1. Create a new branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Test your changes: `python tests/final_validation.py`
4. Commit your changes: `git commit -am 'Add some feature'`
5. Push to the branch: `git push origin feature/your-feature-name`
6. Submit a pull request

## Code Style

- Follow PEP 8 for Python code
- Use meaningful variable names
- Add docstrings to functions and classes
- Keep CUDA code well-commented

## Testing

- All changes should include appropriate tests
- Run the full test suite before submitting
- Performance regressions should be avoided

## Pull Request Process

1. Ensure your code passes all tests
2. Update documentation if needed
3. Add your changes to the changelog
4. Request review from maintainers

## Issues

- Use the issue tracker for bug reports and feature requests
- Provide minimal reproducible examples for bugs
- Check existing issues before creating new ones