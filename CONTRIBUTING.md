# Contributing to llm-gen-ai

Thank you for your interest in contributing to llm-gen-ai! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/llm-gen-ai.git
   cd llm-gen-ai
   ```
3. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Install Dependencies

```bash
# Using pip
pip install -e ".[dev]"

# Using uv (recommended)
uv pip install -e ".[dev]"
```

### Code Style

We use:
- **Black** for code formatting (100 character line length)
- **isort** for import sorting
- **flake8** for linting

Format your code before committing:

```bash
black src/ examples/ tests/
isort src/ examples/ tests/
flake8 src/ examples/ tests/
```

## Making Changes

### Code Guidelines

1. **Follow Python best practices**
   - Use type hints where appropriate
   - Write descriptive docstrings
   - Keep functions focused and modular

2. **Maintain consistency**
   - Follow the existing code structure
   - Use the same naming conventions
   - Match the project's design patterns

3. **Add tests**
   - Write unit tests for new functionality
   - Ensure existing tests pass
   - Aim for good test coverage

4. **Update documentation**
   - Update README.md if adding features
   - Add docstrings to new functions/classes
   - Update relevant docs/ files

### Project Structure

- `src/llm_gen_ai/` - Core library code
  - `agents/` - Agent implementations
  - `core/` - Core functionality
  - `modules/` - Modular components
- `examples/` - Example scripts
- `experiments/` - Experimental code
- `tests/` - Test suite
- `docs/` - Documentation files

## Testing

Run tests before submitting:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=llm_gen_ai --cov-report=html

# Run specific test file
pytest tests/test_agents.py
```

## Submitting Changes

1. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: Add new feature"
   ```

2. **Use conventional commits** for commit messages:
   - `feat:` - New feature
   - `fix:` - Bug fix
   - `docs:` - Documentation changes
   - `test:` - Test additions/changes
   - `refactor:` - Code refactoring
   - `chore:` - Maintenance tasks

3. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Create a Pull Request** on GitHub
   - Provide a clear description
   - Reference any related issues
   - Ensure CI passes

## Pull Request Guidelines

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] Tests added/updated and passing
- [ ] Documentation updated
- [ ] Commit messages follow conventional commits
- [ ] No merge conflicts
- [ ] PR description clearly explains changes

### PR Description Template

```markdown
## Description
Brief description of what this PR does

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
How has this been tested?

## Related Issues
Fixes #(issue number)
```

## Reporting Issues

### Bug Reports

Include:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS
- Model being used
- Error messages/stack traces

### Feature Requests

Include:
- Clear description of the feature
- Use case and motivation
- Proposed implementation (if any)
- Examples of usage

## Code Review Process

1. Maintainers will review your PR
2. Address any requested changes
3. Once approved, your PR will be merged
4. Your contribution will be acknowledged

## Development Tips

### Running in Development Mode

```bash
# Install in editable mode
pip install -e .

# Your changes will be immediately available
python examples/enhanced_qa_example.py
```

### Debugging

```python
# Enable timing logs in config.py
ENABLE_TIMING_LOGS = True

# Enable verbose output in examples
agent.answer_query(query, show_analysis=True)
```

### Adding New Modules

1. Create module in `src/llm_gen_ai/modules/`
2. Add to `__init__.py` for public API
3. Import in relevant agent files
4. Add tests in `tests/test_modules.py`
5. Document in appropriate docs/ file

## Questions?

- Open an issue for questions
- Use discussions for general topics
- Check existing issues/PRs first

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.

## Thank You!

Your contributions help make llm-gen-ai better for everyone. We appreciate your time and effort!
