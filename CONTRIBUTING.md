# Contributing to RealEstateOptionsGym

Thank you for your interest in contributing to RealEstateOptionsGym! This
document provides guidelines and instructions for contributing.

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/BrandPennington/realestate-options-gym.git
   cd realestate-options-gym
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Code Style

We use the following tools for code quality:

- **black** for code formatting
- **ruff** for linting
- **mypy** for type checking

Run all checks:
```bash
black src tests
ruff check src tests
mypy src
```

## Running Tests

```bash
pytest
```

With coverage:
```bash
pytest --cov=realestate_options_gym --cov-report=html
```

## Pull Request Process

1. Fork the repository and create a feature branch
2. Make your changes with appropriate tests
3. Ensure all tests pass and code style checks pass
4. Update documentation if needed
5. Submit a pull request with a clear description

## Priority Contribution Areas

We especially welcome contributions in these areas:

- [ ] Commercial real estate environments (office, retail, industrial)
- [ ] Multi-agent market simulation
- [ ] Additional interest rate models (LMM, SABR)
- [ ] Integration with PropTech data APIs
- [ ] GPU-accelerated simulation via JAX
- [ ] Additional RL algorithm benchmarks

## Adding a New Environment

1. Create a new file in `src/realestate_options_gym/envs/`
2. Inherit from `BaseRealEstateEnv`
3. Implement required abstract methods:
   - `_create_observation_space()`
   - `_create_action_space()`
   - `_get_observation()`
   - `_calculate_reward()`
   - `_check_terminated()`
4. Register the environment in `__init__.py`
5. Add tests in `tests/test_envs.py`
6. Add documentation and examples

## Adding a New Stochastic Model

1. Create or extend files in `src/realestate_options_gym/models/`
2. Follow the interface patterns of existing models
3. Include mathematical documentation in docstrings
4. Add comprehensive tests
5. Consider numerical stability and edge cases

## Reporting Issues

When reporting issues, please include:

- Python version
- Package versions (`pip freeze`)
- Minimal reproducible example
- Expected vs actual behavior
- Full error traceback if applicable

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow

## License

By contributing, you agree that your contributions will be licensed under the
MIT License.
