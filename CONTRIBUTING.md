# Contributing to Magic Terminal

Thanks for your interest in improving Magic Terminal! This guide explains how to set up your environment, follow our coding standards, and submit high-quality contributions.

## 1. Getting Started

- **Clone the repository**
  ```bash
  git clone https://github.com/Yogesh-developer/magic-terminal.git
  cd magic-terminal
  ```
- **Create a virtual environment (recommended)**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate  # Windows: .venv\Scripts\activate
  ```
- **Install dependencies**
  ```bash
  pip install -r requirements.txt
  pip install -e .[dev]
  ```

## 2. Development Workflow

- **Branching**: Create feature branches from `main` (or `master`) using a descriptive name, e.g. `feature/improve-logging`.
- **Incremental commits**: Keep commits focused and write clear messages describing the change.
- **Code style**: Follow the existing conventions in `ai_terminal/`. Use `black` to format code when needed.

## 3. Testing & Quality Gates

Before opening a pull request, run:

```bash
black ai_terminal tests
pytest
```
The GitHub Actions workflow (`.github/workflows/ci.yml`) runs the automated test suite on every push and pull request.

## 4. Feature Guidelines

- **Safety first**: The CLI audits high-risk commands in `ai_terminal/safety.py`. Ensure new features respect the safety model and add tests when you introduce new heuristics.
- **Configuration**: Persisted settings live in `~/.magic_terminal_config.json`, managed through `ConfigManager`. Update `DEFAULT_CONFIG` and `CONFIG_SCHEMA` when adding new fields.
- **Documentation**: Update `README.md` and any relevant docs to explain new features or changes in workflow. Include usage examples when possible.

## 5. Submitting a Pull Request

1. Ensure your branch is up to date with `main`.
2. Push your branch and open a PR with a clear description of the change and testing performed.
3. Address feedback promptly; reviewers may request updates to tests, docs, or implementation details.

Thanks again for contributing! Your help makes Magic Terminal better for everyone.
