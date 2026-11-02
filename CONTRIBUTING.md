# Contributing to Synapse Agent

Thank you for your interest in contributing to `synapse-agent`! We welcome contributions from the community.

## Development Setup

1. Fork the repository and clone your fork locally:
   ```bash
   git clone https://github.com/<your-username>/synapse-agent.git
   cd synapse-agent
   ```

2. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
   ```

3. Install editable package with development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Running Tests

Before submitting a pull request, ensure all tests pass:
```bash
pytest tests/
```

To run tests with coverage reporting:
```bash
pytest --cov=synapse tests/
```

## Guidelines

- **Zero Hardcoded Dependencies:** Core memory operations must remain functional using Python standard library and SQLite WAL mode.
- **Zero Emojis:** Maintain a clean, professional codebase and documentation style without emojis.
- **English Codebase:** All code comments, docstrings, variable names, and commit messages must be in English.
- **Atomic Commits:** Provide clear, imperative commit messages (e.g. `feat: add graph cycle detection`, `fix: handle missing db directory`).

## Submitting a Pull Request

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Commit your changes and push to your fork.
3. Open a Pull Request against the `main` branch with a clear description of changes.
