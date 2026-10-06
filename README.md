# Objective

This project aims at understanding LLM Fundamentals for API integration and prompting.

# Installation

## On MacOS

```bash
brew install pyenv
brew install uv

# go to folder
pyenv install -v 3.14.7
pyenv local 3.14.7          # writes .python-version
uv init --python "$(pyenv which python)"

# adding packages
uv add anthropic
uv add loguru
uv add dotenv
uv add --dev ruff
```