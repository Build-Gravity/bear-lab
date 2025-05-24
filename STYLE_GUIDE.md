# Style Guide and Coding Conventions

This document outlines the coding style and conventions to be followed in this project.

## General Python Style

*   **PEP 8:** Code should adhere to [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/).
*   **Strict Typing:** All function signatures and variable declarations should use type hints as per [PEP 484 -- Type Hints](https://www.python.org/dev/peps/pep-0484/) and [PEP 526 -- Syntax for Variable Annotations](https://www.python.org/dev/peps/pep-0526/).
*   **Docstrings:** All modules, functions, classes, and methods should have clear and concise docstrings as per [PEP 257 -- Docstring Conventions](https://www.python.org/dev/peps/pep-0257/). Google Python Style Docstrings are preferred.
*   **Imports:** Imports should be organized with standard library imports first, then third-party imports, then local application/library specific imports. Each group should be separated by a blank line.
*   **Readability:** Prioritize clear and readable code. Comments should explain *why* something is done, not *what* is being done if the code itself is clear.

## Linters and Formatters (To Be Decided/Used)

*   Consider using tools like Black for automated code formatting and Flake8/Pylint for linting, and MyPy for static type checking.
*   (We will confirm and update this section as we establish our tooling in Cursor.) 