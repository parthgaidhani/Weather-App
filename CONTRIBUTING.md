# Contributing to Weather Intelligence Platform

First off, thanks for taking the time to contribute! 🎉

## 🧭 How to Contribute

### Reporting Bugs
- Use the **Bug Report** issue template.
- Include steps to reproduce, expected vs actual behavior, and your environment (OS, Python version).

### Suggesting Features
- Open a **Feature Request** issue.
- Explain the *problem* you're solving, not just the solution.

### Submitting Pull Requests
1. Fork the repo & create a feature branch:
   ```bash
   git checkout -b feat/amazing-thing
   ```
2. Make your changes — follow the code style below.
3. Run `make lint && make test-imports` before pushing.
4. Open a PR using the **Pull Request Template**.

## 🧑‍💻 Code Style

- **Formatter:** `black` (line length 100)
- **Linter:** `ruff`
- **Imports:** `isort` profile = black
- **Type hints:** required for public functions
- **Docstrings:** Google style for modules, NumPy style OK for math-heavy code

## 📁 Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(models): add XGBoost hyperparameter tuning
fix(api): handle missing city gracefully
docs(readme): clarify Docker instructions
```

## ✅ Definition of Done

- [ ] Code lints (`make lint` passes)
- [ ] Module imports cleanly (`make test-imports` passes)
- [ ] Public functions have docstrings
- [ ] README / docs updated if behavior changed
- [ ] No secrets, credentials, or large data files committed

---

Questions? Open an issue or start a discussion.
