# Contributing to AtiSim

Thank you for your help. You can send a bug report, a question or a pull request through
[GitHub issues](https://github.com/MatusGib/AtiSim/issues).

## Before you start

Read the [Development Manual](https://matusgib.github.io/AtiSim/development-manual.html). It
gives the structure of the code, the rules for changes, and the test procedures. The source of
the manual is `docs/development-manual.md`.

## Send a pull request

1. Prepare the development environment (Development Manual, section 2).
2. Make your change. Obey the rules in section 5 of the Development Manual.
3. Run the tests:

   ```bash
   python -m pytest -q
   python -m pytest --nbval-lax notebooks/
   ```

4. Build the documentation:

   ```bash
   python -m sphinx -b html -W docs docs/_build/html
   ```

5. Add your change to `CHANGELOG.md`, under `[Unreleased]`.
6. Send the pull request. Say what you changed, what you measured, and what you did not change.

The continuous integration does steps 3 and 4 again for each pull request.

## Code of conduct

This project uses the [Contributor Covenant](CODE_OF_CONDUCT.md).
