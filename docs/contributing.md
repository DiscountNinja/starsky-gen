# Contributing to Starsky-gen

Thank you for your interest in contributing to Starsky-gen!

Starsky-gen is a procedural starfield and nebula generator intended for artists, game developers, and other creative projects. Contributions should help keep generation reproducible, rendering behavior predictable, and the tool useful to both new and experienced users.

## Before You Start

- Read the project README for installation instructions, usage, and supported features.
- For substantial changes, open an issue or discussion first so we can agree on the intended behavior.
- Keep contributions focused. Unrelated refactoring makes rendering changes harder to review and validate.

## Development Standards

### Reproducibility

- Use the project's existing seeded random-number mechanisms.
- Do not introduce unseeded randomness into generation paths.
- Given the same seed, configuration, and supported environment, generation should be reproducible.
- If a change intentionally affects reproducibility, document the reason.

### Rendering and Visual Quality

- Respect the responsibilities of existing rendering stages and layers.
- Avoid changing unrelated rendering behavior when implementing a specific feature or fix.
- Validate rendering changes using a fixed seed and known configuration.
- Where practical, compare before-and-after renders and inspect intermediate layers to isolate regressions.
- Document intentional changes to the appearance of existing features or look targets.

### Resolution and Output

- Preserve intended visual behavior across supported output resolutions.
- Prefer density- or scale-based calculations when a feature is intended to scale with the output.
- Avoid introducing resolution-specific constants without a clear reason.
- Preserve the correctness of supported output formats and their projection requirements.

### CLI and Configuration

- Keep command-line options clearly named and documented.
- Provide sensible defaults for new options.
- Preserve compatibility with existing commands where practical; document breaking changes when necessary.
- Ensure guided-mode behavior stays consistent with direct CLI usage.
- Consider automation and batch generation when changing command-line behavior.

### Code Quality

- Follow the existing conventions and architecture of the code being modified.
- Prefer clear, focused changes over unnecessary abstractions or broad rewrites.
- Handle invalid input and expected failure cases explicitly.
- Avoid adding dependencies unless they provide meaningful value.
- Keep comments focused on intent, non-obvious behavior, and important constraints.

### Documentation

Update relevant documentation when adding options, changing defaults, altering output behavior, or introducing dependencies.

Examples and commands should reflect actual supported behavior.

## Testing Changes

Before submitting a contribution:

1. Run the relevant tests, if available.
2. Verify that the application starts and the affected functionality works.
3. For rendering changes, generate a comparison using a fixed seed and configuration.
4. Check for unintended changes to existing look targets or output formats.
5. Update documentation and examples as needed.

If a test cannot be run, explain why in the pull request.

## Pull Requests

A useful pull request describes:

- What changed.
- Why the change is needed.
- How the change was tested.
- Any intentional changes to generated output.
- Any compatibility considerations or known limitations.

For visual changes, screenshots or before-and-after comparisons are encouraged.

Please keep pull requests focused and open to discussion. The goal is to improve Starsky-gen while preserving its reliability and visual quality.

Thank you for helping make Starsky-gen better!
