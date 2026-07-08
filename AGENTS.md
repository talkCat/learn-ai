# AGENTS.md

## Workspace Purpose

This workspace is for learning AI, Transformer architectures, large language models, and related engineering practices.

The user is a software developer with basic AI knowledge and some prior agent-building experience. Prefer practical explanations that connect AI concepts to software engineering, model usage, and real implementation trade-offs.

## Current Structure

- `learn-ai/`: main learning area.
- `learn-ai/README.md`: initial project note.

This workspace is not currently a full software project and does not yet have a package manager, build system, or test suite. `git status` does not currently detect a valid repository at the root.

## How To Work Here

- Communicate with the user in Chinese unless they ask otherwise.
- Favor clear, practical learning paths over abstract theory dumps.
- When explaining AI topics, connect concepts to code, APIs, model behavior, and engineering workflows.
- Preserve user-created notes and learning artifacts. Do not delete, overwrite, or heavily reorganize files unless the user explicitly asks.
- Keep new files focused and easy to scan. This workspace is for learning, so documentation clarity matters more than clever structure.
- If adding code examples, prefer small runnable examples with comments that explain the AI-specific parts.
- If adding project scaffolding, first inspect the existing files and choose the lightest structure that serves the learning goal.

## Suggested Organization

When adding learning material, prefer this general structure:

- `learn-ai/notes/`: concept notes, summaries, study logs.
- `learn-ai/examples/`: small runnable code examples.
- `learn-ai/projects/`: larger hands-on experiments.
- `learn-ai/resources/`: curated links, papers, courses, and references.

Only create these directories when they are actually needed.

## AI Learning Preferences

The user is interested in learning topics such as:

- Transformer fundamentals.
- Hugging Face Transformers.
- LLM application development.
- Agents and tool-use systems.
- Practical model inference, fine-tuning, evaluation, and deployment.

Good explanations should usually include:

- What the concept is.
- Why it matters.
- How it appears in real projects.
- A minimal code or workflow example when useful.
- Common pitfalls for software engineers entering AI.

## External Repositories

When discussing external open-source projects such as `huggingface/transformers`:

- Treat them as references and learning targets, not as code to vendor into this workspace by default.
- If cloning or downloading dependencies is needed, ask or use the appropriate approval flow.
- Prefer reading official docs and small source files before recommending broad source-code dives.

## Commands

No standard commands are defined yet.

If future agents add a runnable project, update this section with exact commands, for example:

- setup command
- test command
- lint command
- run command

## Verification

For documentation-only changes, verify by reading the edited file.

For code changes, run the narrowest relevant command available. If no test or build command exists, state that clearly in the final response.

## Handoff Notes For Future Agents

- Start by reading this file and `learn-ai/README.md`.
- Check the current directory structure before making changes.
- Ask fewer broad questions; propose a sensible next learning step when the user is exploring.
- Keep the workspace beginner-friendly but not superficial. The user can handle engineering depth when it is grounded in practical examples.
