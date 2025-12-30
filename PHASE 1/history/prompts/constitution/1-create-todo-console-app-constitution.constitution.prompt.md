---
id: 1
title: Create TODO Console App Constitution
stage: constitution
date: 2025-12-30
surface: agent
model: claude-opus-4-5-20251101
feature: none
branch: master
user: user
command: sp.constitution
labels: [constitution, todo-app, python]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - .specify/memory/constitution.md
tests:
---

## Prompt

"TODO Console App (Python) – Unified Constitution

\- Purpose
 - Build a fast, reliable, single-user TODO application for the command line.
 - Prioritize simplicity, correctness, and extensibility.

\- Platform \& Boundaries
 - Console-based only (no GUI, no web).
 - Python 3.x using standard library only.
 - No external dependencies or frameworks.
 - No authentication, networking, or cloud storage.

\- Core Features (Mandatory)
 - Add tasks with a non-empty title and optional description.
 - View all tasks in a clear, ordered list.
 - Update existing task details by ID.
 - Delete tasks explicitly by ID.
 - Mark and unmark tasks as complete.

\- Data Model
 - Tasks stored in memory as dictionaries or simple objects.
 - Required fields:
   - id (unique integer, immutable)
   - title (string, non-empty)
   - completed (boolean)
 - Optional field:
   - description (string)

\- Task Rules
 - Each task must have a unique ID.
 - Deleted task IDs must not be reused.
 - Completion status must always be visible.

\- Program Structure
 - Single entry point: main().
 - Menu-driven loop using while True.
 - Business logic separated into small, focused functions.
 - Global mutable state limited to the task list only.

\- Required Functions
 - add\_task(tasks)
 - view\_tasks(tasks)
 - update\_task(tasks, task\_id)
 - delete\_task(tasks, task\_id)
 - toggle\_task\_completion(tasks, task\_id)

\- Input \& Output Rules
 - All user input must be validated.
 - Invalid input must not crash the program.
 - Re-prompt on invalid input.
 - Task display must show:
   - ID
   - Title
   - Completion status (\[✓] or \[ ])

\- Error Handling
 - No unhandled exceptions.
 - Graceful handling of:
   - Invalid or missing task IDs
   - Empty task list
   - Non-numeric input

\- Persistence (Optional, MVP+)
 - JSON file storage only.
 - Load data on startup.
 - Save on every data mutation.

\- Coding Discipline
 - Follow PEP 8 naming conventions.
 - One responsibility per function.
 - Avoid deeply nested conditionals.
 - Code must be readable and beginner-friendly.

"

## Response snapshot

Created a comprehensive constitution for the TODO Console App with six core principles: Console-First Interface, Python Standard Library Only, Data Integrity, Error Handling, Persistence, and Task Management Core. The constitution defines platform boundaries, development standards, and governance rules.

## Outcome

- ✅ Impact: Constitution created with all required principles and governance structure
- 🧪 Tests: N/A
- 📁 Files: Updated .specify/memory/constitution.md with new constitution content
- 🔁 Next prompts: Implement the TODO console application based on the constitution
- 🧠 Reflection: The constitution clearly defines the constraints and principles for the TODO console app development

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Begin implementing the TODO console app functions according to the constitution