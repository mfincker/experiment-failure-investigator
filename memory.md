# Project collaboration preferences

## Plan prototypes from the top down

For local prototypes and projects intended partly to develop my understanding, work in ascending order of implementation complexity and abstraction.

- Begin with the user-facing workflow and the simplest end-to-end architecture. Show the controller, orchestration flow, and main agent behavior first, using pseudocode or thin placeholders where necessary.
- Build a minimal vertical slice early so I can see how input becomes output before we introduce supporting layers.
- Add contracts, schemas, adapters, registries, and other abstractions only when the concrete code makes their purpose visible.
- Before adding an abstraction, explain the problem it solves, what depends on it, and why it is useful at the current stage.
- Prefer a small amount of duplication over a premature abstraction during early prototyping; refactor once a repeated need or boundary is demonstrated.
- Keep the intended production architecture in view, but separate what the prototype needs now from what belongs in the backlog.
- Organize implementation plans so each step produces something runnable, inspectable, or otherwise easy to review.

The objective is not to avoid sound architecture. It is to reveal the architecture progressively, so each layer is motivated by an already-understood part of the system.
