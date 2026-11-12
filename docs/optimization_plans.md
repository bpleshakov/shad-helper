# Optimization plans

Audit date: 2026-11-12. Status: analysis only; recommendations are not implemented.

The project has worthwhile architectural improvement opportunities. The audit
found very little clearly unused code; the main opportunity is making
responsibilities and execution flow easier to follow.

The review covered all 33 production modules, their dependencies, references
across tests and scripts, and read-only lint checks.

The unused-code findings are small:

- [`check_audio_job()`](../shad_helper/jobs.py#L40) only forwards to `check_job()`.
  Its only repository callers are tests. It is a removal candidate if external
  callers do not need that compatibility name.
- Every production module is reachable in the static dependency map from the CLI
  or worker entry points.
- Production code and scripts pass Ruff's unused-import and unused-variable
  checks. Five warnings in the tests are false positives: those imports are
  accessed dynamically through [`globals()`](../tests/test_auth.py#L25).

Static analysis cannot prove that every branch executes, but the audit found no
large abandoned subsystem.

The recommended improvements, in priority order, are:

1. **Make stage results explicit.**

   Notes generation returns either a finished TeX path or a manual-handoff
   directory. Callers determine success by comparing paths—see
   [`pipeline.py`](../shad_helper/pipeline.py#L51) and
   [`cli.py`](../shad_helper/cli.py#L170).

   A small result object containing `status` and `path` would make completion
   explicit. Similarly, Anki could return its ending card position alongside the
   package path, avoiding the caller reopening `report.json`. These changes would
   clarify the contracts between stages.

2. **Remove dependencies that point back into orchestration.**

   Both illustration modules import
   [`notes._record_request()`](../shad_helper/notes.py#L517), while notes generation
   depends on those illustration modules. Local imports make this workable, but
   understanding one component requires following a dependency cycle.

   Move request recording into a small shared module. Likewise, put the shared
   TeX validator behind a public interface: currently
   [`anki_latex.py`](../shad_helper/anki_latex.py#L15) imports a private function
   from the full notes renderer. Preserve the existing shared validation rules.

3. **Separate the largest workflows by responsibility.**

   [`_run_notes()`](../shad_helper/notes.py#L719) spans 203 lines and combines
   preparation, imports, automatic generation, chapter splitting, checkpoints,
   fallback, illustration processing, and publication.

   [`write_project()`](../shad_helper/rendering.py#L332) spans 202 lines and combines
   document organization, block rendering, asset handling, and manifest
   construction.

   Extract named steps with clear inputs and outputs. Start with ordinary
   functions in the existing modules. Moving the long notes prompt into
   `notes_prompt.py`, following the existing `anki_prompt.py` convention, would
   also make navigation easier.

4. **Simplify the test infrastructure before substantial refactoring.**

   Some tests parse production source into an AST, remove imports, and execute
   reconstructed modules—see
   [`test_openai_client.py`](../tests/test_openai_client.py#L85). This couples tests
   to source structure and obscures what actually runs.

   Normal imports with mocked dependencies would be easier to maintain. The
   client already accepts injectable transport and token-provider arguments.

   There is also a concrete baseline mismatch:
   [`DEFAULT_REASONING_EFFORT`](../shad_helper/openai_client.py#L16) is `"low"`,
   while [test assertions](../tests/test_openai_client.py#L115) expect `"max"`.
   Reconcile the intended behavior before using these tests to judge a refactor.

5. **Provide a short, accurate architecture guide.**

   The README is 664 lines and the stage-contract document is 2,045 lines. A short
   reading guide should explain entry points, stage inputs/outputs, artifact
   ownership, and completion/failure states.

   Some existing documentation has drifted: the
   [partial-failure description](stage-contracts.md#L1967) describes older
   extraction behavior, whereas
   [current extraction](../shad_helper/extraction.py#L315) preserves previous
   completed evidence and publishes the replacement only after required stages
   finish.

The existing sequential pipeline, separate model worker processes, resumable
checkpoints, and shared OpenAI client provide a useful foundation. The recommended
first code change is explicit stage results, followed by removing the
request-recording dependency cycles.

Verification was limited to source inspection, static reference/dependency
analysis, and Ruff. The test suite, model requests, and rendering pipeline were
not run. No files were changed during the audit, and the Git working tree was
clean before this document was added.
