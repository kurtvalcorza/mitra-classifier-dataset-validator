# DIMER portal build contract — classification validator (issue #3)

Traceability for *Align validator repository with DIMER portal build contract*.
DIMER's Pipeline Builder builds this repository's default branch with the
repository **root** as the Docker build context and launches the validation
container by the portal naming convention (`validate.py`).

This is a compatibility/compliance change only. It does **not** alter classifier
validation semantics — `validator.py` (the tested implementation and the source
of the cross-repo shared block) is unchanged.

| Contract requirement | Status | Evidence |
|---|---|---|
| `Dockerfile` at repository root | ✅ | root `Dockerfile` |
| `requirements.txt`, `README.md` at root | ✅ | present at root |
| Root `validate.py` entrypoint delegating to `validator.py` | ✅ | `validate.py` → `from validator import main` / `sys.exit(main())` |
| Dockerfile invokes `validate.py` | ✅ | `CMD ["python", "validate.py"]`; `COPY validator.py validate.py ./` |
| `validator.py` preserved as implementation (tests + parity intact) | ✅ | `validator.py` byte-identical to prior `main`; `scripts/check_shared.py` green |
| `.gitignore` / `.dockerignore` exclude local dataset/result artifacts | ✅ | `.dockerignore` added; `.gitignore` extended with `*.csv`,`*.zip`,`data/`,`datasets/`,`results/`,`result.json` |
| Image consumes `DIMER_DATASET_DIR`, writes JSON to `DIMER_RESULT_PATH` | ✅ | unchanged in `validator.py`; env contract preserved |
| Builds from repository root as DIMER CodeBuild does | ✅ | `docker build .` from root; context limited by `.dockerignore` |
| Validator unit suite + shared-code parity green after the change | ✅ | `pytest` + `scripts/check_shared.py` (see PR checks) |

Container execution reaches the existing `validator.main()` through `validate.py`,
preserving current validation and result behavior.

`Closes #3`.
