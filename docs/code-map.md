# Code map

[Home](../README.md)

Start with these files rather than the retained historical dependency trees.

| Question | Entry point |
|---|---|
| What functions does the organizer call? | [`solution.py`](../submission/solution.py) |
| What runs, and in what order? | [`pipeline.py`](../submission/pipeline.py) |
| How are recovered recipes connected to a fresh run? | [`stages.py`](../submission/stages.py) |
| What is the exact recipe? | [`METHOD.md`](../submission/METHOD.md) |
| What data/splits and teacher settings are fixed? | [`assets/`](../submission/assets/) |
| How does one causal inference step work? | [`submission.py`](../submission/runtime/track2/submission.py) |
| How is the base forecast produced? | [`base_submission.py`](../submission/runtime/track2/base_submission.py) |
| How is fluctuation shape combined? | [`causal_shape.py`](../submission/runtime/track2/causal_shape.py) |
| How does the feature readout adapt? | [`feature_readout_runtime.py`](../submission/runtime/track2/feature_readout_runtime.py) |
| How are regional amplitudes updated? | [`regional_amplitude.py`](../submission/runtime/track2/regional_amplitude.py) |
| How are uncertainty CDFs refreshed? | [`periodic_cdf.py`](../submission/runtime/track2/periodic_cdf.py) |
| How is the environment pinned? | [`Dockerfile`](../submission/Dockerfile) |

`legacy/` and `teacher_legacy/` preserve imports and recipe dependencies from the
competition training code. Some filenames describe earlier experiments. Their
presence does not mean every file runs in the final method: `pipeline.schedule(2)`
and `stages.py` define the executed path. Historical absolute paths and artifact
identifiers in those modules are adapted by the fixed stage driver.

The public package intentionally retains these files so the original dependency
chain remains inspectable and byte-identical. Run the documented Final API rather
than historical experiment scripts directly.
