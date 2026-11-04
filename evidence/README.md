# Execution and result evidence

[Results guide](../docs/results.md)

| File | Source and scope |
|---|---|
| `training_report.json` | Unchanged 31-stage training report from the October 10 Docker run |
| `final_api.json` | Unchanged GPU Final API and state-reset check |
| `fresh_replay.json` | Unchanged complete public replay of the newly trained model |
| `sealed_replay.json` | Unchanged complete public replay of sealed submission 948801 |
| `validation_report.json` | Unchanged validation supervisor result, including internal quality gate |
| `execution_summary.json` | Selected hardware/isolation/timing fields from the historical training supervisor; source hash included |
| `provenance.json` | SHA256 and original relative paths for the five unchanged records |
| `paired_quality.csv` | Values and differences calculated from the two replay reports |
| `official_results.json` | Separate transcription of the live official final-results row, with URL and retrieval date |

Replay input, target, manifest, scorer, and evaluator-code hashes match between
the two reports. Prediction hashes intentionally differ because the trained
weights differ. All six compilation/graph flags are true in both reports.

The public score records do not contain private evaluation samples. Container
paths inside historical receipts are retained for traceability and are not local
paths readers need to reproduce. The complete training intermediates, models,
private emails, cloud-account settings, and billing logs are not part of this
public evidence selection.
