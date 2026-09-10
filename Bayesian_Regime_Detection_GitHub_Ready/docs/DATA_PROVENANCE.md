
# Data Provenance

The supplied ZIP's README explicitly identifies all files as **synthetic demo data**, not sourced observations.

This distinction is critical:
- The model pipeline can be tested on this archive.
- Model mechanics can be demonstrated.
- Code and dashboard can be developed.
- The resulting probabilities can be used as educational outputs.

But the outputs must NOT be described as historical empirical findings about the real Indian equity market until the raw files are replaced by verified source data.

The supplied GDP quarter labels are malformed (`Q%q-YYYY`). The pipeline preserves the row order and maps the values sequentially from 2010Q1. This is a preprocessing assumption and should be replaced by correctly labelled source data.
