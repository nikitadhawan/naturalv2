# ADR 0001: One curated Reddit report per known author

## Status

Accepted

## Decision

Within each experiment, concatenate curated candidate records that share a non-null
`author_key` before extraction. Keep unkeyed records separate, remove duplicate
permalinks, and order records by date and permalink.

Label text written by the author separately from initial-post context written by
someone else. Preserve the downstream string-valued `report` contract. This applies
only to records already selected for the experiment, not the author's full history.

## Consequences

Each known author contributes one analysis row and one row-level bootstrap unit.
Conflicting experiences remain visible to the extraction model, while prolific
authors produce longer prompts.
