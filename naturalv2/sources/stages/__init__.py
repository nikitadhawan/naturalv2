"""Generic curation stages shared across sources.

This namespace aggregates reusable, source-agnostic stages that plug into the
curation pipeline. Stages here typically depend only on the core curation
interfaces (``CurationContext``, ``StageState``, and ``SourceStage``) and may
leverage language models for higher-level extraction tasks.
"""

from naturalv2.sources.stages.synonyms import SynonymStage
