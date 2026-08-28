import pandas as pd

from naturalv2.pipeline.ownership import OWNERSHIP_COL, gate_treatment_ownership


def test_treatment_ownership_gate():
    data = pd.DataFrame({OWNERSHIP_COL: ["Yes", "No", "Unclear"]})
    assert gate_treatment_ownership(data).index.tolist() == [0]
