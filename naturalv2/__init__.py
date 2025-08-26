"""NATURAL: NATural language analysis to Understand ReAL effects"""

import pandas as pd


# Use new copy-view behaviour using Copy-on-Write
# See: https://pandas.pydata.org/docs/user_guide/copy_on_write.html#copy-on-write-enabling
pd.options.mode.copy_on_write = True

# Infer sequence of str objects as PyArrow-backed StringDtype
# see: https://pandas.pydata.org/docs/user_guide/migration-3-strings.html#brief-introduction-to-the-new-default-string-dtype
pd.options.future.infer_string = True
