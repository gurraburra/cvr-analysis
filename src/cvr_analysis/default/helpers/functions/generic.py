#%%
import numpy as np
import json
import hashlib
from collections.abc import Mapping, Sequence
# %% remove nan values
def removeNan(timeseries):
    # series is nan
    timeseries_nan = np.any(np.isnan(timeseries), axis = 1)
    # remove trailing and leadning
    # check if all nan
    if np.all(timeseries_nan):
        timeseries_leading_nan = timeseries.shape[0]
    else:
        timeseries_leading_nan = timeseries_nan.argmin()
    timeseries_trailing_nan = timeseries_nan[::-1].argmin()
    # check intermediate
    if np.any(timeseries_nan[timeseries_leading_nan:len(timeseries) - timeseries_trailing_nan]):
        raise ValueError("Series a contains intermediate nan values.")
    # return
    return timeseries[timeseries_leading_nan:len(timeseries) - timeseries_trailing_nan], timeseries_leading_nan, timeseries_trailing_nan

# %%
##############################################
# create hash
##############################################
def _normalize_numbers(x):
    # Normalize floats
    if isinstance(x, float):
        # Treat -0.0 and 0.0 as the same
        if x == 0.0:
            return 0.0
        return x
    # Recurse into dicts
    if isinstance(x, Mapping):
        return {k: _normalize_numbers(v) for k, v in x.items()}
    # Recurse into lists/tuples
    if isinstance(x, Sequence) and not isinstance(x, (str, bytes, bytearray)):
        t = type(x)
        return t(_normalize_numbers(v) for v in x)
    return x

def stable_dict_hash(d: dict) -> str:
    normalized = _normalize_numbers(d)
    s = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()
