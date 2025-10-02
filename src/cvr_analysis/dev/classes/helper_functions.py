#%%
import numpy as np
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
