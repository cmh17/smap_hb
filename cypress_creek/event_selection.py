import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# peaks over threshold algorithm

# rainfall below 0.1 in assumed to be zero

# 
df = pd.read_csv("gage_data.csv")
df['dateTime'] = pd.to_datetime(df['dateTime'])

# look at one gage to start
df1 = df[df["variable"] == 'Discharge, cubic feet per second'][df["gage_number"] == 8069000]
print(df1)

# plt.figure(figsize=(10, 6))
# scatter = plt.scatter(df1['dateTime'], df1['value'])
# plt.show()

# use a recursive digital filter to separate baseflow:
def lyne_hollick_filter(alpha, q_f_prev, q, q_prev):
    """
    lyne_hollick_filter to remove baseflow from a timeseries of flow
    
    :alpha: filter parameter to change shape of separation
    :q_f_prev: quickflow response at previous instant
    :q: original streamflow
    :q_prev: original streamflow at previous instant
    :return: q_f quickflow response
    """
    q_f = alpha*q_f_prev + (1+alpha)/2*(q - q_prev)
    if q_f <= 0:
        q_f = 0

    q_b = q - q_f

# apply the filter using the method described in Ladson et al. 2013
# first reflect 30 values of flow at start and end of filter

# initial values for each pass
# first pass: quickflow values set to initial value of streamflow for padded dataset, q31
# second pass: backward, start with q_{n-30}
# third pass: start with first value from q_b time series from second pass

# first pass: