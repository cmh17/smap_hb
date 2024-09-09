import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# peaks over threshold algorithm

# rainfall below 0.1 in assumed to be zero

# 
df = pd.read_csv("E:\hydro\smap_hb\gage_data.csv")
df['dateTime'] = pd.to_datetime(df['dateTime'])
# print(df)

# look at one gage to start
df1 = df[df["variable"] == 'Discharge, cubic feet per second']
df1 = df1[df1["gage_number"] == 8069000]
# print(df1)


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
    return(q_f,q_b)

# apply the filter using the method described in Ladson et al. 2013
# first reflect 30 values of flow at start and end of filter

# initial values for each pass
# first pass: quickflow values set to initial value of streamflow for padded dataset, q31
# second pass: backward, start with q_{n-30}
# third pass: start with first value from q_b time series from second pass

# first pass:


# need to start at q31 and assume that q_f for 1-30 are just the given values
a_init = 0.9
q = df1["value"].to_numpy()
first_pass_qf = []
first_pass_qf = np.empty(len(q), dtype=object)
first_pass_qb = np.empty(len(q), dtype=object)
first_pass_qf[0] = q[0]


print("q:",q)
print(first_pass_qf)

for i in range(1,len(q)):
    # print(a_init,first_pass_qf[i-1],q[i],q[i-1])
    first_pass_qf[i],first_pass_qb[i]  = lyne_hollick_filter(a_init,first_pass_qf[i-1],q[i],q[i-1])



first_pass_qb = q - first_pass_qf

print(first_pass_qf)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(df1['dateTime'], first_pass_qf,s=1)
scatter1 = plt.scatter(df1['dateTime'], first_pass_qb,s=1)
plt.show()

# second pass:
second_pass_qf = np.empty(len(q), dtype=object)
second_pass_qb = np.empty(len(q), dtype=object)
second_pass_qf[len(q)-1] = first_pass_qf[len(q)-1]

for i in range(len(q)-1,0, -1):
    second_pass_qf[i],second_pass_qb[i] = lyne_hollick_filter(a_init,first_pass_qf[i-1],first_pass_qb[i],first_pass_qb[i-1])

plt.figure(figsize=(10, 6))
scatter = plt.scatter(df1['dateTime'], second_pass_qf,s=1)
scatter1 = plt.scatter(df1['dateTime'], second_pass_qb,s=1)
plt.show()

# # third pass
# third_pass_qf = np.empty(len(q), dtype=object)
# third_pass_qb = np.empty(len(q), dtype=object)
# third_pass_qf[0] = second_pass_qf[0]

# for i in range(1,len(q)):
#     # print(a_init,first_pass_qf[i-1],q[i],q[i-1])
#     third_pass_qf[i],third_pass_qb[i]  = lyne_hollick_filter(a_init,second_pass_qf[i-1],second_pass_qb[i],second_pass_qb[i-1])


