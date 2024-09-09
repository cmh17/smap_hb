import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# with open('cypress_creek\cypress_creek_gages.json') as file:
#     gages = json.load(file)

# # print(gages)

# # Pretty Print JSON
# gages_formatted = json.dumps(gages, indent=4, sort_keys=True)
# # print(gages_formatted)
# gage_values = gages["value"]["timeSeries"]

# # print([x.keys() for x in gage_values])
# print(gage_values[5]["sourceInfo"]["siteCode"][0]["value"])
# print(gage_values[5]["variable"]["variableDescription"])
# print(gage_values[5]["variable"]["unit"]["unitCode"])

# # print(gage_values[6]["values"][0].keys())
# print(gage_values[6]["values"][0]["qualifier"][0]["qualifierCode"])
# # [print(x["value"],x["dateTime"]) for x in gage_values[7]["values"][0]["value"]]

# # [print(x["value"]) for x in gage_values[5]["values"][0]["value"]]

# rows = []
# for x in gage_values:
#     gage_number = x["sourceInfo"]["siteCode"][0]["value"]
#     variable = x["variable"]["variableDescription"]
#     print(x["values"][0]["qualifier"])
#     if len(x["values"][0]["qualifier"]) > 0:
#         qualifier = x["values"][0]["qualifier"][0]["qualifierCode"]
#     else:
#         qualifier = None

#     for y in x["values"][0]["value"]:
#         value = y["value"]
#         dateTime = y["dateTime"]
#         rows.append({
#                 'gage_number': gage_number,
#                 'variable': variable,
#                 'qualifier': qualifier,
#                 'value': value,
#                 'dateTime' : dateTime
#             })
    
# df = pd.DataFrame(rows)
# print(df)
# df.to_csv("gage_data.csv")

# Load data
df = pd.read_csv("gage_data.csv")
df['dateTime'] = pd.to_datetime(df['dateTime'])
df = df[df["variable"] ==  'Discharge, cubic feet per second']

# Unique category labels
color_labels = df['gage_number'].unique()

# List of RGB triplets
rgb_values = sns.color_palette("Set2", len(color_labels))

# Map label to RGB
color_map = dict(zip(color_labels, rgb_values))
category_colors = df['gage_number'].map(color_map)

# Create scatter plot
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['dateTime'], df['value'], c=category_colors, s=1)

# Create custom legend
handles = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                      markerfacecolor=color_map[label], markersize=5) for label in color_labels]

# Add the legend to the plot, adjusting its position
plt.legend(handles=handles, title='Gage Number', bbox_to_anchor=(1.02, 1), loc='upper left')

# Show the plot
plt.tight_layout()
plt.show()

# take log transform and then get quantiles

df_sample = df[df["gage_number"] == 8068720][["dateTime","value"]]
df_sample["log_value"] = np.log10(df_sample["value"])
df_sample = df_sample.dropna()
print(df_sample)
print("10-year flow: ", np.quantile(df_sample["log_value"], 0.9), 10**np.quantile(df_sample["log_value"],0.9))

# do analysis with both streamflow and rainfall
# one big event in april of 2016
# don't really need to look at Harvey bc 
# eventually if there's so much rainfall,
# then soil moisture doesn't matter

# look at moderate events where soil moisture
# antecedent conditions could make a difference

# one of the gages from the FWS has soil moisture
# good to look at the time series to plot for different
# rainfall magnitues, how long it takes soil moisture to go back

# look at peak in rainfall and peak in soil moisture:
# want to know if we have a significant rainfall event, how
# many days will the moisture stay in the soil?

# then if we have another rainfall event in the next seven
# days, maybe there will be more of a likelihood of flooding

# if there are back-to-back events, pay special attention to those

# place to start: quantiles
# first filter out 80th percentile plus
# and then zoom in from there
# get rid of like 95th percentile bc don't want super extreme

# poke True
# look at FWS gages!!

#####

# 1. pick some rain events between 2015 and 2019
# a. try to find 3 events - they don't have to have caused widespread
# flooding; just high flow/high rain events
# b. make slides showing results from analysis
# 2. get the model from True and try to run it
# 3. try to download some test data for our region

# AMS eligibility is pretty strict, so this is a good opportunity