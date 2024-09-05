import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# Unique category labels
color_labels = df['gage_number'].unique()

# List of RGB triplets
rgb_values = sns.color_palette("Set2", len(color_labels))

# Map label to RGB
color_map = dict(zip(color_labels, rgb_values))
category_colors = df['gage_number'].map(color_map)

# Create scatter plot
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['dateTime'], df['value'], c=category_colors, s=2)

# Create custom legend
handles = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                      markerfacecolor=color_map[label], markersize=5) for label in color_labels]

# Add the legend to the plot, adjusting its position
plt.legend(handles=handles, title='Gage Number', bbox_to_anchor=(1.02, 1), loc='upper left')

# Show the plot
plt.tight_layout()
plt.show()