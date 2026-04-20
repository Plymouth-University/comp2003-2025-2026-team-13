import pandas as pd

#read activity.csv
df = pd.read_csv("activity.csv", header=None)
df.columns=["timestamp", "activity"]

#set time stamp to a datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])

#sort the timestamp. Probably not needed but just in case
df = df.sort_values("timestamp")

#get the next time stamp and compare the difference
df["next_time"] = df["timestamp"].shift(-1)
df["duration"] = (df["next_time"] - df["timestamp"]).dt.total_seconds()

#remove the duration values that end up being several days long 
df.loc[df["duration"] > 1200, "duration"] = 0

#remove the dates without duration
df = df.dropna(subset=["duration"])

#get the date off the timestamp
df["date"] = df["timestamp"].dt.date

summary = df.groupby(["date", "activity"])["duration"].sum().unstack(fill_value=0)

#if a comlumn is 0, set it to 0
for col in ["Sitting", "Standing", "Laying", "Falls"]:
    if col not in summary.columns:
        summary[col] = 0


#set the times to minutes, put them into the correct format and save to activity_profile.csv
summary = summary[["Sitting", "Standing", "Laying", "Falls"]]
summary = summary / 60
summary = summary.round(2)
summary.to_csv("activity_profile.csv")
print(summary)