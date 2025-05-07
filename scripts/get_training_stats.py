import xarray as xr, json, numpy as np, os, collections

workspace = os.path.dirname(os.getcwd())

stat_train_path = f"{workspace}/data/combined_output/static_interp.zarr"
dyn_train_path = f"{workspace}/data/combined_output/dynamic_updated_averaged_interp.zarr"

print("Opening training tiles / years only …")
# For now use training split where we use all tiles but just 2016-2017 data for training
# And test on 2017-2018
stat_train_ds = xr.open_zarr(stat_train_path)

stat_stats = collections.defaultdict(dict)
# for static, we're using all of them in both train and test, so just get stats for all data
# Maybe change this later so we have a train/test split for the static data

for var in sorted(stat_train_ds.data_vars):
    da = stat_train_ds[var]
    vmin = float(da.min().compute())
    vmax = float(da.max().compute())
    mean = float(da.mean().compute())
    std  = float(da.std().compute())
    stat_stats[var]["min"]  = vmin
    stat_stats[var]["max"]  = vmax
    stat_stats[var]["mean"] = mean
    stat_stats[var]["std"]  = std
    print(f"{var}: min={vmin:.3f}, max={vmax:.3f}, mean={mean:.3f}, std={std:.3f}")
    
stat_stats_file = os.path.join(workspace, "data", "combined_output", "static_norm_stats.json")
with open(stat_stats_file, "w") as f:
    json.dump(stat_stats, f, indent=2)

dyn_train_ds = (
    xr.open_zarr(dyn_train_path)
      .sel(time=slice("2016-01-01", "2017-12-31"))
)

dyn_stats = collections.defaultdict(dict)

for var in sorted(dyn_train_ds.data_vars):
    da = dyn_train_ds[var]
    vmin = float(da.min().compute())
    vmax = float(da.max().compute())
    mean = float(da.mean().compute())
    std  = float(da.std().compute())
    dyn_stats[var]["min"]  = vmin
    dyn_stats[var]["max"]  = vmax
    dyn_stats[var]["mean"] = mean
    dyn_stats[var]["std"]  = std
    print(f"{var}: min={vmin:.3f}, max={vmax:.3f}, mean={mean:.3f}, std={std:.3f}")

dyn_stats_file = os.path.join(workspace, "data", "combined_output", "dynamic_norm_stats.json")
with open(dyn_stats_file, "w") as f:
    json.dump(dyn_stats, f, indent=2)
print(f"Saved stats at {dyn_stats_file}")
