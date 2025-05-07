import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Need to refactor this
workspace = os.path.dirname(os.getcwd())

static = xr.open_dataset("~/../../work/nv25/Carrie/cypress_creek/data/combined_output/static_interp.zarr")

var_names = np.array([i for i in static.data_vars])

X = static.to_array().to_numpy()

Xmat = X.reshape(X.shape[0],(X.shape[1]*X.shape[2]))

Xmat_std =  np.std(Xmat,axis=1) # std for 32nd var is zero, so creates NAs to divide by it

indices_with_nonzero_var = np.setdiff1d(np.arange(Xmat.shape[0]),np.argwhere(Xmat_std == 0))

Xmat_cleaned = Xmat[indices_with_nonzero_var,:]
X_cleaned = X[indices_with_nonzero_var,:,:]
var_names_cleaned = var_names[indices_with_nonzero_var]

Xmat_cleaned_standardized = (np.transpose(Xmat_cleaned) - np.transpose(np.mean(Xmat_cleaned,axis=1)))/np.transpose(np.std(Xmat_cleaned,axis=1)) # So this creates NAs


import sklearn.decomposition

pca = sklearn.decomposition.PCA(n_components=10)
#Fit the model
pca.fit(Xmat_cleaned_standardized)
#Print the explained variances
print(pca.explained_variance_)

explained_variance = pca.explained_variance_/np.sum(pca.explained_variance_)
print("Explained variance: ", explained_variance)

n_pcs= pca.components_.shape[0]

most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]

print("Most important predictors in each PC: ", var_names_cleaned[most_important])

plt.figure(figsize=(10,5))
plt.plot(pca.explained_variance_/np.sum(pca.explained_variance_),lw=4)
plt.ylabel('Relative explained variance',fontsize=16)
plt.xlabel('Principal component',fontsize=16)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig("pca_variance_explained.png")

Y = pca.transform(Xmat_cleaned_standardized)

Xpred_std = pca.inverse_transform(Y)
#"Destandardize"
Xpred = Xpred_std*np.std(np.transpose(Xmat_cleaned),axis=0) + np.mean(np.transpose(Xmat_cleaned),axis=0)

Xpred_t = np.transpose(Xpred)

# Reshape the predictions so I can plot them
Xpred_reshaped = Xpred_t.reshape(Xpred_t.shape[0],3600,3600)

# top_k = 10
# out_dir = Path(f"{workspace}/figures/pca_top_predictors")
# dpi = 300
# cbar_kwargs = dict(orientation="horizontal",pad=0.03, shrink=0.90)
# out_dir.mkdir(parents=True, exist_ok=True)

# # Go through most important predictors
# for rank, idx in enumerate(most_important[:top_k], start=1):

#     # Same scale for original and reproduced
#     vmin = np.min(X_cleaned[idx])
#     vmax = np.max(X_cleaned[idx])

#     # Create figure
#     fig, axes = plt.subplots(1, 2, figsize=(10, 5))   # 1 row × 2 columns
#     titles = ["Original", "Reproduced"]

#     # Left panel – origianal
#     im0 = axes[0].imshow(X_cleaned[idx], vmin=vmin, vmax=vmax, cmap="viridis")
#     axes[0].set_title(titles[0], fontsize=16)
#     axes[0].set_ylabel(var_names_cleaned[idx], fontsize=14)
#     axes[0].set_xticks([]), axes[0].set_yticks([])
#     plt.colorbar(im0, ax=axes[0], **cbar_kwargs)

#     # Right panel – reproduced
#     im1 = axes[1].imshow(Xpred_reshaped[idx], vmin=vmin, vmax=vmax, cmap="viridis")
#     axes[1].set_title(titles[1], fontsize=16)
#     axes[1].set_xticks([]), axes[1].set_yticks([])
#     plt.colorbar(im1, ax=axes[1], **cbar_kwargs)

#     fig.tight_layout()

#     # Save each one
#     safe_name = var_names_cleaned[idx].replace(" ", "_")
#     filename  = out_dir / f"{rank:02d}_{safe_name}.png"
#     fig.savefig(filename, dpi=dpi, bbox_inches="tight")
#     plt.close(fig)

#     print(f"Saved {filename}")

# Save the PCs so that I can transform the tiles into PC components
