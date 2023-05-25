#azimuthally averaged advection of angular momentum

import numpy as np
import xarray as xr

# Load your dataset
ds1 = xr.open_dataset('azimuthal_avg_radialWind_cross_350.nc')
ds2 = xr.open_dataset('azimuthal_avg_tangentialWind_cross_350.nc')
ds3 = xr.open_dataset('azimuthal_avg_verticalWind_cross_350.nc')

# Extract the necessary variables
u = ds1['vx3'].values  # radial wind
v = ds2['vy3'].values  # tangential wind
w = ds3['w'].values  # vertical wind

dx = 1000.0  # x (radial) grid spacing in meters
dy = 1000.0  # y (tangential) grid spacing in meters
dz = 30.0    # z (vertical) grid spacing in meters

# Calculate r, the distance to the center of the storm
# Assuming that center is at grid point (600, 600)
xgrid = np.arange(0, 1202)
ygrid = np.arange(0, 1202)
xdist = (xgrid[:, np.newaxis] - 600) * dx
ydist = (ygrid[np.newaxis, :] - 600) * dy
r = np.sqrt(xdist**2 + ydist**2)
r_3d = np.repeat(r[np.newaxis, :, :], 70, axis=0)  # Repeat r to match the 3D shape of wind fields

# Calculate angular momentum m = r * v
m = v * r_3d

# Compute gradients of m
dm_dx = np.gradient(m, dx, axis=2)  # Gradient along x-axis
dm_dy = np.gradient(m, dy, axis=1)  # Gradient along y-axis
dm_dz = np.gradient(m, dz, axis=0)  # Gradient along z-axis

# Compute advections
Adv_r = u * dm_dx
Adv_t = v * dm_dy
Adv_z = w * dm_dz

# Add up to get total advection
Adv_total = Adv_r + Adv_t + Adv_z

# Perform azimuthal averaging
bins = np.arange(0, np.max(r), dx)  # Define the bins to average over
bin_indices = np.digitize(r, bins)  # Get bin indices for each grid point
Adv_total_avg = np.empty((70, len(bins)))

# Loop over vertical levels and bins to compute the average
for k in range(70):
    for i in range(len(bins)):
        if np.any(bin_indices == i):
            Adv_total_avg[k, i] = np.mean(Adv_total[k, bin_indices == i])
        else:
            Adv_total_avg[k, i] = np.nan


# Convert Adv_total_avg to xarray DataArray
# Here I am assuming the vertical level dimension is 'level' and the radial distance dimension is 'r'


level_coord = np.arange(70)  # Replace with the correct range if different
r_coord = xr.DataArray(bins, dims='r', coords={'r': bins})

Adv_total_avg_da = xr.DataArray(Adv_total_avg, dims=['level', 'r'], coords={'level': level_coord, 'r': r_coord})

# Convert Adv_total_avg_da to xarray Dataset for exporting to netCDF
Adv_total_avg_ds = Adv_total_avg_da.to_dataset(name='Adv_total_avg')

# Write to netCDF
Adv_total_avg_ds.to_netcdf('adv_total_avg.nc')
