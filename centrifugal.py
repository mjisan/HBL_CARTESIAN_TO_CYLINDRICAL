import numpy as np
import xarray as xr

# Assuming you have a DataArray for the tangential wind 'vt'
ds = xr.open_dataset('test_vt.nc')
vt = ds['vy3']
vr = ds['vx3']
z0 = ds['z0']

print(np.shape(z0))
print(np.shape(vr))

# Assuming storm center at (x_center, y_center)
x_center = 600
y_center = 600

# Calculate the grid relative to storm center
x = np.arange(vt.shape[1]) - x_center
y = np.arange(vt.shape[0]) - y_center
y, x = np.meshgrid(y, x)

# Calculate distance to storm center
r = np.sqrt(x**2 + y**2)

# Calculate centrifugal term
centrifugal_term = vt**2 / r

# If you wish to handle division by zero (at the storm center),
# you can use np.where to set the centrifugal term to 0 at the center:
centrifugal_term = np.where(r != 0, vt**2 / r, 0)

print("Minimum centrifugal term (ignoring NaNs): ", np.nanmin(centrifugal_term))
print("Maximum centrifugal term (ignoring NaNs): ", np.nanmax(centrifugal_term))

f = 6.38e-05  # Coriolis parameter

# Assuming you have a 2D numpy array for tangential wind "vt"
# with dimensions (1202, 1202)

# Calculate Coriolis term
coriolis_term = (2 * f * vt)/10

# Print the minimum and maximum values of Coriolis term
print("Minimum Coriolis term: ", np.nanmin(coriolis_term))
print("Maximum Coriolis term: ", np.nanmax(coriolis_term))


kappa = 0.4  # Von Karman's constant

# Assuming you have a 2D numpy array for zonal wind "um" and surface roughness "z0"
# both with dimensions (1202, 1202)

# A small number to avoid log(0) situation
epsilon = 1e-8

# Calculate the frictional force at the surface
#F_r = - (kappa**2 * vr / np.log(z0 + epsilon))**2

# Print the minimum and maximum values of Frictional term
#print("Minimum Frictional term: ", np.nanmin(F_r))
#print("Maximum Frictional term: ", np.nanmax(F_r))






print("Shape of vr before operation: ", vr.shape)

# Calculate log(z0 + epsilon)
log_z0 = np.log(z0.values + epsilon)
print("Shape of log_z0: ", log_z0.shape)

# Calculate division result
division_result = vr.values / log_z0
print("Shape of division result: ", division_result.shape)

# Calculate square
squared_result = division_result ** 2
print("Shape of squared result: ", squared_result.shape)

# Multiply by kappa squared
F_r = - (kappa ** 2 * squared_result)
print("Shape of F_r: ", F_r.shape)

print("Minimum Frictional term: ", np.nanmin(F_r))
print("Maximum Frictional term: ", np.nanmax(F_r))
