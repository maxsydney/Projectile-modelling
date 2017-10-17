import pyproj

# Example position data, should be somewhere in Germany
x = -5047000.173
y = 2568000.878
z = -2924000.025

ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
lon, lat, alt = pyproj.transform(ecef, lla, x, y, z, radians=False)

print(lat, lon, alt)