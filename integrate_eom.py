import numpy as np
import scipy.integrate
import pyproj
import matplotlib.pyplot as plt 
from matplotlib import style
style.use('ggplot')

# Define constants
g = 9.81            # Acceleration due to gravity (ms^-2)
G = 6.67408e-11     # Gravitational constant (m^3 kg^-1 s^-2)
Me = 5.972e24       # Mass of earth (kg)
Re = 6371e3         # Radius of earth (m)
R = 8.31477         # Ideal gas constant (J mol^-1 K^-1)
M = 0.0289644       # Molar mass of dry air (kg mol^-1)
L = 0.0065          # Temeprature lapse rate (K Km^-1)
P0 = 101325         # Standard pressure at sea level (Pa)
T0 = 288.15         # Standard temperature at sea level (K)

def main():
    initial_conditions = [0, 0, 100000, 451, 892, 0]    # Initial conditions in local coords
    initial_position = [-5126024, 2609012, -2970630]    # Iniital position in ECEF coords
    res = run_simulation(initial_conditions, projectile='box', time=3000)

    dist = np.sqrt(res[:,0]**2 + res[:,1]**2)
    t_end = np.where(np.diff(np.sign(res[:,2])))[0][0]
    xfinal = res[:,0][t_end]
    yfinal = res[:,1][t_end]
    total_dist = dist[t_end]
    impact_coords = [xfinal, yfinal, Re]

    [x, y, z] = get_impact_coords(initial_position, impact_coords)

    lat, long = cvt_coordinates(x, y, z)

    print(lat, long)



    # print("Final X position: {:.2f}".format(xfinal/1000))
    # print("Final Y position: {:.2f}".format(yfinal/1000))
    # print("Total distance travelled: {:.2f} km".format(total_dist/1000), flush=True)

def projectile_EOM(y, t, mass, Cd, A):
    '''
    Returns state space equations of motion for falling particle in rectangular coordinate system
    accounting for quadratic drag, changing atmospheric conditions and varying gravity force.
    '''
    rho = get_density(y[2])
    dydt = np.zeros_like(y)

    # y vector of the form y = [x, y, z, Vx, Vy, Vz]
    dydt[0] = y[3]
    dydt[1] = y[4]
    dydt[2] = y[5]
    dydt[3] = (-1/2 * Cd * rho * A / mass) * np.sqrt(y[3]**2 + y[4]**2 + y[5]**2) * y[3]
    dydt[4] = (-1/2 * Cd * rho * A / mass) * np.sqrt(y[3]**2 + y[4]**2 + y[5]**2) * y[4]
    dydt[5] = -G*Me / ((y[2] + Re)**2) - (1/2 * Cd * rho * A / mass) * np.sqrt(y[3]**2 + y[4]**2 + y[5]**2) * y[5]

    return dydt

def get_density(altitude):
    """
    Calculate the air density as a function of altitude. Rough approximation only to
    approximate the trajectory of falling debris. 

    0m - 35000m      -> https://en.wikipedia.org/wiki/Density_of_air#Altitude
    35000m - >84000m -> http://www.braeunig.us/space/atmmodel.htm   
    """
    if altitude <= 35000:
        T = T0-L*altitude
        P = P0 * (1 - L*altitude/T0)**(g*M/(R*L))

    elif altitude <= 47000:
        T = 139.05 - 2.8 * altitude / 1000
        P = 868.0187 * (228.65 / (228.65 + 2.8 * ((altitude/1000)-32)))**(34.1632/2.8) / 10

    elif altitude <= 51000:
        T = 270.65
        P = 110.9063 * np.exp(-34.1632 * (altitude/1000) - 47 / 270.65)

    elif altitude <= 71000:
        T = 413.45 - 2.8 * altitude / 1000
        P = 66.93887 * (270.65 / (270.65 - 2.8 * (altitude/1000 - 51))) ** (34.1632 / -2.8) 
    
    else:
        if altitude > 84852:
            altitude = 84852
        T = 356.65 - 2 * altitude / 1000
        P = 3.956420 * (214.65 / (214.65 - 2 * (altitude/1000 - 71)))**(34.1632 / -2)
    
    rho = P*M/(R*T)
    return rho

def run_simulation(y0, projectile='plate', time=1000):
    if projectile == 'plate':
        mass = 16
        A = 4
        Cd = 1.28
    elif projectile == 'box':
        mass = 40
        A = 1.6384
        Cd = 1.28
    elif projectile == 'cylinder':
        mass = 100
        A = .12
        Cd = 0.47
    else:
        raise ValueError("Please enter a valid projectile")

    t = np.linspace(0, time, time*100)
    res = scipy.integrate.odeint(projectile_EOM, y0, t, args=(mass, Cd, A))
    return res

def get_rotation_matrix(i_v):
    """
    Finds rotation matrix that maps to a coordinate frame with Z axis aligned
    with input vector.
    From http://www.j3d.org/matrix_faq/matrfaq_latest.html#Q38
    Adapted from https://stackoverflow.com/questions/43507491/imprecision-with-rotation-matrix-to-align-a-vector-to-an-axis
    """
    unit = [0, 0, 1]
    # Normalize vector length
    i_v /= np.linalg.norm(i_v)

    # Get axis
    uvw = np.cross(i_v, unit)

    # compute trig values - no need to go through arccos and back
    rcos = np.dot(i_v, unit)
    rsin = np.linalg.norm(uvw)

    #normalize and unpack axis
    if not np.isclose(rsin, 0):
        uvw /= rsin
    u, v, w = uvw

    # Compute rotation matrix - re-expressed to show structure
    return (
        rcos * np.eye(3) +
        rsin * np.array([
            [ 0,  w, -v],
            [-w,  0,  u],
            [ v, -u,  0]
        ]) +
        (1.0 - rcos) * uvw[:,None] * uvw[None,:]
    )

def get_impact_coords(initial_position, impact_coords):
    """
    Calculates the ECEF coordinates of impact using the inverse transform
    from global to local coordinates
    """
    R = get_rotation_matrix(initial_position)
    return np.matmul(impact_coords, np.transpose(R))          # Rotation matrix is orthonormal >> use transpose rather that inv

def cvt_coordinates(x, y, z):
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    lon, lat, alt = pyproj.transform(ecef, lla, x, y, z, radians=False)
    return lat, lon

# plt.figure()

# plt.subplot(231)
# plt.plot(t, res[:,0])
# plt.xlabel('Time (s)')
# plt.ylabel('Distance (m)')
# plt.title('X Distance')

# plt.subplot(232)
# plt.plot(t, res[:,1])
# plt.xlabel('Time (s)')
# plt.ylabel('Distance (m)')
# plt.title('Y Distance')

# plt.subplot(233)
# plt.plot(t, res[:,2])
# plt.xlabel('Time (s)')
# plt.ylabel('Height (m)')
# plt.title('Altitude')

# plt.subplot(234)
# plt.plot(t, res[:,3])
# plt.xlabel('Time (s)')
# plt.ylabel('Velocity (m/s)')
# plt.title('X Velocity')

# plt.subplot(235)
# plt.plot(t, res[:,4])
# plt.xlabel('Time (s)')
# plt.ylabel('Velocity (m/s)')
# plt.title('Y Velocity')

# plt.subplot(236)
# plt.plot(t, res[:,5])
# plt.xlabel('Time (s)')
# plt.ylabel('Velocity (m/s)')
# plt.title('Vertical Velocity')

# plt.tight_layout()

# plt.figure()
# plt.plot(dist, res[:,2])
# plt.xlabel('Distance travelled (m)')
# plt.ylabel('Altitude (m)')
# plt.show()


if __name__ == '__main__':
    main()