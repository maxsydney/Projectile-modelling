import numpy as np

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
V = [44.674, 88.359, 6371]
N = [-5126, 2609, -2970]
R = get_rotation_matrix(N)

R = np.matmul(V, np.transpose(R))# - [6378.1, 0, 0]
print(R)
