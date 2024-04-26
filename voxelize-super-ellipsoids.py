import argparse
import itertools
import json
import math
import numpy as np
# import matplotlib.pyplot as plt

from datetime import datetime
from typing import Dict, List, Tuple
from pydantic import BaseModel
from datetime import datetime
from tqdm import tqdm

class LovamapDomainData(BaseModel):
    bead_count: int
    bead_voxel_count: Dict[str, int]
    created: datetime
    data_type: str
    domain_size: List[float]
    hip_file: str
    voxel_count: int
    voxel_size: float
    bead_data: Dict[str, List[Tuple[int, int]]]

class LovamapDomainDataEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return str(obj)
        return super().default(obj)

def formatted_print(data):
    max_len = max(len(key) for key in data.keys())
    for key, value in data.items():
        print(f'{key:>{max_len}} : {value}')

def quaternion_to_rotation(q):
    s = q[0]
    x = q[1]
    y = q[2]
    z = q[3]
    R = np.zeros((3,3))
    R[0,0] = 1 - 2*y**2 - 2*z**2
    R[0,1] = 2*x*y - 2*z*s
    R[0,2] = 2*x*z + 2*y*s
    R[1,0] = 2*x*y + 2*z*s
    R[1,1] = 1 - 2*x**2 - 2*z**2
    R[1,2] = 2*y*z - 2*x*s
    R[2,0] = 2*x*z - 2*y*s
    R[2,1] = 2*y*z + 2*x*s
    R[2,2] = 1 - 2*x**2 - 2*y**2
    return R

def super_ellipsoid(x, y, z, a, b, c, ep1, ep2):
    aa = np.power((x/a)**2, 1/ep2)
    bb = np.power((y/b)**2, 1/ep2)
    cc = np.power((z/c)**2, 1/ep1)
    return np.power(aa + bb, ep2/ep1) + cc - 1

def optimized_super_ellipsoid(xyz, radii, eps):
    # Ensure xyz, radii, and eps are NumPy arrays for efficient computation
    xyz = np.asarray(xyz)
    radii = np.asarray(radii)
    eps = np.asarray(eps)
    tmp = np.power((xyz / radii) ** 2,  1 / np.array([eps[-1], *eps[::-1]]))
    return np.power(np.sum(tmp[..., :2], axis=1), eps[1] / eps[0]) + tmp[..., 2] - 1

def super_ellipsoid(xyz, radii, eps):
    x, y, z = xyz
    a, b, c = radii
    ep1, ep2 = eps
    aa = np.power((x/a)**2, 1/ep2)
    bb = np.power((y/b)**2, 1/ep2)
    cc = np.power((z/c)**2, 1/ep1)
    return np.power(aa + bb, ep2/ep1) + cc - 1

# Important: these are inclusive ranges!
def collapse_ranges(arr):
    gaps = [[b, e] for b, e in zip(arr, arr[1:]) if b + 1 < e]
    ranges = iter(arr[:1] + sum(gaps, []) + arr[-1:])
    return [(b, e) for b, e in zip(ranges, ranges)]

# def plot(min_corner, max_corner, voxels_in_ellipsoids):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.set_aspect('equal')
#     ax.set_xlim(min_corner[0], max_corner[0])
#     ax.set_ylim(min_corner[1], max_corner[1])
#     ax.set_zlim(min_corner[2], max_corner[2])
#     for voxels in voxels_in_ellipsoids:
#         ax.scatter([v[0] for v in voxels], [v[1] for v in voxels], [v[2] for v in voxels])
#     plt.show()

def main(input, dx, output):
    data = None
    with open(input, 'r') as file:
        data = [list(map(float, line.strip().split(','))) for line in file.readlines() if not line.startswith('#')]

    if data is None:
        print('No data in file')
        return

    # The data contains the following in the order described:
    # rx, ry, rz, eps1, eps2, x, y, z, q0, q1, q2, q3

    ellipsoids = []
    min_corner = np.array([np.inf, np.inf, np.inf])
    max_corner = np.array([-np.inf, -np.inf, -np.inf])
    for (rx, ry, rz, eps1, eps2, x, y, z, q0, q1, q2, q3) in data:
        radius = np.array([rx, ry, rz])
        center = np.array([x, y, z])
        min_corner = np.minimum(min_corner, center - radius)
        max_corner = np.maximum(max_corner, center + radius)
        R = quaternion_to_rotation([q0, q1, q2, q3])
        eps = [eps1, eps2]
        ellipsoids.append((radius, center, R, eps))

    extents = max_corner - min_corner

    num_voxels_x = int(math.ceil(extents[0] / dx))
    num_voxels_y = int(math.ceil(extents[1] / dx))
    num_voxels_z = int(math.ceil(extents[2] / dx))

    # i_stride = 1
    j_stride = num_voxels_x
    k_stride = num_voxels_x * num_voxels_y

    max_corner = min_corner + np.array([num_voxels_x, num_voxels_y, num_voxels_z]) * dx
    extents = max_corner - min_corner

    info = {
        'Number of particles': len(data),
        'Min corner': min_corner,
        'Max corner': max_corner,
        'Extents': extents,
        'Grid size': f'{num_voxels_x} x {num_voxels_y} x {num_voxels_z}',
        'Number of voxels': num_voxels_x * num_voxels_y * num_voxels_z,
        'Voxel size': dx
    }
    formatted_print(info)

    x_ticks = np.linspace(min_corner[0], max_corner[0], num_voxels_x + 1)
    y_ticks = np.linspace(min_corner[1], max_corner[1], num_voxels_y + 1)
    z_ticks = np.linspace(min_corner[2], max_corner[2], num_voxels_z + 1)

    # Gridpoints lie at the center of voxels
    x_pts = [x + dx/2 for x in x_ticks[0:-1]]
    y_pts = [y + dx/2 for y in y_ticks[0:-1]]
    z_pts = [z + dx/2 for z in z_ticks[0:-1]]

    voxel_matlab_idxs_in_ellipsoids = []
    voxels_in_ellipsoids = []
    for radius, center, R, eps in tqdm(ellipsoids):
        bbox_min = center - (radius + dx)
        bbox_max = center + (radius + dx)

        bbox_i_pts = [i for i in range(len(x_pts)) if bbox_min[0] <= x_pts[i] <= bbox_max[0]]
        bbox_j_pts = [j for j in range(len(y_pts)) if bbox_min[1] <= y_pts[j] <= bbox_max[1]]
        bbox_k_pts = [k for k in range(len(z_pts)) if bbox_min[2] <= z_pts[k] <= bbox_max[2]]

        bbox_x_pts = [x_pts[i] for i in bbox_i_pts]
        bbox_y_pts = [y_pts[j] for j in bbox_j_pts]
        bbox_z_pts = [z_pts[k] for k in bbox_k_pts]

        ijk = [(i, j, k) for i, j, k in itertools.product(bbox_i_pts, bbox_j_pts, bbox_k_pts)]

        x, y, z = np.meshgrid(bbox_x_pts, bbox_y_pts, bbox_z_pts, indexing='ij')
        points = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
        voxel_transformed = (R.T @ (points - center).T).T
        values = optimized_super_ellipsoid(voxel_transformed, radius, eps)

        voxels = []
        voxel_idxs = []
        for (i, j, k), value in zip(ijk, values):
            if value < 0:
                voxels.append((x_pts[i], y_pts[j], z_pts[k]))
                # Matlab is column-major, and I think ndgrid respects this...
                # But Matlab is also 1-indexed
                voxel_idxs.append((i + 1) + j * j_stride + k * k_stride)

        # Slow but brute forece way - keep this around for debugging
        # voxels = []
        # voxel_idxs = []
        # for i, j, k in itertools.product(bbox_i_pts, bbox_j_pts, bbox_k_pts):
        #     voxel = np.array([x_pts[i], y_pts[j], z_pts[k]])
        #     voxel_shifted = voxel - center
        #     voxel_final = R.T @ voxel_shifted
        #     f = super_ellipsoid(voxel_final, radius, eps)
        #     if f < 0:
        #         voxels.append(voxel)
        #         # Matlab is column-major, and I think ndgrid respects this...
        #         # But Matlab is also 1-indexed
        #         voxel_idxs.append((i + 1) + j * j_stride + k * k_stride)

        voxel_idxs.sort()
        voxels_in_ellipsoids.append(voxels)
        voxel_matlab_idxs_in_ellipsoids.append(collapse_ranges(voxel_idxs))

    # Debugging
    # with open('optimized.txt', 'w') as file:
    #     for i, voxel_idxs in enumerate(voxel_matlab_idxs_in_ellipsoids, 1):
    #         file.write(f'Ellipsoid {i}\n')
    #         for start, end in voxel_idxs:
    #             file.write(f'{start} {end}\n')
    #         file.write('\n')

    lovamapDomainData = LovamapDomainData(
        bead_count       = len(ellipsoids),
        bead_voxel_count = {str(i): len(vxls) for i, vxls in enumerate(voxel_matlab_idxs_in_ellipsoids, 1)},
        created          = datetime.now(),
        data_type        = 'labeled',
        domain_size      = extents.tolist(),
        hip_file         = 'N/A',
        voxel_count      = num_voxels_x * num_voxels_y * num_voxels_z,
        voxel_size       = dx,
        bead_data        = {str(i): vxls for i, vxls in enumerate(voxel_matlab_idxs_in_ellipsoids, 1)}
    )

    with open(output, 'w') as file:
        json.dump(lovamapDomainData.model_dump(), file, indent=2, cls=LovamapDomainDataEncoder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create LOVAMAP input json file representing a super-ellipsoid domain')
    parser.add_argument('--input', type=str, help='Name of the input CSV file containing super-ellipsoid data')
    parser.add_argument('--dx', type=float, help='Voxel size')
    parser.add_argument('--output', type=str, help='Name of the output JSON file')
    args = parser.parse_args()

    # if args.input is None:
    #     print('No input filename provided')
    #     exit(1)
    # if args.dx is None:
    #     print('No voxel size provided')
    #     exit(1)
    # if args.output is None:
    #     print('No output filename provided')
    #     exit(1)

    main(args.input, args.dx, args.output)

