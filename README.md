# super-ellipsoids

## Installation

The main script requires a Python environment with certain packages. It is recommended to setup a
new Python virtual environment (to avoid contaminating an existing one), and install the
requirements using `pip install -r requirements.txt`.

## Usage

The Python script offers a basic user manual via `python voxelize-super-ellipsoids.py --help`.
However, here is an example command to use the script:

```
python voxelize-super-ellipsoids.py --input particles.csv --dx 0.00002 --output super-ellipsoids-voxelized.json
```

The CSV input file must have the following columns in the following order:

```
rx,ry,rz,eps1,eps2,x,y,z,q0,q1,q2,q3
```
