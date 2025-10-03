import numpy as np
from numba import cuda
import math
import pyvista as pv

@cuda.jit
def kernel_vertex_facet_detection(positions, facet_indices, r, rq, FOGC, dmin_v, dmin_t, M_cloth):
    v_id = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    f_id = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    N = positions.shape[0]
    M = facet_indices.shape[0]
    if v_id >= N or f_id >= M:
        return
    vx = positions[v_id, 0]
    vy = positions[v_id, 1]
    vz = positions[v_id, 2]
    v0 = facet_indices[f_id, 0]
    v1 = facet_indices[f_id, 1]
    v2 = facet_indices[f_id, 2]
    x0 = positions[v0, 0]; y0 = positions[v0, 1]; z0 = positions[v0, 2]
    x1 = positions[v1, 0]; y1 = positions[v1, 1]; z1 = positions[v1, 2]
    x2 = positions[v2, 0]; y2 = positions[v2, 1]; z2 = positions[v2, 2]
    cx = (x0 + x1 + x2) / 3.0
    cy = (y0 + y1 + y2) / 3.0
    cz = (z0 + z1 + z2) / 3.0
    dx = vx - cx
    dy = vy - cy
    dz = vz - cz
    dist = math.sqrt(dx*dx + dy*dy + dz*dz)
    if f_id >= M_cloth:
        cuda.atomic.min(dmin_v, v_id, dist)
        cuda.atomic.min(dmin_t, f_id, dist)
    if dist < r:
        FOGC[v_id, f_id] = 1.0

@cuda.jit
def kernel_edge_edge_detection(positions, edge_indices, r, rq, EOGC, dmin_e, E_cloth):
    e_id = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    e2_id = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    E = edge_indices.shape[0]
    if e_id >= E or e2_id >= E:
        return
    if e_id == e2_id:
        return
    v0 = edge_indices[e_id, 0]
    v1 = edge_indices[e_id, 1]
    x1 = positions[v0, 0]; y1 = positions[v0, 1]; z1 = positions[v0, 2]
    x2 = positions[v1, 0]; y2 = positions[v1, 1]; z2 = positions[v1, 2]
    mx1 = 0.5 * (x1 + x2); my1 = 0.5 * (y1 + y2); mz1 = 0.5 * (z1 + z2)
    u0 = edge_indices[e2_id, 0]
    u1 = edge_indices[e2_id, 1]
    ux1 = positions[u0, 0]; uy1 = positions[u0, 1]; uz1 = positions[u0, 2]
    ux2 = positions[u1, 0]; uy2 = positions[u1, 1]; uz2 = positions[u1, 2]
    mx2 = 0.5 * (ux1 + ux2); my2 = 0.5 * (uy1 + uy2); mz2 = 0.5 * (uz1 + uz2)
    dx = mx1 - mx2; dy = my1 - my2; dz = mz1 - mz2
    dist = math.sqrt(dx*dx + dy*dy + dz*dz)
    if e2_id >= E_cloth:
        cuda.atomic.min(dmin_e, e_id, dist)
    if dist < r:
        EOGC[e_id, e2_id] = 1.0

@cuda.jit
def kernel_compute_conservative(dmin_v, gamma_p, b_v):
    v_id = cuda.grid(1)
    if v_id >= dmin_v.shape[0]:
        return
    b_v[v_id] = gamma_p * dmin_v[v_id]

@cuda.jit
def kernel_vbd_iteration(X, Y, mass, ext_forces, X_out):
    v_id = cuda.grid(1)
    N = X.shape[0]
    if v_id >= N:
        return
    xv = X[v_id, 0]; yv = X[v_id, 1]; zv = X[v_id, 2]
    yx = Y[v_id, 0]; yy = Y[v_id, 1]; yz = Y[v_id, 2]
    m = mass[v_id]
    fx = -m * (xv - yx) + ext_forces[v_id, 0]
    fy = -m * (yv - yy) + ext_forces[v_id, 1]
    fz = -m * (zv - yz) + ext_forces[v_id, 2]
    X_out[v_id, 0] = xv + fx / (m + 1e-12)
    X_out[v_id, 1] = yv + fy / (m + 1e-12)
    X_out[v_id, 2] = zv + fz / (m + 1e-12)

@cuda.jit
def kernel_truncate_displacements(X, Xprev, b_v, moved_flags):
    v_id = cuda.grid(1)
    N = X.shape[0]
    if v_id >= N:
        return
    dx = X[v_id, 0] - Xprev[v_id, 0]
    dy = X[v_id, 1] - Xprev[v_id, 1]
    dz = X[v_id, 2] - Xprev[v_id, 2]
    dist = math.sqrt(dx*dx + dy*dy + dz*dz)
    bv = b_v[v_id]
    if dist > bv and dist > 1e-12:
        s = bv / dist
        X[v_id, 0] = Xprev[v_id, 0] + dx * s
        X[v_id, 1] = Xprev[v_id, 1] + dy * s
        X[v_id, 2] = Xprev[v_id, 2] + dz * s
        moved_flags[v_id] = 1
    else:
        moved_flags[v_id] = 0

@cuda.jit
def kernel_project_springs(X, edge_indices, rest_lengths, k, X_out):
    e_id = cuda.grid(1)
    if e_id >= edge_indices.shape[0]:
        return
    v0 = edge_indices[e_id, 0]
    v1 = edge_indices[e_id, 1]
    p0x = X[v0, 0]; p0y = X[v0, 1]; p0z = X[v0, 2]
    p1x = X[v1, 0]; p1y = X[v1, 1]; p1z = X[v1, 2]
    dx = p1x - p0x; dy = p1y - p0y; dz = p1z - p0z
    dist = math.sqrt(dx*dx + dy*dy + dz*dz)
    if dist < 1e-12:
        return
    rest = rest_lengths[e_id]
    factor = k * (dist - rest) / dist / 2.0
    cx = dx * factor
    cy = dy * factor
    cz = dz * factor
    cuda.atomic.add(X_out, (v0, 0), cx)
    cuda.atomic.add(X_out, (v0, 1), cy)
    cuda.atomic.add(X_out, (v0, 2), cz)
    cuda.atomic.sub(X_out, (v1, 0), cx)
    cuda.atomic.sub(X_out, (v1, 1), cy)
    cuda.atomic.sub(X_out, (v1, 2), cz)

def simulation_step(cloth_X, facet_indices, edge_indices, rest_lengths, Xprev, Y, mass, ext_forces,
                    r=0.01, rq=0.05, gamma_p=0.45, niter=10, threads_per_block=1024):
    N_cloth = cloth_X.shape[0]
    M_cloth = facet_indices.shape[0]
    E_cloth = edge_indices.shape[0]
    combined_facet_indices = facet_indices.astype(np.int32)
    combined_edge_indices = edge_indices.astype(np.int32)
    M = combined_facet_indices.shape[0]
    E = combined_edge_indices.shape[0]
    positions = cloth_X.copy()
    d_positions = cuda.to_device(positions)
    d_Xprev = cuda.to_device(Xprev)
    d_Y = cuda.to_device(Y)
    d_mass = cuda.to_device(mass)
    d_ext_forces = cuda.to_device(ext_forces)
    d_facet_indices = cuda.to_device(combined_facet_indices)
    d_edge_indices = cuda.to_device(combined_edge_indices)
    d_rest_lengths = cuda.to_device(rest_lengths)
    d_FOGC = cuda.device_array((N_cloth, M), dtype=np.float32)
    d_EOGC = cuda.device_array((E_cloth, E), dtype=np.float32)
    d_dmin_v = cuda.to_device(np.full(N_cloth, 1e10, dtype=np.float32))
    d_dmin_t = cuda.to_device(np.full(M, 1e10, dtype=np.float32))
    d_dmin_e = cuda.to_device(np.full(E, 1e10, dtype=np.float32))
    d_b_v = cuda.device_array(N_cloth, dtype=np.float32)
    d_X_out = cuda.device_array_like(d_Y)
    d_X_temp = cuda.device_array_like(d_Y)
    d_moved_flags = cuda.device_array(N_cloth, dtype=np.int32)
    blocks_v = (N_cloth + threads_per_block - 1) // threads_per_block
    blocks_e = (E_cloth + threads_per_block - 1) // threads_per_block
    block_dim_2d = (32, 32)
    grid_dim_vf = ((N_cloth + block_dim_2d[0] - 1) // block_dim_2d[0],
                   (M + block_dim_2d[1] - 1) // block_dim_2d[1])
    grid_dim_ee = ((E_cloth + block_dim_2d[0] - 1) // block_dim_2d[0],
                   (E + block_dim_2d[1] - 1) // block_dim_2d[1])
    collisionDetectionRequired = True
    for _ in range(niter):
        if collisionDetectionRequired:
            kernel_vertex_facet_detection[grid_dim_vf, block_dim_2d](
                d_positions, d_facet_indices, r, rq, d_FOGC, d_dmin_v, d_dmin_t, M_cloth)
            cuda.synchronize()
            kernel_edge_edge_detection[grid_dim_ee, block_dim_2d](
                d_positions, d_edge_indices, r, rq, d_EOGC, d_dmin_e, E_cloth)
            cuda.synchronize()
            kernel_compute_conservative[blocks_v, threads_per_block](d_dmin_v, gamma_p, d_b_v)
            cuda.synchronize()
            d_Xprev.copy_to_device(d_positions[:N_cloth])
            collisionDetectionRequired = False
        kernel_vbd_iteration[blocks_v, threads_per_block](d_positions[:N_cloth], d_Y, d_mass, d_ext_forces, d_X_out)
        cuda.synchronize()
        d_X_temp.copy_to_device(d_X_out)
        kernel_project_springs[blocks_e, threads_per_block](d_X_out, d_edge_indices, d_rest_lengths, 0.5, d_X_temp)
        cuda.synchronize()
        d_X_out.copy_to_device(d_X_temp)
        kernel_truncate_displacements[blocks_v, threads_per_block](d_X_out, d_Xprev, d_b_v, d_moved_flags)
        cuda.synchronize()
        moved_flags = d_moved_flags.copy_to_host()
        if np.sum(moved_flags) > 0.01 * N_cloth:
            collisionDetectionRequired = True
        d_positions[:N_cloth].copy_to_device(d_X_out)
    return d_positions[:N_cloth].copy_to_host()

def create_cloth(nx=40, ny=40, size=2.0):
    nverts = (nx + 1) * (ny + 1)
    vertices = np.zeros((nverts, 3), dtype=np.float32)
    dx = size / nx
    dy = size / ny
    k = 0
    for j in range(ny + 1):
        for i in range(nx + 1):
            vertices[k] = [i * dx - size / 2, j * dy - size / 2, 0.0]
            k += 1
    nfacets = nx * ny * 2
    facet_indices = np.zeros((nfacets, 3), dtype=np.int32)
    k = 0
    for j in range(ny):
        for i in range(nx):
            v00 = i + j * (nx + 1)
            v10 = (i + 1) + j * (nx + 1)
            v01 = i + (j + 1) * (nx + 1)
            v11 = (i + 1) + (j + 1) * (nx + 1)
            facet_indices[k] = [v00, v10, v01]
            k += 1
            facet_indices[k] = [v01, v10, v11]
            k += 1
    edge_set = set()
    for f in facet_indices:
        for pair in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])]:
            a, b = sorted(pair)
            edge_set.add((a, b))
    edge_indices = np.array(list(edge_set), dtype=np.int32)
    rest_lengths = np.zeros(len(edge_indices), dtype=np.float32)
    for ei, e in enumerate(edge_indices):
        p0 = vertices[e[0]]
        p1 = vertices[e[1]]
        rest_lengths[ei] = np.linalg.norm(p1 - p0)
    return vertices, facet_indices, edge_indices, rest_lengths

if __name__ == "__main__":
    vertices, facet_indices, edge_indices, rest_lengths = create_cloth(40, 40, 2.0)
    X = vertices.copy()
    V = np.zeros_like(X)
    mass = np.ones((len(vertices),), dtype=np.float32) * (1.0 / len(vertices))
    h = 0.02
    g = -0.5
    ext_forces = np.zeros_like(X)
    ext_forces[:, 2] = mass * g * h**2

    nx = 40
    ny = 40
    fixed = list(range(ny * (nx + 1), (ny + 1) * (nx + 1)))
    fixed_pos = vertices[fixed].copy()

    faces = np.hstack([np.full((len(facet_indices), 1), 3, dtype=np.int32), facet_indices]).ravel()
    mesh = pv.PolyData(X, faces)

    plotter = pv.Plotter(title="CudaAlgo")
    plotter.add_mesh(mesh, show_edges=True, color='white', opacity=0.8)

    step_actor = plotter.add_text("Step: 0", position='upper_right', font_size=12, color='black')
    step = 0
    step_in_progress = False

    def step_sim():
        global X, V, step, step_in_progress
        if step_in_progress:
            return  # Skip if a step is already in progress
        step_in_progress = True
        try:
            Y = X + V * h
            Xprev = X.copy()
            X_new = simulation_step(X, facet_indices, edge_indices, rest_lengths, Xprev, Y, mass, ext_forces)
            X_new[fixed] = fixed_pos
            V = (X_new - X) / h
            X[:] = X_new
            mesh.points[:] = X
            step += 1
            step_actor.SetText(3, f"Step: {step}")
            plotter.update()
        finally:
            step_in_progress = False  # Ensure flag is cleared even if an error occurs

    plotter.add_key_event('s', step_sim)  # Single step on 's' press, locked until complete
    plotter.show()