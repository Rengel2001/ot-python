"""
Exact semi-discrete OT in 2D via Newton's method.

Takes the 2D UMAP embedding and computes the optimal height vector h
using exact cell area computation (convex hull + polygon clipping).

Uses the quadratic power diagram formulation throughout:
  W_i = {x : ||x - p_i||^2 - h_i <= ||x - p_j||^2 - h_j, for all j}
which is equivalent to:
  W_i = {x : 2<x, p_i> - ||p_i||^2 + h_i >= 2<x, p_j> - ||p_j||^2 + h_j}

The lower convex hull of paraboloid-lifted points (p_i, ||p_i||^2 - h_i)
gives the weighted Delaunay triangulation, dual to this power diagram.
Newton's method with exact Hessian (shared boundary lengths) converges
quadratically.

For higher dimensions (Group B), the MC approximation is used instead,
since exact power diagram construction is exponential in dimension.

References:
  - Gu, Luo, Sun, Yau. "Variational Principles for Minkowski Type Problems,
    Discrete Optimal Transport, and Discrete Monge-Ampere Equations" (2013)
  - An, Guo, Lei, Luo, Yau, Gu. "AE-OT" (ICLR 2020)
"""

import logging
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from src.visualization import plot_power_diagram

log = logging.getLogger(__name__)

MAX_ITER = 100
CONVERGENCE_THRESHOLD = 1e-6
LOG_EVERY = 5
SUBSAMPLE_N = 1000


def _clip_polygon(polygon, a, b, c):
    if not polygon:
        return []
    output = []
    n = len(polygon)
    for i in range(n):
        curr = polygon[i]
        nxt = polygon[(i + 1) % n]
        d_curr = a * curr[0] + b * curr[1] - c
        d_nxt = a * nxt[0] + b * nxt[1] - c
        if d_curr >= -1e-12:
            output.append(curr)
            if d_nxt < -1e-12:
                t = d_curr / (d_curr - d_nxt)
                output.append((curr[0] + t * (nxt[0] - curr[0]),
                              curr[1] + t * (nxt[1] - curr[1])))
        elif d_nxt > 1e-12:
            t = d_curr / (d_curr - d_nxt)
            output.append((curr[0] + t * (nxt[0] - curr[0]),
                          curr[1] + t * (nxt[1] - curr[1])))
    return output


def _polygon_area(polygon):
    n = len(polygon)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    return abs(area) / 2.0


def _get_neighbors(P, h):
    N = len(P)
    pnorms = np.sum(P ** 2, axis=1)
    lifted = np.column_stack([P, pnorms - h])

    hull = ConvexHull(lifted, qhull_options="QJ")

    neighbors = {i: set() for i in range(N)}
    for simplex, eq in zip(hull.simplices, hull.equations):
        if eq[2] < 0:  # lower facet
            i, j, k = simplex
            neighbors[i].update([j, k])
            neighbors[j].update([i, k])
            neighbors[k].update([i, j])

    return neighbors


def _cell_halfplane(P, h, pnorms, i, j):
    a = P[i, 0] - P[j, 0]
    b = P[i, 1] - P[j, 1]
    c = (pnorms[i] - pnorms[j] + h[j] - h[i]) / 2.0
    return a, b, c


def _compute_cells(P, h, neighbors, pnorms):
    N = len(P)
    bbox = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    cells = []
    for i in range(N):
        poly = list(bbox)
        for j in neighbors.get(i, set()):
            a, b, c = _cell_halfplane(P, h, pnorms, i, j)
            poly = _clip_polygon(poly, a, b, c)
            if not poly:
                break
        cells.append(poly)
    return cells


def _boundary_length_on_line(polygon, a, b, c):
    if len(polygon) < 3:
        return 0.0
    pts = []
    n = len(polygon)
    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]
        d1 = a * p1[0] + b * p1[1] - c
        d2 = a * p2[0] + b * p2[1] - c
        if abs(d1) < 1e-10:
            pts.append(p1)
        if d1 * d2 < -1e-20:
            t = d1 / (d1 - d2)
            pts.append((p1[0] + t * (p2[0] - p1[0]),
                       p1[1] + t * (p2[1] - p1[1])))
    if len(pts) < 2:
        return 0.0
    max_d2 = 0.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dx = pts[j][0] - pts[i][0]
            dy = pts[j][1] - pts[i][1]
            max_d2 = max(max_d2, dx * dx + dy * dy)
    return np.sqrt(max_d2)


def _build_hessian(P, h, cells, neighbors, pnorms):
    N = len(P)
    H = lil_matrix((N, N))
    done = set()
    for i in range(N):
        for j in neighbors.get(i, set()):
            key = (min(i, j), max(i, j))
            if key in done:
                continue
            done.add(key)
            a = P[i, 0] - P[j, 0]
            b = P[i, 1] - P[j, 1]
            dist = np.sqrt(a * a + b * b)
            if dist < 1e-12:
                continue
            _, _, c = _cell_halfplane(P, h, pnorms, i, j)
            bl = _boundary_length_on_line(cells[i], a, b, c)
            val = bl / (2.0 * dist)
            H[i, j] = val
            H[j, i] = val
    for i in range(N):
        H[i, i] = -H[i, :].sum() - 1e-8
    return H


def _exact_semidiscrete_ot(P):
    N = len(P)
    h = np.zeros(N)
    target = 1.0 / N
    pnorms = np.sum(P ** 2, axis=1)

    start_time = time.time()
    log.info("  Exact OT: N=%d, threshold=%s", N, CONVERGENCE_THRESHOLD)

    for iteration in range(MAX_ITER):
        neighbors = _get_neighbors(P, h)
        cells = _compute_cells(P, h, neighbors, pnorms)
        areas = np.array([_polygon_area(c) for c in cells])

        grad = areas - target
        max_err = np.max(np.abs(grad))
        empty = int(np.sum(areas < 1e-12))
        elapsed = time.time() - start_time

        if (iteration + 1) % LOG_EVERY == 0 or iteration == 0 or max_err < CONVERGENCE_THRESHOLD:
            log.info("  Iter %3d | max_err=%.8f | empty=%d | %.1fs",
                     iteration + 1, max_err, empty, elapsed)

        if max_err < CONVERGENCE_THRESHOLD:
            log.info("  Converged at iteration %d (%.1fs)", iteration + 1, elapsed)
            break

        H = _build_hessian(P, h, cells, neighbors, pnorms)

        # Newton step: fix h[0]=0 to remove translational degree of freedom
        dh = np.zeros(N)
        H_sub = H[1:, 1:].tocsc()
        dh[1:] = spsolve(H_sub, grad[1:])
        dh -= dh.mean()

        # Backtracking line search: require error reduction and no empty cells
        step = 1.0
        for _ in range(30):
            h_trial = h + step * dh
            h_trial -= h_trial.mean()
            nbrs = _get_neighbors(P, h_trial)
            trial_cells = _compute_cells(P, h_trial, nbrs, pnorms)
            trial_areas = np.array([_polygon_area(c) for c in trial_cells])
            trial_err = np.max(np.abs(trial_areas - target))
            if np.all(trial_areas > 1e-14) and trial_err < max_err:
                break
            step *= 0.5
        else:
            log.info("  Line search failed at iter %d, step=%.2e", iteration + 1, step)

        h = h + step * dh
        h -= h.mean()

    return h


def run(Z_umap, y, output_dir):
    z_min = Z_umap.min(axis=0)
    z_max = Z_umap.max(axis=0)
    P_full = (Z_umap - z_min) / (z_max - z_min + 1e-8)

    idx = np.random.choice(len(P_full), SUBSAMPLE_N, replace=False)
    P = P_full[idx]
    y_sub = y[idx]

    log.info("SDOT: Running exact semi-discrete OT (N=%d, subsampled from %d)...",
             len(P), len(P_full))
    h = _exact_semidiscrete_ot(P)

    log.info("SDOT: Rendering power diagram...")
    plot_power_diagram(P, h, y_sub, save_path=output_dir / "sdot-uniform.png")
