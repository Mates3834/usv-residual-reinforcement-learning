import heapq
import math
import numpy as np


def _heuristic(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def hybrid_astar(
    start,
    goal,
    obstacles=None,
    resolution=2.0,
    bounds=(-5.0, 50.0, -5.0, 40.0),
):
    """
    Lightweight Hybrid-A*-style planner.

    The state contains (x, y, heading_index). Motion primitives preserve a
    coarse heading state, making this more vehicle-oriented than a plain 2-D
    grid search while remaining compact and easy to understand.

    Returns:
        path: ndarray with shape (N, 2)
    """

    if obstacles is None:
        obstacles = []

    xmin, xmax, ymin, ymax = bounds
    headings = np.linspace(-np.pi, np.pi, 16, endpoint=False)

    def quantize(x, y, psi):
        ix = int(round((x - xmin) / resolution))
        iy = int(round((y - ymin) / resolution))
        ih = int(np.argmin(np.abs(np.angle(np.exp(1j * (headings - psi))))))
        return ix, iy, ih

    def dequantize(node):
        ix, iy, ih = node
        x = xmin + ix * resolution
        y = ymin + iy * resolution
        return x, y, headings[ih]

    def valid(x, y):
        if not (xmin <= x <= xmax and ymin <= y <= ymax):
            return False
        for ox, oy, radius in obstacles:
            if math.hypot(x - ox, y - oy) <= radius + 1.0:
                return False
        return True

    sx, sy, spsi = start
    gx, gy = goal

    start_node = quantize(sx, sy, spsi)
    open_heap = [(0.0, start_node)]
    came_from = {}
    g_cost = {start_node: 0.0}
    closed = set()

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)

        x, y, psi = dequantize(current)

        if math.hypot(x - gx, y - gy) <= resolution * 1.5:
            nodes = [current]
            while nodes[-1] in came_from:
                nodes.append(came_from[nodes[-1]])
            nodes.reverse()
            path = np.array([[dequantize(n)[0], dequantize(n)[1]] for n in nodes])
            return path

        for steering_delta in (-1, 0, 1):
            ih_next = (current[2] + steering_delta) % len(headings)
            psi_next = headings[ih_next]
            nx = x + resolution * math.cos(psi_next)
            ny = y + resolution * math.sin(psi_next)

            if not valid(nx, ny):
                continue

            nxt = quantize(nx, ny, psi_next)
            turn_penalty = 0.25 * abs(steering_delta)
            tentative = g_cost[current] + resolution + turn_penalty

            if tentative < g_cost.get(nxt, float("inf")):
                came_from[nxt] = current
                g_cost[nxt] = tentative
                priority = tentative + _heuristic((nx, ny), (gx, gy))
                heapq.heappush(open_heap, (priority, nxt))

    raise RuntimeError("No feasible path found.")
