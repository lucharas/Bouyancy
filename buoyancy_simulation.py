import math
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# --- Constants ---
A = 10.0      # Length/Width (a)
B = 5.0       # Height (b)
AREA_TARGET = 30.0 # Target submerged area

# Simulation parameters
ANGLE_START_DEG = 0.0
ANGLE_END_DEG = 90.0
ANGLE_STEP_DEG = 5.0

TOLERANCE_AREA = 1e-6
MAX_ITERATIONS = 100

def get_rect_vertices(a, b):
    """
    Returns vertices of the rectangle centered at (0,0).
    Order: Bottom-Left, Bottom-Right, Top-Right, Top-Left (Standard counter-clockwise)
    Coordinates: x in [-a/2, a/2], y in [-b/2, b/2]
    """
    w = a / 2.0
    h = b / 2.0
    return [
        (-w, -h),
        (w, -h),
        (w, h),
        (-w, h)
    ]

def polygon_area(vertices):
    """Calculates the area of a non-self-intersecting polygon."""
    n = len(vertices)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return 0.5 * abs(area)

def polygon_centroid(vertices):
    """Calculates the centroid (Cx, Cy) of a polygon."""
    n = len(vertices)
    if n < 3:
        return (0.0, 0.0) # Should not happen for valid polygons

    area = 0.0
    cx = 0.0
    cy = 0.0

    # Calculate signed area for centroid formula
    signed_area = 0.0

    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        cross = x1 * y2 - x2 * y1
        signed_area += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross

    signed_area *= 0.5
    if abs(signed_area) < 1e-9:
        return (0.0, 0.0)

    cx /= (6 * signed_area)
    cy /= (6 * signed_area)

    return (cx, cy)

def clip_polygon(vertices, nx, ny, c):
    """
    Clips a polygon by the half-plane nx*x + ny*y <= c.
    Uses Sutherland-Hodgman algorithm.
    """
    def is_inside(x, y):
        return nx * x + ny * y <= c

    def intersection(x1, y1, x2, y2):
        # Line p1-p2 intersects line nx*x + ny*y = c
        # P(t) = P1 + t*(P2 - P1)
        # n dot (P1 + t*(P2 - P1)) = c
        # n dot P1 + t * n dot (P2 - P1) = c
        # t = (c - n dot P1) / (n dot (P2 - P1))

        dot_p1 = nx * x1 + ny * y1
        dot_diff = nx * (x2 - x1) + ny * (y2 - y1)

        if abs(dot_diff) < 1e-9:
            return (x1, y1) # Should not happen if endpoints are on different sides

        t = (c - dot_p1) / dot_diff
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    output_list = vertices

    # In this specific case, we only have one clipping plane (the waterline).
    # So we just run one pass of Sutherland-Hodgman.

    input_list = output_list
    output_list = []

    if not input_list:
        return []

    s = input_list[-1]
    for e in input_list:
        if is_inside(e[0], e[1]):
            if not is_inside(s[0], s[1]):
                output_list.append(intersection(s[0], s[1], e[0], e[1]))
            output_list.append(e)
        elif is_inside(s[0], s[1]):
            output_list.append(intersection(s[0], s[1], e[0], e[1]))
        s = e

    return output_list

def calculate_submerged_properties(theta_rad, c, rect_vertices):
    # Normal vector for: y * cos(theta) - x * sin(theta) <= c
    # -sin(theta) * x + cos(theta) * y <= c
    nx = -math.sin(theta_rad)
    ny = math.cos(theta_rad)

    clipped_poly = clip_polygon(rect_vertices, nx, ny, c)
    area = polygon_area(clipped_poly)
    centroid = polygon_centroid(clipped_poly)

    return area, centroid

def solve_c(theta_rad, target_area, rect_vertices):
    # Determine bounds for c
    nx = -math.sin(theta_rad)
    ny = math.cos(theta_rad)

    # Project all vertices onto normal to find min/max c
    projections = [nx * v[0] + ny * v[1] for v in rect_vertices]
    c_min = min(projections)
    c_max = max(projections)

    # Check if target area is achievable (it should be if 0 < target < total)
    total_area = A * B
    if target_area <= 0:
        return c_min - 0.1 # Fully outside
    if target_area >= total_area:
        return c_max + 0.1 # Fully inside

    # Bisection
    low = c_min
    high = c_max

    for _ in range(MAX_ITERATIONS):
        mid = (low + high) / 2.0
        area, _ = calculate_submerged_properties(theta_rad, mid, rect_vertices)

        if abs(area - target_area) < TOLERANCE_AREA:
            return mid

        if area < target_area:
            low = mid
        else:
            high = mid

    return (low + high) / 2.0

def visualize_results(rect_w, rect_h, results_list):
    """
    Visualizes the rectangle and the path of the Center of Buoyancy.
    """
    if not HAS_MATPLOTLIB:
        print("\n[INFO] matplotlib not installed. Visualization skipped.")
        return

    # Unpack results
    # results_list contains tuples: (angle_deg, c, cb_x, cb_y, area)
    cb_x_vals = [r[2] for r in results_list]
    cb_y_vals = [r[3] for r in results_list]
    angles = [r[0] for r in results_list]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Draw Rectangle
    # Coordinates of bottom-left corner
    bl_x = -rect_w / 2.0
    bl_y = -rect_h / 2.0
    rect_patch = patches.Rectangle((bl_x, bl_y), rect_w, rect_h,
                                   linewidth=2, edgecolor='black', facecolor='none', label='Hull Cross-section')
    ax.add_patch(rect_patch)

    # Scatter plot of CB points
    scatter = ax.scatter(cb_x_vals, cb_y_vals, c=angles, cmap='viridis', label='Center of Buoyancy')
    plt.colorbar(scatter, label='Heel Angle (deg)')

    # Plot path line
    ax.plot(cb_x_vals, cb_y_vals, linestyle='--', color='gray', alpha=0.5)

    # Mark the start and end
    ax.text(cb_x_vals[0], cb_y_vals[0], f'{angles[0]:.0f}°', fontsize=9, verticalalignment='bottom')
    ax.text(cb_x_vals[-1], cb_y_vals[-1], f'{angles[-1]:.0f}°', fontsize=9, verticalalignment='bottom')

    # Add geometric center of the rectangle
    ax.plot(0, 0, 'rx', label='Geometric Center (0,0)')

    ax.set_title(f'Trajectory of Center of Buoyancy (CB)\nConstant Area = {AREA_TARGET}')
    ax.set_xlabel('x (width)')
    ax.set_ylabel('y (height)')
    ax.axhline(0, color='black', linewidth=0.5, linestyle=':')
    ax.axvline(0, color='black', linewidth=0.5, linestyle=':')

    # Set equal aspect ratio to ensure rectangle isn't distorted
    ax.set_aspect('equal', adjustable='box')

    # Set limits with some padding
    margin = 1.0
    ax.set_xlim(-rect_w/2 - margin, rect_w/2 + margin)
    ax.set_ylim(-rect_h/2 - margin, rect_h/2 + margin)

    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    print("\n[INFO] Displaying plot...")
    plt.show()

def main():
    print(f"Simulation Parameters:")
    print(f"Rectangle: {A} x {B}")
    print(f"Target Area: {AREA_TARGET}")
    print(f"Angle Range: [{ANGLE_START_DEG}, {ANGLE_END_DEG}] step {ANGLE_STEP_DEG}")
    print("-" * 60)
    print(f"{'Theta(deg)':<12} | {'c':<12} | {'CB_x':<12} | {'CB_y':<12} | {'Calc Area':<12}")
    print("-" * 60)

    rect_vertices = get_rect_vertices(A, B)

    results = []

    # Generate angles
    # Using integer loop to avoid floating point accumulation errors in step
    num_steps = int((ANGLE_END_DEG - ANGLE_START_DEG) / ANGLE_STEP_DEG) + 1

    for i in range(num_steps):
        angle_deg = ANGLE_START_DEG + i * ANGLE_STEP_DEG
        if angle_deg > ANGLE_END_DEG + 1e-9:
            break

        theta_rad = math.radians(angle_deg)

        c_opt = solve_c(theta_rad, AREA_TARGET, rect_vertices)

        final_area, cb = calculate_submerged_properties(theta_rad, c_opt, rect_vertices)

        print(f"{angle_deg:<12.1f} | {c_opt:<12.4f} | {cb[0]:<12.4f} | {cb[1]:<12.4f} | {final_area:<12.4f}")
        results.append((angle_deg, c_opt, cb[0], cb[1], final_area))

    visualize_results(A, B, results)

if __name__ == "__main__":
    main()
