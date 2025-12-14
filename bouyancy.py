import math
import csv

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# --- Constants ---
# User Coordinate System:
# x: Longitudinal (ignored in 2D)
# y: Transverse (Width/Beam)
# z: Vertical (Height/Depth)

A = 24.0      # Hull Beam (Width, User axis 'y')
B = 5.0       # Hull Depth (Height, User axis 'z')
H = 0.0       # Distance between Geometric Center (GC) and Center of Gravity (CG).
              # CG is the reference point (0,0).
              # We assume GC is located at (0, H) in the (y, z) plane.
              # If H > 0, GC is above CG.

KELL_HGHT = 0.0 # 'kell_hght' as requested for CSV metadata.
                # Interpreted as a parameter associated with keel height or position.

AREA_TARGET = 5.0 # Target submerged area

# Simulation parameters
ANGLE_START_DEG = 0.0
ANGLE_END_DEG = 90.0
ANGLE_STEP_DEG = 1.0

# Numerical parameters
TOLERANCE_AREA = 1e-6
MAX_ITERATIONS = 100

def get_rect_vertices(width, height, v_offset):
    """
    Returns vertices of the rectangle representing the hull cross-section.
    The coordinate system is:
      Internal x -> User y (Transverse)
      Internal y -> User z (Vertical)

    The Geometric Center (GC) of the hull is at (0, v_offset).
    The Center of Gravity (CG) is at (0, 0).

    Vertices are returned in counter-clockwise order.
    """
    w = width / 2.0
    h = height / 2.0

    # Vertices relative to (0,0) which is CG.
    # GC is at (0, v_offset).
    return [
        (-w, v_offset - h), # Bottom-Left
        (w, v_offset - h),  # Bottom-Right
        (w, v_offset + h),  # Top-Right
        (-w, v_offset + h)  # Top-Left
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
        return (0.0, 0.0)

    area = 0.0
    cx = 0.0
    cy = 0.0

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
        dot_p1 = nx * x1 + ny * y1
        dot_diff = nx * (x2 - x1) + ny * (y2 - y1)

        if abs(dot_diff) < 1e-9:
            return (x1, y1)

        t = (c - dot_p1) / dot_diff
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    input_list = vertices
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
    """
    Calculates area, centroid (CB), and waterline length (hull_wet) for a given cut.
    """
    # Normal vector for: y * cos(theta) - x * sin(theta) <= c
    # Note: Using internal coordinates (x, y) which correspond to user (y, z).
    nx = -math.sin(theta_rad)
    ny = math.cos(theta_rad)

    clipped_poly = clip_polygon(rect_vertices, nx, ny, c)
    area = polygon_area(clipped_poly)
    centroid = polygon_centroid(clipped_poly)

    # Calculate hull_wet (waterline length)
    # This is the length of the segment(s) of the clipping line that bound the submerged polygon.
    hull_wet = 0.0
    n = len(clipped_poly)
    if n >= 2:
        for i in range(n):
            p1 = clipped_poly[i]
            p2 = clipped_poly[(i + 1) % n]

            # Check if edge lies on the clipping line nx*x + ny*y = c
            val1 = nx * p1[0] + ny * p1[1] - c
            val2 = nx * p2[0] + ny * p2[1] - c

            if abs(val1) < 1e-7 and abs(val2) < 1e-7:
                dist = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                hull_wet += dist

    return area, centroid, hull_wet

def solve_c(theta_rad, target_area, rect_vertices):
    """
    Finds the line constant 'c' such that the submerged area equals target_area.
    """
    nx = -math.sin(theta_rad)
    ny = math.cos(theta_rad)

    projections = [nx * v[0] + ny * v[1] for v in rect_vertices]
    c_min = min(projections)
    c_max = max(projections)

    total_area = A * B
    if target_area <= 0:
        return c_min - 0.1
    if target_area >= total_area:
        return c_max + 0.1

    low = c_min
    high = c_max

    for _ in range(MAX_ITERATIONS):
        mid = (low + high) / 2.0
        area, _, _ = calculate_submerged_properties(theta_rad, mid, rect_vertices)

        if abs(area - target_area) < TOLERANCE_AREA:
            return mid

        if area < target_area:
            low = mid
        else:
            high = mid

    return (low + high) / 2.0

def save_csv_data(filename, results, metadata):
    try:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write Metadata
            # Format: Key, Value
            for key, value in metadata.items():
                writer.writerow([key, value])

            # Write Header
            writer.writerow(['angle', 'CB_y', 'CB_z', 'hull_wet'])

            # Write Data
            for row in results:
                # row structure: (angle_deg, c, cb_y, cb_z, area, hull_wet)
                writer.writerow([
                    f"{row[0]:.2f}",
                    f"{row[2]:.4f}",
                    f"{row[3]:.4f}",
                    f"{row[5]:.4f}"
                ])
        print(f"[INFO] Data saved to {filename}")
    except IOError as e:
        print(f"[ERROR] Could not write to CSV: {e}")

def visualize_results(rect_w, rect_h, h_offset, results_list):
    if not HAS_MATPLOTLIB:
        print("\n[INFO] matplotlib not installed. Visualization skipped.")
        return

    cb_y_vals = [r[2] for r in results_list]
    cb_z_vals = [r[3] for r in results_list]
    angles = [r[0] for r in results_list]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Draw Hull (Rectangle)
    # GC is at (0, h_offset).
    # Bottom-left: (-w/2, h_offset - h/2)
    bl_y = -rect_w / 2.0
    bl_z = h_offset - rect_h / 2.0

    rect_patch = patches.Rectangle((bl_y, bl_z), rect_w, rect_h,
                                   linewidth=2, edgecolor='black', facecolor='none', label='Hull Cross-section')
    ax.add_patch(rect_patch)

    # Scatter plot of CB points
    # Internal x -> y, Internal y -> z
    scatter = ax.scatter(cb_y_vals, cb_z_vals, s=5, c=angles, cmap='viridis', label='Center of Buoyancy')
    plt.colorbar(scatter, label='Heel Angle (deg)')

    # Plot path
    ax.plot(cb_y_vals, cb_z_vals, linestyle='--', color='gray', alpha=0.5)

    # Start/End labels
    ax.text(cb_y_vals[0], cb_z_vals[0], f'{angles[0]:.0f}°', fontsize=9, verticalalignment='bottom')
    ax.text(cb_y_vals[-1], cb_z_vals[-1], f'{angles[-1]:.0f}°', fontsize=9, verticalalignment='bottom')

    # CG (0,0)
    ax.plot(0, 0, 'rx', label='CG (0,0)')

    # GC (0, H)
    ax.plot(0, h_offset, 'bx', label=f'GC (0,{h_offset})')

    ax.set_title(f'Trajectory of Center of Buoyancy (CB)\nConstant Area = {AREA_TARGET}')
    ax.set_xlabel('y (transverse / width)')
    ax.set_ylabel('z (vertical / height)')
    ax.axhline(0, color='black', linewidth=0.5, linestyle=':')
    ax.axvline(0, color='black', linewidth=0.5, linestyle=':')

    ax.set_aspect('equal', adjustable='box')

    margin = 1.0
    ax.set_xlim(-rect_w/2 - margin, rect_w/2 + margin)

    # Adjust Y limits (user Z) to encompass hull and CG
    z_min = min(bl_z, 0) - margin
    z_max = max(bl_z + rect_h, 0) + margin
    ax.set_ylim(z_min, z_max)

    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    print("\n[INFO] Displaying plot...")
    plt.show()

def main():
    print(f"Simulation Parameters:")
    print(f"Hull: Beam(A)={A}, Depth(B)={B}")
    print(f"Offset H (GC above CG)={H}")
    print(f"Target Area: {AREA_TARGET}")
    print(f"Angle Range: [{ANGLE_START_DEG}, {ANGLE_END_DEG}]")
    print("-" * 75)
    print(f"{'Theta(deg)':<10} | {'c':<10} | {'CB_y':<10} | {'CB_z':<10} | {'hull_wet':<10} | {'Area':<10}")
    print("-" * 75)

    # Calculate vertices with CG at (0,0)
    rect_vertices = get_rect_vertices(A, B, H)

    results = []

    # Generate angles with higher resolution for < 10 degrees
    angles_to_sim = []
    curr = ANGLE_START_DEG
    while curr <= ANGLE_END_DEG + 1e-9:
        angles_to_sim.append(curr)

        step = ANGLE_STEP_DEG
        if curr < 10.0 - 1e-9:
            step = ANGLE_STEP_DEG / 10.0

        curr = round(curr + step, 4)

    for angle_deg in angles_to_sim:
        if angle_deg > ANGLE_END_DEG + 1e-9:
            break

        theta_rad = math.radians(angle_deg)

        c_opt = solve_c(theta_rad, AREA_TARGET, rect_vertices)

        final_area, cb, hull_wet = calculate_submerged_properties(theta_rad, c_opt, rect_vertices)

        # cb is (x, y) in internal coords -> (y, z) in user coords
        cb_y = cb[0]
        cb_z = cb[1]

        print(f"{angle_deg:<10.2f} | {c_opt:<10.4f} | {cb_y:<10.4f} | {cb_z:<10.4f} | {hull_wet:<10.4f} | {final_area:<10.4f}")
        results.append((angle_deg, c_opt, cb_y, cb_z, final_area, hull_wet))

    # Save to CSV
    metadata = {
        'hull_beam': A,
        'hull_depth': B,
        'kell_hght': KELL_HGHT,
        'H_GC_from_CG': H
    }
    save_csv_data('buoyancy_data.csv', results, metadata)

    visualize_results(A, B, H, results)

if __name__ == "__main__":
    main()
