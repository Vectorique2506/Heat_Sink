#Physics Engine of the project:

import numpy as np

# ── Physical constants ──────────────────────────────────────────────
MATERIALS = {
    "aluminium": {"k": 205, "name": "Aluminium"},
    "copper":    {"k": 385, "name": "Copper"},
}

FIN_CONFIGS = {
    "rectangular": {"efficiency": 0.82, "area_factor": 1.0},
    "triangular":  {"efficiency": 0.74, "area_factor": 0.85},
    "pin":         {"efficiency": 0.88, "area_factor": 1.3},
}

H_CONV      = 25      # convection coefficient W/m²K
T_AMBIENT   = 25.0    # ambient air temperature °C
N_FINS      = 8       # number of fins
GRID_ROWS   = 40      # grid resolution (height)
GRID_COLS   = 60      # grid resolution (width)


# ── Core solver ─────────────────────────────────────────────────────
def solve_heat_distribution(
    power_w,
    material="aluminium",
    fin_type="rectangular",
):
    """
    Given CPU power in watts, returns a 2D temperature
    grid T[row, col] representing the heat sink cross-section.

    Row 0    = fin tips  (coolest — exposed to air)
    Row -1   = chip base (hottest — heat source)
    """
    k        = MATERIALS[material]["k"]
    fin_eff  = FIN_CONFIGS[fin_type]["efficiency"]
    area_fac = FIN_CONFIGS[fin_type]["area_factor"]

    rows, cols = GRID_ROWS, GRID_COLS

    # Thermal resistance of this configuration
    R_th = 1.0 / (H_CONV * N_FINS * area_fac * fin_eff * (0.5 + k / 800.0))

    # Base plate temperature (chip contact surface)
    T_base = T_AMBIENT + max(power_w,15.0)* R_th * 200

    # Build the temperature grid using finite differences
    # Each row is linearly interpolated between T_base (bottom) and T_ambient (top)
    # with a nonlinear decay that models real fin cooling behaviour
    T = np.zeros((rows, cols))

    chip_rows = int(rows * 0.15)     # bottom 15% = chip
    base_rows = int(rows * 0.10)     # next 10%   = base plate
    fin_rows  = rows - chip_rows - base_rows   # rest = fins

    fin_w  = max(1, (cols - N_FINS) // N_FINS)  # fin width in grid cells

    for r in range(rows):
        for c in range(cols):

            # ── chip zone ──
            if r >= rows - chip_rows:
                hotspot = 1 - abs(c / cols - 0.5) * 1.5
                hotspot = max(0.3, hotspot)
                T[r, c] = T_base + power_w * 0.003 * hotspot

            # ── base plate zone ──
            elif r >= rows - chip_rows - base_rows:
                depth = (rows - chip_rows - r) / base_rows
                T[r, c] = T_base - depth * (T_base - T_ambient_at_base(T_base)) * 0.2

            # ── fin zone ──
            else:
                fin_idx  = c // (fin_w + 1)
                in_fin   = (c % (fin_w + 1)) < fin_w and fin_idx < N_FINS
                rel_h    = 1.0 - r / fin_rows          # 1 at base, 0 at tip

                if in_fin:
                    # Temperature decays along fin height (Newton cooling)
                    if fin_type == "triangular":
                        decay = rel_h ** 1.4 * fin_eff
                    else:
                        decay = rel_h ** 0.8 * fin_eff
                    T[r, c] = T_AMBIENT + (T_base - T_AMBIENT) * decay
                else:
                    # Air gap between fins — cooler
                    T[r, c] = T_AMBIENT + (T_base - T_AMBIENT) * rel_h * 0.25

    return T


def T_ambient_at_base(T_base):
    return T_AMBIENT + (T_base - T_AMBIENT) * 0.15


# ── Derived metrics ─────────────────────────────────────────────────
def compute_metrics(T, power_w, material="aluminium", fin_type="rectangular"):
    """
    Returns key thermal performance metrics as a dict.
    """
    k       = MATERIALS[material]["k"]
    fin_eff = FIN_CONFIGS[fin_type]["efficiency"]

    T_max  = round(float(np.max(T)), 1)
    T_min  = round(float(np.min(T)), 1)
    R_th   = round((T_max - T_AMBIENT) / max(power_w, 0.1), 3)
    safe   = T_max < 85.0

    return {
        "T_max_c":          T_max,
        "T_min_c":          T_min,
        "thermal_resistance": R_th,
        "fin_efficiency_pct": round(fin_eff * 100),
        "material":         MATERIALS[material]["name"],
        "fin_type":         fin_type,
        "is_safe":          safe,
        "warning":          None if safe else f"DANGER: {T_max}°C exceeds 85°C limit!",
    }


# ── Quick standalone test ────────────────────────────────────────────
if __name__ == "__main__":
    test_power = 65.0   # watts — typical MacBook load

    print("=" * 50)
    print(f"Simulating heat sink at {test_power}W\n")

    for mat in ["aluminium", "copper"]:
        for fin in ["rectangular", "triangular", "pin"]:
            T    = solve_heat_distribution(test_power, mat, fin)
            m    = compute_metrics(T, test_power, mat, fin)
            flag = "OK" if m["is_safe"] else "HOT"
            print(f"[{flag}] {m['material']:12s} + {fin:12s} "
                  f"→ Tmax: {m['T_max_c']}°C  "
                  f"R_th: {m['thermal_resistance']} °C/W  "
                  f"Fin eff: {m['fin_efficiency_pct']}%")

    print("\nGrid shape:", T.shape)
    print("Sample T[0,0] (fin tip):", round(T[0, 0], 2), "°C")
    print("Sample T[-1,30] (chip center):", round(T[-1, 30], 2), "°C")
    
    