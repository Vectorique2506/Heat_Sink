import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.widgets as widgets
import numpy as np
from sensor import get_sensor_data
from solver import solve_heat_distribution, compute_metrics, MATERIALS, FIN_CONFIGS

# ── Simulation state (user-controllable) ─────────────────────────────
state = {
    "material":      "copper",
    "fin_type":      "rectangular",
    "n_fins":        8,
    "h_conv":        25.0,
    "power_override": None,   # None = use real sensor
    "stress_test":   False,
}

INTERVAL     = 1000
MAX_HISTORY  = 60
time_history = []
temp_history = []
tmax_history = []
tick         = [0]

# ── Figure layout ────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 9))
fig.patch.set_facecolor("#0f0f0f")
fig.suptitle("Real-Time CPU Heat Sink Thermal Simulator  —  Interactive",
             color="white", fontsize=13, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig,
                       hspace=0.45, wspace=0.35,
                       left=0.06, right=0.97,
                       top=0.91, bottom=0.30)

ax_heat = fig.add_subplot(gs[:, 0:2])
ax_line = fig.add_subplot(gs[0, 2])
ax_bar  = fig.add_subplot(gs[1, 2])

for ax in [ax_heat, ax_line, ax_bar]:
    ax.set_facecolor("#1a1a1a")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")

# ── Heatmap ──────────────────────────────────────────────────────────
dummy_T  = np.full((40, 60), 25.0)
heatmap  = ax_heat.imshow(dummy_T, cmap="inferno",
                           vmin=25, vmax=100,
                           aspect="auto", origin="upper")
cbar = fig.colorbar(heatmap, ax=ax_heat, fraction=0.03, pad=0.02)
cbar.set_label("Temperature (°C)", color="white", fontsize=9)
cbar.ax.yaxis.set_tick_params(color="white")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

ax_heat.set_title("2D Heat Distribution", color="white", fontsize=11, pad=8)
ax_heat.set_xlabel("Fin width →", color="#aaa", fontsize=8)
ax_heat.set_ylabel("← Fin tip        Chip →", color="#aaa", fontsize=8)
ax_heat.tick_params(colors="#555")

for label, y_pos in [("Fin tips", 2), ("Mid fins", 16),
                      ("Base plate", 33), ("Chip", 38)]:
    ax_heat.text(61, y_pos, label, color="#aaa", fontsize=7, va="center")

txt_info = ax_heat.text(1, 1.5, "", color="white",
                         fontsize=9, fontweight="bold")
txt_warn = ax_heat.text(30, 1.5, "", color="#ff4444",
                         fontsize=9, fontweight="bold", ha="center")

# ── Line chart ───────────────────────────────────────────────────────
ax_line.set_title("Temperature over time", color="white", fontsize=9)
ax_line.set_xlabel("Seconds", color="#aaa", fontsize=7)
ax_line.set_ylabel("°C", color="#aaa", fontsize=7)
ax_line.tick_params(colors="#aaa", labelsize=7)
ax_line.axhline(85, color="#ff4444", linewidth=0.8,
                linestyle="--", alpha=0.7)
line_temp, = ax_line.plot([], [], color="#ff6b35",
                           linewidth=1.5, label="CPU temp")
line_tmax, = ax_line.plot([], [], color="#ffd700",
                           linewidth=1.2, linestyle="--",
                           label="Sim Tmax")
ax_line.legend(fontsize=6, facecolor="#1a1a1a",
               labelcolor="white", loc="upper left")

# ── Bar chart ────────────────────────────────────────────────────────
ax_bar.set_title("Fin geometry comparison", color="white", fontsize=9)
ax_bar.set_ylabel("Tmax (°C)", color="#aaa", fontsize=7)
ax_bar.tick_params(colors="#aaa", labelsize=7)
bar_labels = ["Rectangular", "Triangular", "Pin"]
bar_colors = ["#378ADD", "#EF9F27", "#1D9E75"]
bars       = ax_bar.bar(bar_labels, [25, 25, 25],
                         color=bar_colors, alpha=0.85)
ax_bar.axhline(85, color="#ff4444", linewidth=0.8,
               linestyle="--", alpha=0.7)
ax_bar.set_ylim(20, 110)
bar_texts = [ax_bar.text(i, 27, "", ha="center",
                          color="white", fontsize=7, fontweight="bold")
             for i in range(3)]

# ── Interactive controls ─────────────────────────────────────────────
ctrl_color = "#1a1a1a"
txt_color  = "white"

# --- Fin count slider ---
ax_fins = fig.add_axes([0.06, 0.20, 0.25, 0.025])
ax_fins.set_facecolor(ctrl_color)
slider_fins = widgets.Slider(ax_fins, "Fins", 4, 20,
                              valinit=8, valstep=1,
                              color="#378ADD")
slider_fins.label.set_color(txt_color)
slider_fins.valtext.set_color(txt_color)

# --- Fan speed slider ---
ax_fan = fig.add_axes([0.06, 0.155, 0.25, 0.025])
ax_fan.set_facecolor(ctrl_color)
slider_fan = widgets.Slider(ax_fan, "Fan speed", 5, 100,
                             valinit=25, valstep=5,
                             color="#1D9E75")
slider_fan.label.set_color(txt_color)
slider_fan.valtext.set_color(txt_color)

# --- Power override slider ---
ax_pow = fig.add_axes([0.06, 0.11, 0.25, 0.025])
ax_pow.set_facecolor(ctrl_color)
slider_pow = widgets.Slider(ax_pow, "Power (W)", 0, 150,
                             valinit=0, valstep=5,
                             color="#EF9F27")
slider_pow.label.set_color(txt_color)
slider_pow.valtext.set_color(txt_color)

# --- Material radio ---
ax_mat = fig.add_axes([0.40, 0.08, 0.12, 0.14])
ax_mat.set_facecolor(ctrl_color)
ax_mat.set_title("Material", color=txt_color, fontsize=8, pad=4)
radio_mat = widgets.RadioButtons(ax_mat, ["aluminium", "copper"],
                                  activecolor="#ff6b35")
for lbl in radio_mat.labels:
    lbl.set_color(txt_color)
    lbl.set_fontsize(9)

# --- Fin type radio ---
ax_fin = fig.add_axes([0.57, 0.06, 0.14, 0.18])
ax_fin.set_facecolor(ctrl_color)
ax_fin.set_title("Fin type", color=txt_color, fontsize=8, pad=4)
radio_fin = widgets.RadioButtons(ax_fin,
                                  ["rectangular", "triangular", "pin"],
                                  activecolor="#ff6b35")
for lbl in radio_fin.labels:
    lbl.set_color(txt_color)
    lbl.set_fontsize(9)

# --- Stress test button ---
ax_stress = fig.add_axes([0.76, 0.10, 0.10, 0.06])
btn_stress = widgets.Button(ax_stress, "Stress test",
                             color="#3d0000", hovercolor="#7a0000")
btn_stress.label.set_color("#ff4444")
btn_stress.label.set_fontsize(9)

# --- Reset button ---
ax_reset = fig.add_axes([0.88, 0.10, 0.08, 0.06])
btn_reset = widgets.Button(ax_reset, "Reset",
                            color="#1a1a2e", hovercolor="#2a2a4e")
btn_reset.label.set_color("#aaaaff")
btn_reset.label.set_fontsize(9)

# ── Control callbacks ────────────────────────────────────────────────
def on_fins_change(val):
    state["n_fins"] = int(slider_fins.val)

def on_fan_change(val):
    state["h_conv"] = float(slider_fan.val)

def on_power_change(val):
    v = float(slider_pow.val)
    state["power_override"] = v if v > 0 else None

def on_material(label):
    state["material"] = label

def on_fin_type(label):
    state["fin_type"] = label

def on_stress(event):
    state["stress_test"] = True
    slider_pow.set_val(120)

def on_reset(event):
    state["stress_test"]    = False
    state["material"]       = "copper"
    state["fin_type"]       = "rectangular"
    state["n_fins"]         = 8
    state["h_conv"]         = 25.0
    state["power_override"] = None
    slider_fins.set_val(8)
    slider_fan.set_val(25)
    slider_pow.set_val(0)
    radio_mat.set_active(1)
    radio_fin.set_active(0)

slider_fins.on_changed(on_fins_change)
slider_fan.on_changed(on_fan_change)
slider_pow.on_changed(on_power_change)
radio_mat.on_clicked(on_material)
radio_fin.on_clicked(on_fin_type)
btn_stress.on_clicked(on_stress)
btn_reset.on_clicked(on_reset)

# ── Update loop ──────────────────────────────────────────────────────
def update(_frame):
    data     = get_sensor_data()
    cpu_temp = data["cpu_temp_c"]
    usage    = data["cpu_usage_percent"]

    # Power source — override slider or real sensor
    if state["power_override"] and state["power_override"] > 0:
        power = state["power_override"]
    else:
        power = max(data["cpu_power_w"],
                    25.0 + usage * 0.5)

    # Solve with current state
    from solver import N_FINS as _
    import solver as slv
    slv.N_FINS  = state["n_fins"]
    slv.H_CONV  = state["h_conv"]

    T       = solve_heat_distribution(power,
                                       state["material"],
                                       state["fin_type"])
    metrics = compute_metrics(T, power,
                               state["material"],
                               state["fin_type"])

    # Update heatmap
    heatmap.set_data(T)
    heatmap.set_clim(vmin=25,
                     vmax=max(100, metrics["T_max_c"] + 10))

    mode = "STRESS" if state["stress_test"] else "LIVE"
    txt_info.set_text(
        f"[{mode}]  CPU: {cpu_temp}°C  |  "
        f"Power: {power:.1f}W  |  "
        f"Usage: {usage}%  |  "
        f"Sim Tmax: {metrics['T_max_c']}°C  |  "
        f"Fins: {state['n_fins']}  |  "
        f"Fan: {state['h_conv']:.0f} W/m²K"
    )
    txt_warn.set_text(metrics["warning"] or "")

    # History
    tick[0] += 1
    time_history.append(tick[0])
    temp_history.append(cpu_temp)
    tmax_history.append(metrics["T_max_c"])
    if len(time_history) > MAX_HISTORY:
        time_history.pop(0)
        temp_history.pop(0)
        tmax_history.pop(0)

    line_temp.set_data(time_history, temp_history)
    line_tmax.set_data(time_history, tmax_history)
    ax_line.set_xlim(max(0, tick[0] - MAX_HISTORY), tick[0] + 2)
    all_t = temp_history + tmax_history
    ax_line.set_ylim(min(all_t) - 5, max(all_t) + 10)
    ax_line.legend(fontsize=6, facecolor="#1a1a1a",
                   labelcolor="white", loc="upper left")

    # Bar chart — all 3 fin types at current power
    fin_types = ["rectangular", "triangular", "pin"]
    for i, (bar, ft, bt) in enumerate(zip(bars, fin_types, bar_texts)):
        T_fi = solve_heat_distribution(power, state["material"], ft)
        m_fi = compute_metrics(T_fi, power, state["material"], ft)
        tmax = m_fi["T_max_c"]
        bar.set_height(tmax)
        bar.set_color("#ff4444" if tmax >= 85 else bar_colors[i])
        bt.set_text(f"{tmax}°C")
        bt.set_y(tmax + 1)

    return (heatmap, line_temp, line_tmax,
            txt_info, txt_warn, *bars, *bar_texts)


ani = animation.FuncAnimation(fig, update, interval=INTERVAL,
                               blit=False, cache_frame_data=False)
plt.show()


