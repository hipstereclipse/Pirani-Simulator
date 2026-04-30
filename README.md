# Pirani Vacuum Gauge Simulator

**Based on:** Jousten (2008) "On the gas species dependence of Pirani vacuum gauges"  
*J. Vac. Sci. Technol. A 26, 352–359*

A comprehensive Python desktop application for understanding Pirani gauge physics, 
simulating gas-dependent heat transfer, and exploring the experimental results from the paper.

---

## Quick Start

```bash
# 1. Install dependencies (if needed)
pip install matplotlib numpy

# 2. Run the launcher (checks everything automatically)
python run_pirani.py

# OR run the simulator directly
python pirani_simulator.py
```

### Requirements
- **Python 3.8+**
- **tkinter** (usually included with Python — see Troubleshooting below)
- **matplotlib** ≥ 3.5
- **numpy** ≥ 1.21

---

## Application Tabs

### 💡 How It Works
Educational walkthrough of Pirani gauge physics: energy balance, molecular vs. viscous 
regimes, the combined formula, and why gas species matters.

### 🧪 Gas Explorer
Interactive comparison of all nine gas species from Table I. Sort by molecular mass, 
degrees of freedom, mean velocity, p·λ̄ product, or heat capacity ratio.

### 📐 Gauge Geometry
2D cross-section and 3D visualization of two gauge configurations:
- **Wire-in-cylinder** (conventional, VM1/VM2/VM4 from the paper)
- **MEMS parallel plate** (VM3 from the paper)

Adjust pressure to see Knudsen number change and flow regime transitions.

### 📈 2D Simulator
The core simulation tool. Plot heat transfer Q̇_gas vs. pressure for any combination of gases.

Controls:
- Select any combination of 9 gas species
- Choose gauge configuration (conventional, MEMS, custom wire, custom plate)
- Adjust wire temperature, enclosure temperature, accommodation coefficient
- Adjust wire radius, wire length, enclosure radius
- Toggle log/linear scale
- Show molecular vs. viscous regime breakdown
- Show total electrical power including p₀ offset

### 🌐 3D Simulator
Interactive 3D surface plots exploring multi-parameter dependencies:

1. **Pressure × Accommodation → Q**: See how heat flow depends on both pressure and accommodation coefficient
2. **Pressure × Wire Temp → Q**: Effect of wire temperature on the calibration curve
3. **Pressure × Gap Size → Q**: How geometry affects sensitivity and range
4. **Gas Species × Pressure → CF**: All correction factors evolving with pressure
5. **Molecular vs Viscous (3D)**: Compare regime contributions in a 3D breakdown

All 3D plots support mouse rotation, zoom, and multiple colormaps.

### 🔧 Correction Factors
Four visualization modes for the paper's experimental correction factor data:

1. **Bar Chart**: Mean correction factors with error bars and MEMS comparison
2. **True vs Indicated** (Fig. 4): The classic log-log calibration view
3. **Theory vs Experiment**: Scatter plot comparing Eq. 11 predictions to data
4. **Spread**: Gauge-to-gauge variation as horizontal range bars

### ⚛ Accommodation
Explore accommodation coefficients on tungsten and silicon surfaces (Tables VII & VIII).
Adjust the reference a_N₂ and see how all absolute coefficients scale. The visualization 
highlights unphysical values (> 1).

### 🧮 Calculator
Practical calculation tools:
- **Pressure Correction**: Convert indicated → true pressure for any gas
- **Heat Transfer**: Calculate molecular, viscous, and combined heat flow
- **Quick Reference**: All correction factors at a glance

---

## Key Physics

The fundamental equation for gas heat transfer in the molecular regime:

    Q̇_gas,mol = aE · (f+1)/8 · c̄ · A · (T₁−T₂)/Tx · p

Combined molecular + viscous (Eq. 9):

    Q̇_gas = αp / (1 + gp)

Normalized correction factor:

    CF_X/N₂ = p_cal / p_ind  (for gas species X, calibrated on N₂)

---

## Troubleshooting

### "No module named 'tkinter'"
tkinter comes pre-installed with most Python distributions. If missing:
- **Ubuntu/Debian**: `sudo apt-get install python3-tk`
- **Fedora/RHEL**: `sudo dnf install python3-tkinter`
- **macOS**: `brew install python-tk@3.12` (or reinstall from python.org)
- **Windows**: Reinstall Python, ensuring "tcl/tk and IDLE" is checked

### Plots look strange or don't update
Try resizing the window. If using a HiDPI display, matplotlib should scale 
automatically. You can also set `matplotlib.rcParams['figure.dpi'] = 150` in 
the script.

### 3D plots are slow to rotate
Reduce the number of data points by adjusting parameters. The 3D surface 
plots use 50×50 grids by default.

---

## File Structure

```
pirani_simulator.py   — Main application (all code in one file)
run_pirani.py         — Launcher with dependency checking
requirements.txt      — Python package dependencies
README.md             — This file
```

---

## Reference

K. Jousten, "On the gas species dependence of Pirani vacuum gauges,"
J. Vac. Sci. Technol. A 26, 352–359 (2008).
https://doi.org/10.1116/1.2897314
