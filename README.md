# Pirani Vacuum Gauge Simulator

Interactive Python desktop application for understanding and exploring Pirani gauge behavior across gas species, pressure regimes, and gauge geometries.

This project combines:
- educational visualization of Pirani physics,
- interactive parametric simulation,
- comparison to published correction-factor data,
- practical pressure-correction utilities.

## What This Program Is

This simulator is a physics-informed teaching and engineering exploration tool for thermal-conductivity vacuum gauging.

It is not intended to replace factory calibration or metrology-grade uncertainty analysis. Instead, it helps answer questions such as:
- Why does a Pirani reading depend on gas species?
- How do geometry and accommodation coefficient change sensitivity?
- Where does molecular behavior transition toward viscous behavior?
- How does calibration error evolve over pressure?

## What It Includes

- Gas data explorer for supported species (mass, degrees of freedom, mean molecular speed, and related properties)
- 2D pressure-response simulation with molecular and viscous contribution breakdown
- 3D parameter sweeps for pressure, temperature, geometry, and accommodation
- Correction factor views (including theory vs. experimental comparison)
- Accommodation coefficient exploration for tungsten and silicon references
- Utility calculator for indicated-to-true pressure correction

## How It Works

At a high level, the app models heat transfer from the heated sensing element to gas as pressure-dependent thermal coupling.

Core concepts used by the simulator:

1. Molecular regime term (low pressure, free-molecule transport dominant)
2. Continuum/viscous contribution (higher pressure, collisional transport increases)
3. Smooth bridging expression to combine regimes over a wide pressure range
4. Gas-specific correction behavior relative to nitrogen calibration

The implementation uses tabulated gas properties, configurable geometry, and accommodation-ratio scaling to estimate relative behavior and correction trends.

## Physics Model Summary

Representative relations used in the app include:

$$
\dot{Q}_{\mathrm{mol}} \propto a_E\,\frac{f+1}{8}\,\bar{c}\,A\,\frac{T_1-T_2}{T_x}\,p
$$

and a pressure-bridged form:

$$
\dot{Q}_{\mathrm{gas}} = \frac{\alpha p}{1 + g p}
$$

Correction-factor framing (for gas $X$ vs. nitrogen-calibrated reading):

$$
CF_{X/N_2} = \frac{p_{\mathrm{true}}}{p_{\mathrm{indicated}}}
$$

The exact simulator implementation includes additional practical terms and tuning constants to keep behavior physically anchored across a wide range while still matching expected qualitative trends.

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run with launcher (recommended):

```bash
python run_pirani.py
```

3. Or run directly:

```bash
python pirani_simulator.py
```

## Requirements

- Python 3.8+
- tkinter (usually bundled with Python installers)
- numpy
- matplotlib

## Repository Structure

- pirani_simulator.py: Main GUI application and simulation logic
- run_pirani.py: Lightweight launcher and startup checks
- requirements.txt: Python dependencies
- README.md: Project overview and usage
- docs/SOURCES.md: Source list and data provenance notes

## Data, Sources, and Provenance

The primary scientific basis for the gas-species dependence treatment is:

- K. Jousten, "On the gas species dependence of Pirani vacuum gauges," J. Vac. Sci. Technol. A 26, 352-359 (2008), https://doi.org/10.1116/1.2897314

Additional constants and engineering assumptions in the code are documented as implementation choices to support simulation stability and educational visualization. See docs/SOURCES.md for a structured source/provenance breakdown.

## Limitations

- This is a simulator, not a traceable calibration system.
- Results depend on model assumptions and selected gauge configuration presets.
- Absolute accuracy at extremes should be treated cautiously; relative trend analysis is the primary use case.

## Troubleshooting

If tkinter is unavailable on your platform, install the platform package for Tk support and rerun.

- Ubuntu/Debian: sudo apt-get install python3-tk
- Fedora/RHEL: sudo dnf install python3-tkinter
- Windows: reinstall Python with Tcl/Tk selected

If plots render slowly, reduce simulation resolution or simplify 3D sweeps.
