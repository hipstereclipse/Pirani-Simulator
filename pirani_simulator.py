#!/usr/bin/env python3
"""
Pirani Vacuum Gauge Simulator
Based on: Jousten (2008) "On the gas species dependence of Pirani vacuum gauges"
J. Vac. Sci. Technol. A 26, 352-359

A comprehensive educational tool for understanding Pirani gauge physics,
gas-dependent heat transfer, correction factors, and accommodation coefficients.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import art3d
from matplotlib import cm
import matplotlib.colors as mcolors
import sys
import math
import time
import traceback
from collections import deque

# ══════════════════════════════════════════════════════════════════════════════
#  PHYSICS DATA FROM THE PAPER
# ══════════════════════════════════════════════════════════════════════════════

kB = 1.380649e-23  # Boltzmann constant (J/K)
R_UNIV = 8.314462618  # Universal gas constant (J/mol/K)
G_STD = 9.80665       # Gravitational acceleration (m/s^2)
SIGMA_SB = 5.670374419e-8  # Stefan-Boltzmann constant (W/m^2/K^4)
AMU_TO_KG = 1.66053906660e-27

# Keep continuum term physically anchored; avoid empirical gain inflation.
VISCOUS_GAIN = 1.0
GRAVITY_Z_CONVECTION_GAIN = 0.22

# Approximate gas transport properties near 300 K for natural-convection scaling.
# mu0: dynamic viscosity (Pa·s), k0: thermal conductivity (W/m/K).
GAS_TRANSPORT = {
    'H2':  {'mu0': 8.9e-6,  'k0': 0.180},
    'He':  {'mu0': 19.6e-6, 'k0': 0.151},
    'Ne':  {'mu0': 31.0e-6, 'k0': 0.049},
    'CO':  {'mu0': 17.2e-6, 'k0': 0.025},
    'N2':  {'mu0': 17.8e-6, 'k0': 0.026},
    'O2':  {'mu0': 20.2e-6, 'k0': 0.026},
    'Ar':  {'mu0': 22.6e-6, 'k0': 0.018},
    'CO2': {'mu0': 14.8e-6, 'k0': 0.016},
    'Kr':  {'mu0': 24.7e-6, 'k0': 0.0094},
    'Xe':  {'mu0': 22.3e-6, 'k0': 0.0056},
}

GAS_DATA = {
    'H2':  {'name': 'Hydrogen',        'symbol': 'H₂',  'm': 2.016,  'f': 5, 'gamma': 1.41, 'cbar': 1764, 'plbar': 12.2e-3, 'color': '#ff6b6b'},
    'He':  {'name': 'Helium',          'symbol': 'He',   'm': 4.003,  'f': 3, 'gamma': 1.67, 'cbar': 1252, 'plbar': 19.0e-3, 'color': '#ffd93d'},
    'Ne':  {'name': 'Neon',            'symbol': 'Ne',   'm': 20.18,  'f': 3, 'gamma': 1.67, 'cbar': 557,  'plbar': 13.6e-3, 'color': '#6bcb77'},
    'CO':  {'name': 'Carbon Monoxide', 'symbol': 'CO',   'm': 28.011, 'f': 5, 'gamma': 1.40, 'cbar': 473,  'plbar': 6.4e-3,  'color': '#4d96ff'},
    'N2':  {'name': 'Nitrogen',        'symbol': 'N₂',   'm': 28.013, 'f': 5, 'gamma': 1.40, 'cbar': 473,  'plbar': 6.4e-3,  'color': '#a0a0a0'},
    'O2':  {'name': 'Oxygen',          'symbol': 'O₂',   'm': 31.999, 'f': 5, 'gamma': 1.40, 'cbar': 444,  'plbar': 7.1e-3,  'color': '#3498db'},
    'Ar':  {'name': 'Argon',           'symbol': 'Ar',   'm': 39.948, 'f': 3, 'gamma': 1.67, 'cbar': 396,  'plbar': 6.8e-3,  'color': '#9b59b6'},
    'CO2': {'name': 'Carbon Dioxide',  'symbol': 'CO₂',  'm': 44.01,  'f': 6, 'gamma': 1.33, 'cbar': 377,  'plbar': 4.3e-3,  'color': '#e67e22'},
    'Kr':  {'name': 'Krypton',         'symbol': 'Kr',   'm': 83.8,   'f': 3, 'gamma': 1.67, 'cbar': 274,  'plbar': 5.3e-3,  'color': '#1abc9c'},
    'Xe':  {'name': 'Xenon',           'symbol': 'Xe',   'm': 131.3,  'f': 3, 'gamma': 1.67, 'cbar': 219,  'plbar': 3.9e-3,  'color': '#e74c3c'},
}

# Experimental correction factors (Table IX)
EXPERIMENTAL_CF = {
    'H2':  {'range': '0.1-13',   'mean': 0.62, 'spread': 0.07, 'cfMin': 0.58, 'cfMax': 0.59, 'vm3': 0.72},
    'He':  {'range': '0.1-13',   'mean': 1.04, 'spread': 0.12, 'cfMin': 0.94, 'cfMax': 1.08, 'vm3': 1.19},
    'Ne':  {'range': '0.5-30',   'mean': 1.39, 'spread': 0.07, 'cfMin': 1.32, 'cfMax': 1.46, 'vm3': 1.42},
    'CO':  {'range': '0.1-200',  'mean': 0.98, 'spread': 0.02, 'cfMin': 0.96, 'cfMax': 1.00, 'vm3': 0.97},
    'N2':  {'range': '0.1-1e4',  'mean': 1.00, 'spread': 0.00, 'cfMin': 1.00, 'cfMax': 1.00, 'vm3': 1.00},
    'O2':  {'range': '0.1-100',  'mean': 1.01, 'spread': 0.03, 'cfMin': 0.98, 'cfMax': 1.04, 'vm3': 1.00},
    'Ar':  {'range': '0.1-90',   'mean': 1.62, 'spread': 0.12, 'cfMin': 1.59, 'cfMax': 1.79, 'vm3': 1.51},
    'CO2': {'range': '0.1-30',   'mean': 0.95, 'spread': 0.03, 'cfMin': 0.95, 'cfMax': 0.97, 'vm3': 0.92},
    'Kr':  {'range': '0.5-90',   'mean': 2.22, 'spread': 0.16, 'cfMin': 2.20, 'cfMax': 2.41, 'vm3': 2.03},
    'Xe':  {'range': '0.5-13',   'mean': 2.71, 'spread': 0.20, 'cfMin': 2.70, 'cfMax': 2.95, 'vm3': 2.48},
}

# Accommodation coefficient ratios (Tables VII & VIII)
ACCOM_RATIOS_W = {'H2': 0.46, 'He': 0.57, 'Ne': 0.93, 'CO': 1.02, 'N2': 1.00, 'O2': 1.01, 'Ar': 1.08, 'CO2': 1.12, 'Kr': 1.14, 'Xe': 1.16}
ACCOM_RATIOS_Si = {'H2': 0.37, 'He': 0.48, 'Ne': 0.89, 'CO': 1.03, 'N2': 1.00, 'O2': 1.02, 'Ar': 1.19, 'CO2': 1.17, 'Kr': 1.28, 'Xe': 1.31}

# ── Shared application state (populated after Tk root is created) ────────────
APP_STATE = {}

THEME_MODES = ('dark', 'light')

# ── Pressure unit conversions ────────────────────────────────────────────────
PRESSURE_UNITS = {
    'Pa':   {'factor': 1.0,           'label': 'Pa',   'name': 'Pascal'},
    'mbar': {'factor': 0.01,          'label': 'mbar', 'name': 'Millibar'},
    'Torr': {'factor': 1.0 / 133.322, 'label': 'Torr', 'name': 'Torr'},
    'µbar': {'factor': 10.0,          'label': 'µbar', 'name': 'Microbar'},
    'atm':  {'factor': 1.0 / 101325,  'label': 'atm',  'name': 'Atmosphere'},
}


def convert_pressure(p_pa, unit='Pa'):
    """Convert pressure from Pa to the specified unit."""
    return p_pa * PRESSURE_UNITS.get(unit, PRESSURE_UNITS['Pa'])['factor']


def format_pressure(p_pa, unit='Pa', fmt='{:.3g}'):
    """Format pressure value with unit label."""
    val = convert_pressure(p_pa, unit)
    return fmt.format(val) + ' ' + PRESSURE_UNITS[unit]['label']


def get_pressure_unit():
    """Return the currently selected pressure unit string."""
    var = APP_STATE.get('pressure_unit')
    if var is not None:
        return var.get()
    return 'mbar'


TEMPERATURE_UNITS = {
    'K': {'label': 'K'},
    'C': {'label': '°C'},
    'F': {'label': '°F'},
}


def convert_temperature(temp, from_unit='K', to_unit='K'):
    """Convert temperature value between K, C, and F."""
    if from_unit == to_unit:
        return temp

    if from_unit == 'K':
        k = temp
    elif from_unit == 'C':
        k = temp + 273.15
    elif from_unit == 'F':
        k = (temp - 32.0) * 5.0 / 9.0 + 273.15
    else:
        k = temp

    if to_unit == 'K':
        return k
    if to_unit == 'C':
        return k - 273.15
    if to_unit == 'F':
        return (k - 273.15) * 9.0 / 5.0 + 32.0
    return k


def format_temperature(temp_k, unit='K', fmt='{:.1f}'):
    """Format a Kelvin temperature into the selected display unit."""
    val = convert_temperature(temp_k, 'K', unit)
    return fmt.format(val) + ' ' + TEMPERATURE_UNITS[unit]['label']


def get_temperature_unit():
    """Return the currently selected temperature unit string."""
    var = APP_STATE.get('temperature_unit')
    if var is not None:
        return var.get()
    return 'C'


def get_theme_mode():
    """Return the currently selected UI theme mode string."""
    var = APP_STATE.get('theme_mode')
    if var is not None:
        mode = var.get()
        if mode in THEME_MODES:
            return mode
    return 'light'


# Gauge configurations — modelled from commercial & research specifications
#
# accuracy_tiers: list of (p_lo_mbar, p_hi_mbar, fraction_of_reading)
# sensor: electro-thermal bridge parameters for the MolecularSim tab

GAUGE_CONFIGS = {
    # ── PSG55x (INFICON) — Conventional Wire Pirani ──────────────────────
    'psg55x': {
        'name': 'PSG55x — Wire Pirani',
        'desc': 'INFICON PSG55x conventional constant-temperature wire\n'
                'Pirani. Tungsten filament ≤160 °C, Wheatstone bridge.\n'
                'Range: 5×10⁻⁵ to 1000 mbar.',
        'T1': 423, 'T2': 296, 'wire_r': 5e-6, 'wire_L': 0.025, 'enc_r': 5.6e-3,
        't_hot_range_k': (373.0, 453.0), 't_cold_range_k': (250.0, 340.0),
        'surface': 'W', 'geometry': 'cylindrical', 'orientation': 'vertical',
        'sat_target_mbar': 1000.0,
        'conv_gain': 0.15, 'conv_p_on_mbar': 100.0, 'conv_transition_n': 1.30,
        'accuracy_tiers': [
            (5e-5, 1e-3, 0.50),
            (1e-3, 100.0, 0.15),
            (100.0, 1000.0, 0.50),
        ],
        'range_mbar': (5e-5, 1000.0),
        'readout_profile': {
            'low_floor_mult': 0.30,
            'low_knee_mult': 14.0,
            'high_knee_frac_sat': 0.22,
            'high_ref_frac_sat': 2.2,
            'high_saturation_bend': 0.30,
            'low_edge_max_err': 0.55,
            'high_edge_max_err': 0.52,
        },
        'sensor': {
            'r0_ohm': 200.0, 'tcr_per_k': 0.0045, 'emissivity': 0.30,
            'bridge_v_bias': 5.0, 'bridge_v_sensor': 2.5,
            'support_lambda_wmk': 160.0, 'support_w_m': 10e-6,
            'support_t_m': 10e-6, 'support_l_m': 5e-3,
            'extra_support_g_wpk': 1e-5,
        },
    },
    # ── PGE300 (INFICON) — Convection-Enhanced Horizontal Wire Pirani ────
    'pge300': {
        'name': 'PGE300 — Convection Pirani',
        'desc': 'INFICON PGE300 convection-enhanced Pirani.\n'
                'Horizontal gold-plated tungsten wire extends\n'
                'usable range to atmosphere via convection.\n'
                'Range: 1.3×10⁻⁴ to 1333 mbar.',
        'T1': 393, 'T2': 296, 'wire_r': 5e-6, 'wire_L': 0.05, 'enc_r': 0.013,
        't_hot_range_k': (353.0, 433.0), 't_cold_range_k': (250.0, 340.0),
        'surface': 'W', 'geometry': 'cylindrical', 'orientation': 'horizontal',
        'sat_target_mbar': 1333.0,
        'conv_gain': 0.65, 'conv_p_on_mbar': 1.0, 'conv_transition_n': 1.20,
        'conv_gas_sensitivity': 0.50,
        'accuracy_tiers': [
            (1.3e-4, 1.3e-3, 1.00),
            (1.3e-3, 530.0,  0.10),
            (530.0,  1333.0, 0.025),
        ],
        'range_mbar': (1.3e-4, 1333.0),
        'readout_profile': {
            'low_floor_mult': 0.35,
            'low_knee_mult': 10.0,
            'high_knee_frac_sat': 0.35,
            'high_ref_frac_sat': 2.8,
            'high_saturation_bend': 0.22,
            'low_edge_max_err': 0.60,
            'high_edge_max_err': 0.30,
        },
        'sensor': {
            'r0_ohm': 150.0, 'tcr_per_k': 0.0045, 'emissivity': 0.15,
            'bridge_v_bias': 5.0, 'bridge_v_sensor': 2.5,
            'support_lambda_wmk': 160.0, 'support_w_m': 10e-6,
            'support_t_m': 10e-6, 'support_l_m': 5e-3,
            'extra_support_g_wpk': 1e-5,
        },
    },
    # ── PPG550 (INFICON) — MEMS Pirani + Piezo Combination ───────────────
    'ppg550': {
        'name': 'PPG550 — MEMS Pirani+Piezo',
        'desc': 'INFICON PPG550 dual-sensor: MEMS Pirani (low p)\n'
                '+ piezoresistive diaphragm (high p), silicon\n'
                'substrate. Orientation-independent.\n'
                'Range: 1×10⁻⁶ to 1333 mbar.',
        'T1': 333, 'T2': 296, 'plate_area': 1e-6, 'gap': 2e-6,
        't_hot_range_k': (313.0, 373.0), 't_cold_range_k': (250.0, 340.0),
        'surface': 'Si', 'geometry': 'plates',
        'sat_target_mbar': 10.0,
        'conv_gain': 0.08, 'conv_p_on_mbar': 140.0, 'conv_transition_n': 1.35,
        'has_piezo': True,
        'piezo_range_mbar': (2.0, 1333.0),
        'piezo_crossover_mbar': (1.5, 2.0),
        'accuracy_tiers': [
            (1e-5,  1e-4,   0.25),
            (1e-4,  2.0,    0.05),
            (2.0,   100.0,  0.01),
            (100.0, 800.0,  0.005),
            (800.0, 1100.0, 0.0025),
            (1100.0, 1333.0, 0.005),
        ],
        'range_mbar': (1e-6, 1333.0),
        'readout_profile': {
            'low_floor_mult': 0.45,
            'low_knee_mult': 8.0,
            'high_knee_frac_sat': 0.55,
            'high_ref_frac_sat': 3.5,
            'high_saturation_bend': 0.12,
            'low_edge_max_err': 0.40,
            'high_edge_max_err': 0.12,
        },
        'sensor': {
            'r0_ohm': 10000.0, 'tcr_per_k': 0.00385, 'emissivity': 0.08,
            'bridge_v_bias': 2.4, 'bridge_v_sensor': 1.2,
            'support_lambda_wmk': 20.0, 'support_w_m': 9e-6,
            'support_t_m': 0.6e-6, 'support_l_m': 500e-6,
            'extra_support_g_wpk': 2e-6,
        },
    },
    # ── Chen et al. 2023 — Composite MEMS Pirani (Research) ──────────────
    'chen_mems': {
        'name': 'Chen 2023 — Composite MEMS',
        'desc': 'Research dual-element MEMS Pirani (Chen et al.\n'
                '2023). Ti thermistor on SiNx cantilevers.\n'
                'P1 (100×100 µm, 78 µm gap) + P2 (100×36 µm,\n'
                '49 µm gap) in series.\n'
                'Range: 6.6×10⁻² to 1.12×10⁵ Pa.',
        'T1': 333, 'T2': 296, 'plate_area': 1.36e-8, 'gap': 60e-6,
        't_hot_range_k': (313.0, 363.0), 't_cold_range_k': (250.0, 340.0),
        'surface': 'Si', 'geometry': 'plates',
        'sat_target_mbar': 1120.0,
        'conv_gain': 0.05, 'conv_p_on_mbar': 200.0, 'conv_transition_n': 1.40,
        'accuracy_tiers': [
            (6.6e-4, 0.1,    0.20),
            (0.1,    10.0,   0.05),
            (10.0,   1120.0, 0.10),
        ],
        'range_mbar': (6.6e-4, 1120.0),
        'readout_profile': {
            'low_floor_mult': 0.40,
            'low_knee_mult': 9.0,
            'high_knee_frac_sat': 0.40,
            'high_ref_frac_sat': 2.4,
            'high_saturation_bend': 0.20,
            'low_edge_max_err': 0.45,
            'high_edge_max_err': 0.22,
        },
        'sensor': {
            'r0_ohm': 5000.0, 'tcr_per_k': 0.003, 'emissivity': 0.10,
            'bridge_v_bias': 2.4, 'bridge_v_sensor': 1.2,
            'support_lambda_wmk': 20.0, 'support_w_m': 9e-6,
            'support_t_m': 0.6e-6, 'support_l_m': 500e-6,
            'extra_support_g_wpk': 2e-6,
        },
    },
    # ── Jousten VM1/VM4 — Wire Pirani (Research, 2008) ───────────────────
    'jousten_wire': {
        'name': 'Jousten VM1/4 — Wire Pirani',
        'desc': 'Research-characterized constant-temperature wire\n'
                'gauge (Jousten 2008). Oxidized tungsten, 10 µm\n'
                'dia, 120 °C. VM4 zero-stability 4.8×10⁻⁶ Pa.',
        'T1': 393, 'T2': 296, 'wire_r': 5e-6, 'wire_L': 0.05, 'enc_r': 0.008,
        't_hot_range_k': (353.0, 433.0), 't_cold_range_k': (250.0, 340.0),
        'surface': 'W', 'geometry': 'cylindrical', 'orientation': 'vertical',
        'sat_target_mbar': 1000.0,
        'conv_gain': 0.25, 'conv_p_on_mbar': 90.0, 'conv_transition_n': 1.25,
        'accuracy_tiers': [
            (5e-4,  1e-3,   0.30),
            (1e-3,  100.0,  0.10),
            (100.0, 1000.0, 0.25),
        ],
        'range_mbar': (5e-4, 1000.0),
        'readout_profile': {
            'low_floor_mult': 0.32,
            'low_knee_mult': 12.0,
            'high_knee_frac_sat': 0.24,
            'high_ref_frac_sat': 2.2,
            'high_saturation_bend': 0.28,
            'low_edge_max_err': 0.52,
            'high_edge_max_err': 0.50,
        },
        'sensor': {
            'r0_ohm': 200.0, 'tcr_per_k': 0.0045, 'emissivity': 0.35,
            'bridge_v_bias': 5.0, 'bridge_v_sensor': 2.5,
            'support_lambda_wmk': 160.0, 'support_w_m': 10e-6,
            'support_t_m': 10e-6, 'support_l_m': 5e-3,
            'extra_support_g_wpk': 1e-5,
        },
    },
    # ── Jousten VM3 — MEMS Pirani (Research, 2008) ───────────────────────
    'jousten_mems': {
        'name': 'Jousten VM3 — MEMS Pirani',
        'desc': 'VM3 MEMS gauge (Jousten 2008). 1 mm² silicon\n'
                'sheet heated to 60 °C, parallel-plate geometry.\n'
                'Widest usable gas species range of all tested\n'
                'gauges.',
        'T1': 333, 'T2': 296, 'plate_area': 1e-6, 'gap': 1e-3,
        't_hot_range_k': (313.0, 373.0), 't_cold_range_k': (250.0, 340.0),
        'surface': 'Si', 'geometry': 'plates',
        'sat_target_mbar': 1000.0,
        'conv_gain': 0.08, 'conv_p_on_mbar': 140.0, 'conv_transition_n': 1.35,
        'accuracy_tiers': [
            (1e-5,   1e-3,   0.25),
            (1e-3,   100.0,  0.05),
            (100.0,  1000.0, 0.15),
        ],
        'range_mbar': (1e-5, 1000.0),
        'readout_profile': {
            'low_floor_mult': 0.40,
            'low_knee_mult': 9.0,
            'high_knee_frac_sat': 0.35,
            'high_ref_frac_sat': 2.5,
            'high_saturation_bend': 0.20,
            'low_edge_max_err': 0.45,
            'high_edge_max_err': 0.20,
        },
        'sensor': {
            'r0_ohm': 10000.0, 'tcr_per_k': 0.00385, 'emissivity': 0.20,
            'bridge_v_bias': 2.4, 'bridge_v_sensor': 1.2,
            'support_lambda_wmk': 70.0, 'support_w_m': 25e-6,
            'support_t_m': 2.0e-6, 'support_l_m': 450e-6,
            'extra_support_g_wpk': 2e-6,
        },
    },
    # ── Custom configs for user experimentation ──────────────────────────
    'custom_wire': {
        'name': 'Custom Wire Gauge',
        'desc': 'User-configurable wire-in-cylinder gauge.\n'
                'Adjust all parameters freely.',
        'T1': 393, 'T2': 296, 'wire_r': 5e-6, 'wire_L': 0.05, 'enc_r': 0.008,
        't_hot_range_k': (333.0, 523.0), 't_cold_range_k': (220.0, 360.0),
        'surface': 'W', 'geometry': 'cylindrical', 'orientation': 'vertical',
        'sat_target_mbar': 1000.0,
        'conv_gain': 0.25, 'conv_p_on_mbar': 90.0, 'conv_transition_n': 1.25,
        'accuracy_tiers': [
            (1e-4,   1e-2,   0.30),
            (1e-2,   100.0,  0.15),
            (100.0,  1000.0, 0.30),
        ],
        'range_mbar': (1e-4, 1000.0),
        'readout_profile': {
            'low_floor_mult': 0.30,
            'low_knee_mult': 12.0,
            'high_knee_frac_sat': 0.25,
            'high_ref_frac_sat': 2.2,
            'high_saturation_bend': 0.30,
            'low_edge_max_err': 0.55,
            'high_edge_max_err': 0.50,
        },
    },
    'custom_plate': {
        'name': 'Custom Parallel Plate',
        'desc': 'User-configurable parallel plate gauge.\n'
                'Adjust all parameters freely.',
        'T1': 353, 'T2': 296, 'plate_area': 1e-4, 'gap': 1e-3,
        't_hot_range_k': (303.0, 433.0), 't_cold_range_k': (220.0, 360.0),
        'surface': 'Si', 'geometry': 'plates',
        'sat_target_mbar': 1000.0,
        'conv_gain': 0.08, 'conv_p_on_mbar': 140.0, 'conv_transition_n': 1.35,
        'accuracy_tiers': [
            (1e-4,   1e-2,   0.25),
            (1e-2,   100.0,  0.10),
            (100.0,  1000.0, 0.20),
        ],
        'range_mbar': (1e-4, 1000.0),
        'readout_profile': {
            'low_floor_mult': 0.40,
            'low_knee_mult': 10.0,
            'high_knee_frac_sat': 0.35,
            'high_ref_frac_sat': 2.4,
            'high_saturation_bend': 0.20,
            'low_edge_max_err': 0.45,
            'high_edge_max_err': 0.20,
        },
    },
}

# Helper to look up accuracy fraction from a gauge config's tiers
def gauge_accuracy_fraction(cfg_key_or_cfg, p_mbar):
    """Return fractional accuracy-of-reading for a gauge config and pressure."""
    if isinstance(cfg_key_or_cfg, str):
        cfg = GAUGE_CONFIGS.get(cfg_key_or_cfg, {})
    else:
        cfg = cfg_key_or_cfg
    tiers = cfg.get('accuracy_tiers', [])
    for p_lo, p_hi, frac in tiers:
        if p_lo <= p_mbar <= p_hi:
            return frac
    # Out of defined range
    range_lo, range_hi = cfg.get('range_mbar', (1e-4, 1000.0))
    if p_mbar < range_lo:
        return 0.50
    if p_mbar > range_hi:
        return 0.50
    return 0.25

# ══════════════════════════════════════════════════════════════════════════════
#  PHYSICS CALCULATIONS
# ══════════════════════════════════════════════════════════════════════════════

def calc_Q_mol_cylinder(aE, f, cbar, r1, L, T1, T2, p):
    """Molecular regime heat flow for wire-in-cylinder (Eq. 3 with Tx=T2)."""
    A = 2 * np.pi * r1 * L
    Tx = T2
    return aE * ((f + 1) / 8.0) * cbar * A * ((T1 - T2) / Tx) * p

def calc_Q_mol_plates(aE1, aE2, f, cbar, A, T1, T2, p):
    """Molecular regime heat flow for parallel plates (Eq. 3 with Tx=(T1+T2)/2)."""
    aE_eff = (aE1 * aE2) / (aE1 + aE2 - aE1 * aE2)
    Tx = (T1 + T2) / 2.0
    return aE_eff * ((f + 1) / 8.0) * cbar * A * ((T1 - T2) / Tx) * p

def calc_Q_visc_cylinder(gamma, plbar, cbar, f, m_amu, L, T1, T2, r1, r2):
    """Viscous regime heat flow for wire-in-cylinder (Eq. 7) — pressure-independent."""
    m = m_amu * AMU_TO_KG
    lambda_p_pa = plbar * 100.0  # convert (m*mbar) -> (m*Pa)
    coeff = (9 * gamma - 5) / 4.0
    q_visc = coeff * (2 * np.pi * lambda_p_pa * cbar) * (f * kB / (2 * m)) * L * (T1 - T2) / np.log(r2 / r1)
    return VISCOUS_GAIN * q_visc

def calc_Q_visc_plates(gamma, plbar, cbar, f, m_amu, A, T1, T2, x):
    """Viscous regime heat flow for parallel plates (Eq. 6) — pressure-independent."""
    m = m_amu * AMU_TO_KG
    lambda_p_pa = plbar * 100.0  # convert (m*mbar) -> (m*Pa)
    coeff = (9 * gamma - 5) / 4.0
    q_visc = coeff * (2 * np.pi * lambda_p_pa * cbar) * (f * kB / (2 * m)) * A * (T1 - T2) / x
    return VISCOUS_GAIN * q_visc

def calc_Q_combined(Q_mol, Q_visc):
    """Combined heat flow using series-resistance analogy (Eq. 8)."""
    if Q_mol == 0 or Q_visc == 0:
        return 0.0
    return 1.0 / (1.0 / Q_mol + 1.0 / Q_visc)


def _shape_factors(config, p_pa):
    """Return (molecular_factor, viscous_factor) including geometry non-idealities.

    For square/cubic cavities, wall-corner anisotropy is weak in free-molecular
    flow and strengthens as pressure rises into transition/viscous regimes.
    """
    if config.get('geometry') != 'square_cavity':
        return 1.0, 1.0

    p_mbar = convert_pressure(np.maximum(np.asarray(p_pa, dtype=np.float64), 0.0), 'mbar')
    p_shape_on = max(float(config.get('shape_p_on_mbar', 40.0)), 1e-9)
    n_shape = max(float(config.get('shape_transition_n', 1.2)), 0.4)
    x = (p_mbar / p_shape_on) ** n_shape
    activation = x / (1.0 + x)

    mol_floor = float(config.get('shape_mol_factor', 0.94))
    visc_floor = float(config.get('shape_visc_factor', 0.84))
    mol_fac = 1.0 - (1.0 - mol_floor) * activation
    visc_fac = 1.0 - (1.0 - visc_floor) * activation

    if np.ndim(mol_fac) == 0:
        return float(mol_fac), float(visc_fac)
    return mol_fac, visc_fac


def _get_gas_transport(gas_key, t_film_k):
    """Return (mu, k, cp_mass, rho) at film temperature for one gas."""
    gas = GAS_DATA[gas_key]
    d = GAS_TRANSPORT.get(gas_key, GAS_TRANSPORT['N2'])
    t = max(float(t_film_k), 180.0)

    # Mild temperature scaling for educational-level natural convection model.
    mu = d['mu0'] * (t / 300.0) ** 0.70
    k = d['k0'] * (t / 300.0) ** 0.85

    m_kg_per_mol = gas['m'] / 1000.0
    cp_molar = ((gas['f'] + 2.0) / 2.0) * R_UNIV
    cp_mass = cp_molar / m_kg_per_mol
    rho = 1.0  # placeholder, set by pressure in convection helpers
    return mu, k, cp_mass, rho


def _mean_thermal_speed(gas_key, wall_temperature_k):
    """Mean molecular thermal speed cbar = sqrt(8kT/pi m) at wall temperature."""
    gas = GAS_DATA[gas_key]
    m = gas['m'] * AMU_TO_KG
    t = max(float(wall_temperature_k), 1.0)
    return math.sqrt((8.0 * kB * t) / (math.pi * m))


def _effective_accommodation(gas_key, surface, aN2):
    """Effective energy accommodation coefficient for a gas/surface pair."""
    accom_table = ACCOM_RATIOS_W if surface == 'W' else ACCOM_RATIOS_Si
    aE_ratio = accom_table.get(gas_key, 1.0)
    return min(float(aN2) * aE_ratio, 1.0)


def _viscous_target_scale(config, aN2, t_hot_k, t_cold_k):
    """Return scale factor so Qvisc/alpha transition hits sat_target_mbar.

    For Q = alpha*P / (1 + alpha*P/Qvisc), the characteristic turnover is
    P* = Qvisc/alpha. We rescale Qvisc to place P* near the requested target.
    """
    target_mbar = config.get('sat_target_mbar', None)
    if target_mbar is None:
        return 1.0

    target_mbar = float(target_mbar)
    if target_mbar <= 0.0:
        return 1.0

    ref_gas = config.get('sat_ref_gas', 'N2')
    gas = GAS_DATA.get(ref_gas, GAS_DATA['N2'])
    t_hot = float(t_hot_k)
    t_cold = float(t_cold_k)
    d_t = max(t_hot - t_cold, 1e-12)
    cbar = _mean_thermal_speed(ref_gas, t_cold)
    aE = _effective_accommodation(ref_gas, config.get('surface', 'W'), aN2)

    if config['geometry'] in ('cylindrical', 'square_cavity'):
        r1 = float(config['wire_r'])
        L = float(config['wire_L'])
        r2 = float(config['enc_r'])
        A = 2.0 * math.pi * r1 * L
        tx = max(t_cold, 1e-9)
        alpha = aE * ((gas['f'] + 1.0) / 8.0) * cbar * A * (d_t / tx)
        q_visc = calc_Q_visc_cylinder(gas['gamma'], gas['plbar'], cbar, gas['f'], gas['m'], L, t_hot, t_cold, r1, r2)
    else:
        A = float(config['plate_area'])
        x = float(config['gap'])
        aE_eff = (aE * aE) / max(aE + aE - aE * aE, 1e-12)
        tx = max(0.5 * (t_hot + t_cold), 1e-9)
        alpha = aE_eff * ((gas['f'] + 1.0) / 8.0) * cbar * A * (d_t / tx)
        q_visc = calc_Q_visc_plates(gas['gamma'], gas['plbar'], cbar, gas['f'], gas['m'], A, t_hot, t_cold, x)

    p_target_pa = target_mbar * 100.0
    if alpha <= 1e-30 or q_visc <= 1e-30:
        return 1.0

    scale = (p_target_pa * alpha) / q_visc
    return float(np.clip(scale, 1e-8, 1e8))


def _convective_gas_sensitivity(config):
    """Return [0,1] gas-property sensitivity for convection augmentation.

    Horizontal wire tends to suppress buoyancy-induced composition sensitivity
    in this enclosed geometry. Square cavities amplify non-uniform flow paths.
    """
    if 'conv_gas_sensitivity' in config:
        return float(np.clip(config['conv_gas_sensitivity'], 0.0, 1.0))

    geom = config.get('geometry')
    orient = config.get('orientation', 'vertical')
    if geom == 'square_cavity':
        return 0.70
    if geom == 'plates':
        return 0.35
    if orient == 'horizontal':
        return 0.15
    return 0.55


def _convective_viscous_multiplier(gas_key, gas, config, p_pa):
    """Return convection boost multiplier added to viscous heat-flow term.

    Semi-empirical natural-convection augmentation activated in viscous/
    transition flow. Gravity acts along the z-direction; orientation changes
    how strongly buoyancy-driven convection couples into gauge heat transfer.
    """
    p_pa = max(float(p_pa), 0.0)
    t_hot = float(config.get('T1', 393.0))
    t_cold = float(config.get('T2', 296.0))
    t_film = max(0.5 * (t_hot + t_cold), 180.0)
    d_t = max(t_hot - t_cold, 0.0)
    if d_t <= 0.0 or p_pa <= 0.0:
        return 0.0

    mu_g, k_g, cp_g, _ = _get_gas_transport(gas_key, t_film)
    mu_ref, k_ref, cp_ref, _ = _get_gas_transport('N2', t_film)
    s = _convective_gas_sensitivity(config)
    mu = mu_ref * ((mu_g / max(mu_ref, 1e-18)) ** s)
    k = k_ref * ((k_g / max(k_ref, 1e-18)) ** s)
    cp_mass = cp_ref * ((cp_g / max(cp_ref, 1e-18)) ** s)

    m_ref = GAS_DATA['N2']['m'] / 1000.0
    m_g = gas['m'] / 1000.0
    m_kg_per_mol = m_ref * ((m_g / max(m_ref, 1e-18)) ** s)
    rho = (p_pa * m_kg_per_mol) / (R_UNIV * t_film)
    rho = max(rho, 1e-12)

    nu = mu / rho
    alpha = k / max(rho * cp_mass, 1e-12)
    pr = float(np.clip(nu / max(alpha, 1e-12), 0.2, 4.0))

    geom = config.get('geometry')
    orient = config.get('orientation', 'vertical')
    if geom == 'plates':
        l_char = max(float(config.get('gap', 1e-3)), 1e-6)
    else:
        r2 = float(config.get('enc_r', 8e-3))
        r1 = float(config.get('wire_r', 5e-6))
        l_char = max(r2 - r1, 5e-5)

    beta = 1.0 / t_film
    ra = G_STD * beta * d_t * (l_char ** 3) / max(nu * alpha, 1e-18)
    ra = float(np.clip(ra, 0.0, 1e12))

    if geom == 'cylindrical' and orient == 'horizontal':
        # Churchill-Chu style correlation for horizontal cylinder.
        nu_nat = 0.36 + (0.518 * (ra ** 0.25)) / ((1.0 + (0.559 / pr) ** (9.0 / 16.0)) ** (4.0 / 9.0))
    else:
        # Vertical-wire / plate-like correlation.
        nu_nat = 0.68 + (0.67 * (ra ** 0.25)) / ((1.0 + (0.492 / pr) ** (9.0 / 16.0)) ** (4.0 / 9.0))

    conv_strength = max(nu_nat - 1.0, 0.0)
    ra_on = max(float(config.get('conv_ra_on', 70.0)), 1e-9)
    ra_n = max(float(config.get('conv_transition_n', 1.25)), 0.4)
    ra_scale = (ra / ra_on) ** ra_n
    activation = ra_scale / (1.0 + ra_scale)

    if geom == 'square_cavity':
        geom_factor = 0.90
    elif geom == 'plates':
        geom_factor = 0.55
    else:
        geom_factor = 1.0

    gain = float(config.get('conv_gain', 0.22))
    mult = gain * geom_factor * conv_strength * activation * (1.0 + GRAVITY_Z_CONVECTION_GAIN)
    return float(np.clip(mult, 0.0, 1.4))


def _convective_viscous_multiplier_vec(gas_key, gas, config, pressures_pa):
    """Vectorized convection boost multiplier for viscous regime."""

    pressures_pa = np.maximum(np.asarray(pressures_pa, dtype=np.float64), 0.0)
    t_hot = float(config.get('T1', 393.0))
    t_cold = float(config.get('T2', 296.0))
    t_film = max(0.5 * (t_hot + t_cold), 180.0)
    d_t = max(t_hot - t_cold, 0.0)
    if d_t <= 0.0:
        return np.zeros_like(pressures_pa)

    mu_g, k_g, cp_g, _ = _get_gas_transport(gas_key, t_film)
    mu_ref, k_ref, cp_ref, _ = _get_gas_transport('N2', t_film)
    s = _convective_gas_sensitivity(config)
    mu = mu_ref * ((mu_g / max(mu_ref, 1e-18)) ** s)
    k = k_ref * ((k_g / max(k_ref, 1e-18)) ** s)
    cp_mass = cp_ref * ((cp_g / max(cp_ref, 1e-18)) ** s)

    m_ref = GAS_DATA['N2']['m'] / 1000.0
    m_g = gas['m'] / 1000.0
    m_kg_per_mol = m_ref * ((m_g / max(m_ref, 1e-18)) ** s)
    rho = (pressures_pa * m_kg_per_mol) / (R_UNIV * t_film)
    rho = np.maximum(rho, 1e-12)

    nu = mu / rho
    alpha = k / np.maximum(rho * cp_mass, 1e-12)
    pr = np.clip(nu / np.maximum(alpha, 1e-12), 0.2, 4.0)

    geom = config.get('geometry')
    orient = config.get('orientation', 'vertical')
    if geom == 'plates':
        l_char = max(float(config.get('gap', 1e-3)), 1e-6)
    else:
        r2 = float(config.get('enc_r', 8e-3))
        r1 = float(config.get('wire_r', 5e-6))
        l_char = max(r2 - r1, 5e-5)

    beta = 1.0 / t_film
    ra = G_STD * beta * d_t * (l_char ** 3) / np.maximum(nu * alpha, 1e-18)
    ra = np.clip(ra, 0.0, 1e12)

    if geom == 'cylindrical' and orient == 'horizontal':
        nu_nat = 0.36 + (0.518 * np.power(ra, 0.25)) / np.power(1.0 + np.power(0.559 / pr, 9.0 / 16.0), 4.0 / 9.0)
    else:
        nu_nat = 0.68 + (0.67 * np.power(ra, 0.25)) / np.power(1.0 + np.power(0.492 / pr, 9.0 / 16.0), 4.0 / 9.0)

    conv_strength = np.maximum(nu_nat - 1.0, 0.0)
    ra_on = max(float(config.get('conv_ra_on', 70.0)), 1e-9)
    ra_n = max(float(config.get('conv_transition_n', 1.25)), 0.4)
    ra_scale = np.power(np.maximum(ra, 0.0) / ra_on, ra_n)
    activation = ra_scale / (1.0 + ra_scale)

    if geom == 'square_cavity':
        geom_factor = 0.90
    elif geom == 'plates':
        geom_factor = 0.55
    else:
        geom_factor = 1.0

    gain = float(config.get('conv_gain', 0.22))
    mult = gain * geom_factor * conv_strength * activation * (1.0 + GRAVITY_Z_CONVECTION_GAIN)
    return np.clip(mult, 0.0, 1.4)

def calc_heat_flow(gas_key, config, p, aN2=0.6, t_hot_override=None, t_cold_override=None):
    """Calculate heat flow for a gas at pressure p using a gauge configuration."""
    gas = GAS_DATA[gas_key]
    accom_table = ACCOM_RATIOS_W if config['surface'] == 'W' else ACCOM_RATIOS_Si
    aE_ratio = accom_table.get(gas_key, 1.0)
    aE = min(aN2 * aE_ratio, 1.0)

    T1 = float(config['T1'] if t_hot_override is None else t_hot_override)
    T2 = float(config['T2'] if t_cold_override is None else t_cold_override)
    cbar = _mean_thermal_speed(gas_key, T2)
    visc_target_scale = _viscous_target_scale(config, aN2, T1, T2)

    if config['geometry'] in ('cylindrical', 'square_cavity'):
        r1 = config['wire_r']
        L = config['wire_L']
        r2 = config['enc_r']
        Q_mol = calc_Q_mol_cylinder(aE, gas['f'], cbar, r1, L, T1, T2, p)
        Q_visc_base = visc_target_scale * calc_Q_visc_cylinder(gas['gamma'], gas['plbar'], cbar, gas['f'], gas['m'], L, T1, T2, r1, r2)
        conv_mult = _convective_viscous_multiplier(gas_key, gas, config, p)
        Q_visc = Q_visc_base * (1.0 + conv_mult)
    else:  # plates
        A = config['plate_area']
        x = config['gap']
        aE2 = aE  # assume same on both surfaces for simplicity
        Q_mol = calc_Q_mol_plates(aE, aE2, gas['f'], cbar, A, T1, T2, p)
        Q_visc_base = visc_target_scale * calc_Q_visc_plates(gas['gamma'], gas['plbar'], cbar, gas['f'], gas['m'], A, T1, T2, x)
        conv_mult = _convective_viscous_multiplier(gas_key, gas, config, p)
        Q_visc = Q_visc_base * (1.0 + conv_mult)

    mol_fac, visc_fac = _shape_factors(config, p)
    Q_mol *= mol_fac
    Q_visc *= visc_fac
    return calc_Q_combined(Q_mol, Q_visc), Q_mol, Q_visc

def calc_heat_flow_vec(gas_key, config, pressures, aN2=0.6, t_hot_override=None, t_cold_override=None):
    """Vectorised heat flow: compute Q_combined, Q_mol, Q_visc for an array of pressures."""
    gas = GAS_DATA[gas_key]
    accom_table = ACCOM_RATIOS_W if config['surface'] == 'W' else ACCOM_RATIOS_Si
    aE_ratio = accom_table.get(gas_key, 1.0)
    aE = min(aN2 * aE_ratio, 1.0)
    T1 = float(config['T1'] if t_hot_override is None else t_hot_override)
    T2 = float(config['T2'] if t_cold_override is None else t_cold_override)
    f = gas['f']; cbar = _mean_thermal_speed(gas_key, T2); gamma = gas['gamma']
    plbar = gas['plbar']; m_amu = gas['m']
    m = m_amu * AMU_TO_KG
    lambda_p_pa = plbar * 100.0
    visc_target_scale = _viscous_target_scale(config, aN2, T1, T2)

    if config['geometry'] in ('cylindrical', 'square_cavity'):
        r1 = config['wire_r']; L = config['wire_L']; r2 = config['enc_r']
        A = 2 * np.pi * r1 * L
        Tx = T2
        Q_mol = aE * ((f + 1) / 8.0) * cbar * A * ((T1 - T2) / Tx) * pressures
        coeff = (9 * gamma - 5) / 4.0
        Q_visc_base = VISCOUS_GAIN * (
            coeff * (2 * np.pi * lambda_p_pa * cbar) * (f * kB / (2 * m)) * L * (T1 - T2) / np.log(r2 / r1)
        )
        Q_visc_base *= visc_target_scale
        conv_mult = _convective_viscous_multiplier_vec(gas_key, gas, config, pressures)
        Q_visc = np.full_like(pressures, Q_visc_base, dtype=np.float64) * (1.0 + conv_mult)
    else:
        A = config['plate_area']; x = config['gap']
        aE2 = aE
        aE_eff = (aE * aE2) / (aE + aE2 - aE * aE2)
        Tx = (T1 + T2) / 2.0
        Q_mol = aE_eff * ((f + 1) / 8.0) * cbar * A * ((T1 - T2) / Tx) * pressures
        coeff = (9 * gamma - 5) / 4.0
        Q_visc_base = VISCOUS_GAIN * (
            coeff * (2 * np.pi * lambda_p_pa * cbar) * (f * kB / (2 * m)) * A * (T1 - T2) / x
        )
        Q_visc_base *= visc_target_scale
        conv_mult = _convective_viscous_multiplier_vec(gas_key, gas, config, pressures)
        Q_visc = np.full_like(pressures, Q_visc_base, dtype=np.float64) * (1.0 + conv_mult)

    mol_fac, visc_fac = _shape_factors(config, pressures)
    Q_mol *= mol_fac
    Q_visc *= visc_fac

    # Combined (series-resistance analogy)
    denom = np.where((Q_mol != 0) & (Q_visc != 0), 1.0 / Q_mol + 1.0 / Q_visc, 1e30)
    Q_combined = 1.0 / denom
    return Q_combined, Q_mol, Q_visc


def calc_correction_factor_theory(gas_key, aN2=0.6, surface='W'):
    """Theoretical correction factor from molecular regime (Eq. 11)."""
    gas = GAS_DATA[gas_key]
    ref = GAS_DATA['N2']
    accom = ACCOM_RATIOS_W if surface == 'W' else ACCOM_RATIOS_Si
    aE_ratio = accom[gas_key] / accom['N2']
    ratio = aE_ratio * ((gas['f'] + 1) / (ref['f'] + 1)) * (gas['cbar'] / ref['cbar'])
    return 1.0 / ratio if ratio != 0 else float('inf')


# ══════════════════════════════════════════════════════════════════════════════
#  APP THEME STYLING
# ══════════════════════════════════════════════════════════════════════════════

FONT_FAMILY = 'Segoe UI'
FONT_MONO = 'Consolas'

COLOR_PALETTES = {
    'dark': {
        'bg':          '#1C1C1E',
        'bg_card':     '#2C2C2E',
        'bg_input':    '#1C1C1E',
        'bg_hover':    '#3A3A3C',
        'border':      '#38383A',
        'text':        '#F5F5F7',
        'text_dim':    '#86868B',
        'text_bright': '#FFFFFF',
        'accent':      '#0A84FF',
        'accent2':     '#30D158',
        'warn':        '#FF9F0A',
        'error':       '#FF453A',
    },
    'light': {
        'bg':          '#F5F5F7',
        'bg_card':     '#FFFFFF',
        'bg_input':    '#FFFFFF',
        'bg_hover':    '#E5E5EA',
        'border':      '#D1D1D6',
        'text':        '#1C1C1E',
        'text_dim':    '#6E6E73',
        'text_bright': '#000000',
        'accent':      '#0A84FF',
        'accent2':     '#30D158',
        'warn':        '#FF9F0A',
        'error':       '#FF3B30',
    },
}

COLORS = dict(COLOR_PALETTES['light'])


def _build_mpl_style():
    return {
        'figure.facecolor': COLORS['bg_card'],
        'axes.facecolor':   COLORS['bg_input'],
        'axes.edgecolor':   COLORS['border'],
        'axes.labelcolor':  COLORS['text'],
        'xtick.color':      COLORS['text_dim'],
        'ytick.color':      COLORS['text_dim'],
        'text.color':       COLORS['text'],
        'grid.color':       COLORS['border'],
        'grid.alpha':       0.4,
        'legend.facecolor': COLORS['bg_card'],
        'legend.edgecolor': COLORS['border'],
        'legend.labelcolor': COLORS['text'],
        'font.family':      'sans-serif',
        'font.sans-serif':  [FONT_FAMILY, 'Helvetica Neue', 'Arial'],
    }


MPL_STYLE = _build_mpl_style()


def _set_color_palette(mode='dark'):
    palette = COLOR_PALETTES.get(mode, COLOR_PALETTES['light'])
    COLORS.clear()
    COLORS.update(palette)
    MPL_STYLE.clear()
    MPL_STYLE.update(_build_mpl_style())


def apply_app_theme(root, mode='dark'):
    """Apply ttk and matplotlib colors for the selected app theme."""
    _set_color_palette(mode)
    style = ttk.Style(root)
    style.theme_use('clam')

    _f  = (FONT_FAMILY, 10)
    _fs = (FONT_FAMILY, 9)
    _fb = (FONT_FAMILY, 10, 'bold')
    _ft = (FONT_FAMILY, 14, 'bold')
    _fh = (FONT_FAMILY, 20, 'bold')
    _fm = (FONT_MONO, 9)

    style.configure('.', background=COLORS['bg'], foreground=COLORS['text'],
                    fieldbackground=COLORS['bg_input'], bordercolor=COLORS['border'],
                    troughcolor=COLORS['bg_input'], selectbackground=COLORS['accent'],
                    selectforeground='#FFFFFF', font=_f)

    style.configure('TNotebook', background=COLORS['bg'], borderwidth=0,
                    tabmargins=[4, 6, 4, 0])
    style.configure('TNotebook.Tab', background=COLORS['bg_card'],
                    foreground=COLORS['text_dim'],
                    padding=[16, 8], font=(FONT_FAMILY, 9, 'bold'))
    style.map('TNotebook.Tab',
              background=[('selected', COLORS['bg_hover']), ('active', COLORS['bg_card'])],
              foreground=[('selected', COLORS['text_bright']), ('active', COLORS['text'])])

    style.configure('TFrame', background=COLORS['bg'])
    style.configure('Card.TFrame', background=COLORS['bg_card'])

    style.configure('TLabel', background=COLORS['bg'], foreground=COLORS['text'], font=_f)
    style.configure('Card.TLabel', background=COLORS['bg_card'], font=_f)
    style.configure('Title.TLabel', font=_ft, foreground=COLORS['accent'],
                    background=COLORS['bg_card'])
    style.configure('Header.TLabel', font=_fh, foreground=COLORS['text_bright'],
                    background=COLORS['bg'])
    style.configure('Dim.TLabel', foreground=COLORS['text_dim'], font=_fs)
    style.configure('Big.TLabel', font=(FONT_FAMILY, 20, 'bold'),
                    foreground=COLORS['text_bright'], background=COLORS['bg_card'])
    style.configure('Accent.TLabel', foreground=COLORS['accent'],
                    background=COLORS['bg_card'], font=_fb)

    style.configure('TButton', background=COLORS['bg_hover'], foreground=COLORS['text'],
                    bordercolor=COLORS['border'], padding=[12, 6], font=_fs,
                    relief='flat')
    style.map('TButton',
              background=[('active', COLORS['accent']), ('pressed', COLORS['accent'])],
              foreground=[('active', '#FFFFFF'), ('pressed', '#FFFFFF')])
    style.configure('Active.TButton', background=COLORS['accent'],
                    foreground='#FFFFFF', bordercolor=COLORS['accent'])
    style.configure('Accent.TButton', background=COLORS['accent'],
                    foreground='#FFFFFF', font=_fb)

    style.configure('TCheckbutton', background=COLORS['bg_card'],
                    foreground=COLORS['text'], font=_f)
    style.map('TCheckbutton', background=[('active', COLORS['bg_card'])])

    style.configure('TRadiobutton', background=COLORS['bg_card'],
                    foreground=COLORS['text'], font=_f)
    style.map('TRadiobutton', background=[('active', COLORS['bg_card'])])

    style.configure('TScale', background=COLORS['bg_card'],
                    troughcolor=COLORS['bg_input'])
    style.configure('Horizontal.TScale', background=COLORS['bg_card'])

    style.configure('TCombobox', fieldbackground=COLORS['bg_input'],
                    background=COLORS['bg_card'], foreground=COLORS['text'],
                    bordercolor=COLORS['border'], arrowcolor=COLORS['accent'], font=_f)

    style.configure('Treeview', background=COLORS['bg_input'],
                    foreground=COLORS['text'], fieldbackground=COLORS['bg_input'],
                    borderwidth=0, font=_fm, rowheight=26)
    style.configure('Treeview.Heading', background=COLORS['bg_card'],
                    foreground=COLORS['accent'], font=(FONT_MONO, 9, 'bold'))
    style.map('Treeview',
              background=[('selected', COLORS['bg_hover'])],
              foreground=[('selected', COLORS['accent'])])

    style.configure('TLabelframe', background=COLORS['bg_card'],
                    foreground=COLORS['accent'], bordercolor=COLORS['border'],
                    relief='flat')
    style.configure('TLabelframe.Label', background=COLORS['bg_card'],
                    foreground=COLORS['accent'], font=_fb)

    style.configure('TEntry', fieldbackground=COLORS['bg_input'],
                    foreground=COLORS['text'], bordercolor=COLORS['border'],
                    insertcolor=COLORS['text'], font=_f)

    style.configure('TSeparator', background=COLORS['border'])

    root.configure(bg=COLORS['bg'])


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER WIDGET: LABELED SLIDER
# ══════════════════════════════════════════════════════════════════════════════

class LabeledSlider(ttk.Frame):
    """A slider with label, value display, and optional unit."""
    def __init__(self, parent, label, from_, to, initial, resolution=None, unit='',
                 fmt='{:.2f}', command=None, value_formatter=None, **kwargs):
        super().__init__(parent, style='Card.TFrame')
        self.fmt = fmt
        self.unit = unit
        self.command = command
        self.value_formatter = value_formatter

        self.var = tk.DoubleVar(value=initial)

        top = ttk.Frame(self, style='Card.TFrame')
        top.pack(fill='x')
        ttk.Label(top, text=label, style='Card.TLabel').pack(side='left')
        self.val_label = ttk.Label(top, text=self._format(initial), style='Accent.TLabel')
        self.val_label.pack(side='right')

        res = resolution if resolution else (to - from_) / 200.0
        self.scale = ttk.Scale(self, from_=from_, to=to, variable=self.var,
                               orient='horizontal', command=self._on_change)
        self.scale.pack(fill='x', padx=2, pady=(2, 4))

    def _format(self, val):
        if self.value_formatter is not None:
            return self.value_formatter(val)
        return self.fmt.format(val) + (' ' + self.unit if self.unit else '')

    def _on_change(self, val):
        v = self.var.get()
        self.val_label.config(text=self._format(v))
        if self.command:
            self.command(v)

    def get(self):
        return self.var.get()

    def set(self, val):
        self.var.set(val)
        self.val_label.config(text=self._format(val))

    def set_value_formatter(self, formatter):
        self.value_formatter = formatter
        self.val_label.config(text=self._format(self.var.get()))

    def set_range(self, from_, to):
        """Update slider bounds while preserving current value if possible."""
        lo = float(min(from_, to))
        hi = float(max(from_, to))
        self.scale.configure(from_=lo, to=hi)
        cur = float(self.var.get())
        if cur < lo:
            self.set(lo)
        elif cur > hi:
            self.set(hi)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER WIDGET: SCROLLABLE CONTROL PANEL
# ══════════════════════════════════════════════════════════════════════════════

class ScrollableControlPanel(ttk.Frame):
    """A vertically scrollable frame for control panels that may overflow on
    smaller / laptop screens.  Pack widgets into ``self.inner``."""

    def __init__(self, parent, width=274, **kwargs):
        super().__init__(parent, **kwargs)
        self._canvas = tk.Canvas(self, bg=COLORS['bg'], highlightthickness=0,
                                 width=width)
        self._vsb = ttk.Scrollbar(self, orient='vertical',
                                  command=self._canvas.yview)
        self.inner = ttk.Frame(self._canvas)

        self.inner.bind('<Configure>', self._on_inner_configure)
        self._win_id = self._canvas.create_window((0, 0), window=self.inner,
                                                   anchor='nw')
        self._canvas.configure(yscrollcommand=self._vsb.set)

        self._vsb.pack(side='right', fill='y')
        self._canvas.pack(side='left', fill='both', expand=True)
        self._canvas.bind('<Configure>', self._on_canvas_configure)
        self._canvas.bind('<Enter>', self._bind_mousewheel)
        self._canvas.bind('<Leave>', self._unbind_mousewheel)

    def _on_inner_configure(self, event):
        self._canvas.configure(scrollregion=self._canvas.bbox('all'))

    def _on_canvas_configure(self, event):
        self._canvas.itemconfig(self._win_id, width=event.width)

    def _bind_mousewheel(self, event):
        self._canvas.bind_all('<MouseWheel>', self._on_mousewheel)

    def _unbind_mousewheel(self, event):
        self._canvas.unbind_all('<MouseWheel>')

    def _on_mousewheel(self, event):
        self._canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')

    def on_theme_changed(self):
        self._canvas.configure(bg=COLORS['bg'])


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER WIDGET: PRESSURE UNIT SELECTOR
# ══════════════════════════════════════════════════════════════════════════════

class PressureUnitSelector(ttk.Frame):
    """Compact pressure unit toggle bound to the global APP_STATE variable."""

    def __init__(self, parent, on_change=None, **kwargs):
        super().__init__(parent, style='Card.TFrame', **kwargs)
        self._on_change = on_change
        ttk.Label(self, text='Unit:', style='Card.TLabel').pack(side='left', padx=(4, 4))
        var = APP_STATE.get('pressure_unit')
        if var is None:
            var = tk.StringVar(value='mbar')
        self._combo = ttk.Combobox(self, textvariable=var,
                                   values=list(PRESSURE_UNITS.keys()),
                                   state='readonly', width=6)
        self._combo.pack(side='left', padx=(0, 4))
        self._combo.bind('<<ComboboxSelected>>', self._changed)

    def _changed(self, event=None):
        if self._on_change:
            self._on_change()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB: HEAT TRANSFER 2D SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════

class Simulator2DTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.gas_vars = {}
        self._update_job = None
        self._build_ui()
        self._on_config_change()
        self._sync_temperature_unit_display()
        self._update_plot()

    def _build_ui(self):
        # Left panel: scrollable controls
        self._scroll_panel = ScrollableControlPanel(self, width=270)
        self._scroll_panel.pack(side='left', fill='y', padx=(6, 0), pady=6)
        ctrl_frame = self._scroll_panel.inner

        # Configuration selector
        cfg_frame = ttk.LabelFrame(ctrl_frame, text=' Gauge Configuration ')
        cfg_frame.pack(fill='x', pady=(0, 8))

        self.config_var = tk.StringVar(value='psg55x')
        for key, cfg in GAUGE_CONFIGS.items():
            ttk.Radiobutton(cfg_frame, text=cfg['name'], variable=self.config_var,
                            value=key, command=self._on_config_change,
                            style='TCheckbutton').pack(anchor='w', padx=8, pady=2)

        # Gas selector
        gas_frame = ttk.LabelFrame(ctrl_frame, text=' Gas Species ')
        gas_frame.pack(fill='x', pady=(0, 8))

        self.gas_vars = {}
        for i, (key, gas) in enumerate(GAS_DATA.items()):
            var = tk.BooleanVar(value=(key in ['N2', 'He', 'Ar', 'Xe']))
            cb = ttk.Checkbutton(gas_frame, text=f"{gas['symbol']} ({gas['name']})",
                                 variable=var, command=self._schedule_update,
                                 style='TCheckbutton')
            cb.pack(anchor='w', padx=8, pady=1)
            self.gas_vars[key] = var

        # Parameter sliders
        param_frame = ttk.LabelFrame(ctrl_frame, text=' Parameters ')
        param_frame.pack(fill='x', pady=(0, 8))

        self.sl_T1 = LabeledSlider(param_frame, 'Wire Temp T₁', 313, 573, 393,
                                              fmt='{:.0f}', command=lambda v: self._schedule_update())
        self.sl_T1.pack(fill='x', padx=6, pady=2)

        self.sl_T2 = LabeledSlider(param_frame, 'Enclosure Temp T₂', 273, 353, 296,
                                              fmt='{:.0f}', command=lambda v: self._schedule_update())
        self.sl_T2.pack(fill='x', padx=6, pady=2)

        self.sl_aN2 = LabeledSlider(param_frame, 'a_N₂ (accommodation)', 0.1, 1.0, 0.6,
                                                fmt='{:.2f}', command=lambda v: self._schedule_update())
        self.sl_aN2.pack(fill='x', padx=6, pady=2)

        self.sl_wire_r = LabeledSlider(param_frame, 'Wire radius', 1, 50, 5,
                                                    unit='μm', fmt='{:.1f}', command=lambda v: self._schedule_update())
        self.sl_wire_r.pack(fill='x', padx=6, pady=2)

        self.sl_wire_L = LabeledSlider(param_frame, 'Wire length', 0.5, 20, 5,
                                                    unit='cm', fmt='{:.1f}', command=lambda v: self._schedule_update())
        self.sl_wire_L.pack(fill='x', padx=6, pady=2)

        self.sl_enc_r = LabeledSlider(param_frame, 'Enclosure radius', 2, 30, 8,
                                                  unit='mm', fmt='{:.1f}', command=lambda v: self._schedule_update())
        self.sl_enc_r.pack(fill='x', padx=6, pady=2)

        # Display options
        opt_frame = ttk.LabelFrame(ctrl_frame, text=' Display ')
        opt_frame.pack(fill='x', pady=(0, 8))

        self.log_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt_frame, text='Logarithmic X-axis', variable=self.log_var,
                        command=self._schedule_update).pack(anchor='w', padx=8, pady=2)

        self.show_regimes_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_frame, text='Show mol/visc regimes', variable=self.show_regimes_var,
                        command=self._schedule_update).pack(anchor='w', padx=8, pady=2)

        self.show_p0_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_frame, text='Show total Q_el (with p₀)', variable=self.show_p0_var,
                        command=self._schedule_update).pack(anchor='w', padx=8, pady=2)

        plot_frame = ttk.Frame(self)
        plot_frame.pack(side='right', fill='both', expand=True, padx=8, pady=8)

        with plt.rc_context(MPL_STYLE):
            self.fig = Figure(figsize=(9, 6), dpi=100)
            self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()
        toolbar.pack(fill='x')

    def _get_config(self):
        key = self.config_var.get()
        cfg = dict(GAUGE_CONFIGS[key])
        cfg['T1'] = self.sl_T1.get()
        cfg['T2'] = self.sl_T2.get()
        if cfg['geometry'] in ('cylindrical', 'square_cavity'):
            cfg['wire_r'] = self.sl_wire_r.get() * 1e-6
            cfg['wire_L'] = self.sl_wire_L.get() * 1e-2
            cfg['enc_r'] = self.sl_enc_r.get() * 1e-3
        else:
            cfg['plate_area'] = (self.sl_wire_L.get() * 1e-2) * (self.sl_wire_r.get() * 1e-6 * 200)
            cfg['gap'] = self.sl_enc_r.get() * 1e-3
        return cfg

    def _on_config_change(self):
        key = self.config_var.get()
        cfg = GAUGE_CONFIGS[key]
        t1_lo, t1_hi = cfg.get('t_hot_range_k', (313.0, 573.0))
        t2_lo, t2_hi = cfg.get('t_cold_range_k', (273.0, 353.0))
        self.sl_T1.set_range(t1_lo, t1_hi)
        self.sl_T2.set_range(t2_lo, t2_hi)
        self.sl_T1.set(cfg['T1'])
        self.sl_T2.set(cfg['T2'])
        if cfg['geometry'] in ('cylindrical', 'square_cavity'):
            self.sl_wire_r.set(cfg['wire_r'] * 1e6)
            self.sl_wire_L.set(cfg['wire_L'] * 1e2)
            self.sl_enc_r.set(cfg['enc_r'] * 1e3)
        else:
            self.sl_wire_r.set(25)
            self.sl_wire_L.set(1)
            self.sl_enc_r.set(cfg.get('gap', 1e-3) * 1e3)
        self._schedule_update()

    def _schedule_update(self, *args):
        if self._update_job is not None:
            self.after_cancel(self._update_job)
        self._update_job = self.after(80, self._update_plot)

    def _sync_temperature_unit_display(self):
        t_unit = get_temperature_unit()
        self.sl_T1.set_value_formatter(lambda v: format_temperature(v, t_unit, fmt='{:.1f}'))
        self.sl_T2.set_value_formatter(lambda v: format_temperature(v, t_unit, fmt='{:.1f}'))

    def on_global_units_changed(self):
        self._sync_temperature_unit_display()
        self._schedule_update()

    def on_theme_changed(self):
        self.fig.patch.set_facecolor(COLORS['bg_card'])

    def _update_plot(self, *args):
        self._update_job = None
        with plt.rc_context(MPL_STYLE):
            self.ax.clear()
            self.ax.set_facecolor(COLORS['bg_input'])
            self.fig.patch.set_facecolor(COLORS['bg_card'])
            cfg = self._get_config()
            aN2 = self.sl_aN2.get()
            pressures = np.logspace(-2, 5, 300)

            unit = get_pressure_unit()
            uf = PRESSURE_UNITS[unit]['factor']
            p_display = pressures * uf

            selected = [k for k, v in self.gas_vars.items() if v.get()]
            show_regimes = self.show_regimes_var.get()
            show_p0 = self.show_p0_var.get()

            for gk in selected:
                gas = GAS_DATA[gk]
                Qs, Qms, Qvs = calc_heat_flow_vec(gk, cfg, pressures, aN2)
                Qs_mw = Qs * 1000
                Qms_mw = Qms * 1000
                Qvs_mw = Qvs * 1000

                self.ax.plot(p_display, Qs_mw, color=gas['color'], linewidth=2, label=gas['symbol'])

                if show_regimes:
                    self.ax.plot(p_display, Qms_mw, color=gas['color'], linewidth=1, linestyle='--', alpha=0.4)
                    self.ax.plot(p_display, Qvs_mw, color=gas['color'], linewidth=1, linestyle=':', alpha=0.4)

            if show_p0:
                Q_n2, _, _ = calc_heat_flow_vec('N2', cfg, pressures, aN2)
                Q_n2_mw = Q_n2 * 1000 + 0.01 * 1000
                self.ax.plot(p_display, Q_n2_mw, color='white', linewidth=1.5, linestyle='-.',
                             alpha=0.5, label='Q_el (N₂+p₀)')

            if self.log_var.get():
                self.ax.set_xscale('log')

            self.ax.set_xlabel(f'Pressure ({unit})', fontsize=11)
            self.ax.set_ylabel('Heat Flow (mW)', fontsize=11)
            self.ax.set_title(f'Gas Heat Transfer — {GAUGE_CONFIGS[self.config_var.get()]["name"]}',
                              fontsize=12, color=COLORS['text_bright'], pad=10)
            leg = self.ax.legend(fontsize=9, loc='upper left')
            if leg is not None:
                leg.get_frame().set_facecolor(COLORS['bg_card'])
                leg.get_frame().set_edgecolor(COLORS['border'])
                for txt in leg.get_texts():
                    txt.set_color(COLORS['text'])
            self.ax.grid(True, alpha=0.3)
            self.ax.tick_params(colors=COLORS['text_dim'])
            for spine in self.ax.spines.values():
                spine.set_color(COLORS['border'])

            if show_regimes:
                regime_txt = '—— combined  - - molecular  ···· viscous(+conv)' if cfg['geometry'] == 'cylindrical' else '—— combined  - - molecular  ···· viscous'
                self.ax.text(0.98, 0.02, regime_txt,
                             transform=self.ax.transAxes, ha='right', fontsize=8,
                             color=COLORS['text_dim'])

            self.fig.tight_layout()
            self.canvas.draw_idle()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB: 3D SURFACE SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════

class Simulator3DTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self._update_job = None
        self._build_ui()
        self._update_plot()

    def _build_ui(self):
        self._scroll_panel = ScrollableControlPanel(self, width=270)
        self._scroll_panel.pack(side='left', fill='y', padx=(6, 0), pady=6)
        ctrl_frame = self._scroll_panel.inner

        # Mode selector
        mode_frame = ttk.LabelFrame(ctrl_frame, text=' 3D Surface Plot Mode ')
        mode_frame.pack(fill='x', pady=(0, 8))

        self.mode_var = tk.StringVar(value='p_aE')
        modes = [
            ('p_aE', 'Pressure × Accommodation → Q'),
            ('p_T', 'Pressure × Wire Temp → Q'),
            ('p_gap', 'Pressure × Gap Size → Q'),
            ('gas_compare', 'Gas Species × Pressure → CF'),
            ('mol_vs_visc', 'Molecular vs Viscous (3D)'),
        ]
        for val, text in modes:
            ttk.Radiobutton(mode_frame, text=text, variable=self.mode_var,
                            value=val, command=self._schedule_update,
                            style='TCheckbutton').pack(anchor='w', padx=8, pady=2)

        # Gas selector for 3D
        gas_frame = ttk.LabelFrame(ctrl_frame, text=' Primary Gas ')
        gas_frame.pack(fill='x', pady=(0, 8))

        self.gas_3d_var = tk.StringVar(value='Ar')
        for key, gas in GAS_DATA.items():
            ttk.Radiobutton(gas_frame, text=f"{gas['symbol']}", variable=self.gas_3d_var,
                            value=key, command=self._schedule_update,
                            style='TCheckbutton').pack(anchor='w', padx=8, pady=1)

        # Config selector
        cfg_frame = ttk.LabelFrame(ctrl_frame, text=' Configuration ')
        cfg_frame.pack(fill='x', pady=(0, 8))

        self.cfg_3d_var = tk.StringVar(value='psg55x')
        for key, cfg in GAUGE_CONFIGS.items():
            ttk.Radiobutton(cfg_frame, text=cfg['name'], variable=self.cfg_3d_var,
                            value=key, command=self._schedule_update,
                            style='TCheckbutton').pack(anchor='w', padx=8, pady=2)

        # Appearance
        app_frame = ttk.LabelFrame(ctrl_frame, text=' Appearance ')
        app_frame.pack(fill='x', pady=(0, 8))

        self.cmap_var = tk.StringVar(value='viridis')
        ttk.Label(app_frame, text='Colormap:').pack(anchor='w', padx=8)
        cmaps = ['viridis', 'plasma', 'inferno', 'coolwarm', 'turbo', 'magma']
        cm_combo = ttk.Combobox(app_frame, textvariable=self.cmap_var, values=cmaps,
                                state='readonly', width=12)
        cm_combo.pack(padx=8, pady=2, anchor='w')
        cm_combo.bind('<<ComboboxSelected>>', lambda e: self._schedule_update())

        self.wireframe_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(app_frame, text='Wireframe overlay', variable=self.wireframe_var,
                        command=self._schedule_update).pack(anchor='w', padx=8, pady=2)

        ttk.Button(ctrl_frame, text='↻ Refresh Plot', command=self._schedule_update).pack(fill='x', pady=4)

        # Plot area
        plot_frame = ttk.Frame(self)
        plot_frame.pack(side='right', fill='both', expand=True, padx=8, pady=8)

        with plt.rc_context(MPL_STYLE):
            self.fig = Figure(figsize=(9, 6), dpi=100)
            self.ax = self.fig.add_subplot(111, projection='3d')

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()
        toolbar.pack(fill='x')

    def _schedule_update(self, *args):
        if self._update_job is not None:
            self.after_cancel(self._update_job)
        self._update_job = self.after(90, self._update_plot)

    def on_theme_changed(self):
        self.fig.patch.set_facecolor(COLORS['bg_card'])

    def _update_plot(self, *args):
        self._update_job = None
        with plt.rc_context(MPL_STYLE):
            self.fig.clear()
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.set_facecolor(COLORS['bg_input'])
            self.fig.patch.set_facecolor(COLORS['bg_card'])

            mode = self.mode_var.get()
            gas_key = self.gas_3d_var.get()
            cfg_key = self.cfg_3d_var.get()
            cfg = dict(GAUGE_CONFIGS[cfg_key])
            cmap_name = self.cmap_var.get()
            wireframe = self.wireframe_var.get()

            gas = GAS_DATA[gas_key]

            if mode == 'p_aE':
                self._plot_pressure_accommodation(cfg, gas_key, cmap_name, wireframe)
            elif mode == 'p_T':
                self._plot_pressure_temperature(cfg, gas_key, cmap_name, wireframe)
            elif mode == 'p_gap':
                self._plot_pressure_gap(cfg, gas_key, cmap_name, wireframe)
            elif mode == 'gas_compare':
                self._plot_gas_comparison(cfg, cmap_name)
            elif mode == 'mol_vs_visc':
                self._plot_mol_vs_visc(cfg, gas_key, cmap_name, wireframe)

            try:
                self.ax.xaxis.pane.fill = False
                self.ax.yaxis.pane.fill = False
                self.ax.zaxis.pane.fill = False
                self.ax.xaxis.pane.set_edgecolor(COLORS['border'])
                self.ax.yaxis.pane.set_edgecolor(COLORS['border'])
                self.ax.zaxis.pane.set_edgecolor(COLORS['border'])
                self.ax.tick_params(colors=COLORS['text_dim'], labelsize=8)
            except Exception:
                pass

            self.fig.tight_layout()
            self.canvas.draw_idle()

    def _plot_pressure_accommodation(self, cfg, gas_key, cmap_name, wireframe):
        P = np.logspace(-1, 4, 36)
        A = np.linspace(0.1, 1.0, 36)
        PP, AA = np.meshgrid(np.log10(P), A)
        Z = np.zeros_like(PP)

        for i in range(len(A)):
            Qs, _, _ = calc_heat_flow_vec(gas_key, cfg, P, A[i])
            Z[i, :] = Qs * 1000

        surf = self.ax.plot_surface(PP, AA, Z, cmap=cmap_name, alpha=0.85, linewidth=0)
        if wireframe:
            self.ax.plot_wireframe(PP, AA, Z, color=COLORS['text_dim'], linewidth=0.3, alpha=0.3)

        self.ax.set_xlabel('log₁₀(Pressure / Pa)', fontsize=9, labelpad=10)
        self.ax.set_ylabel('a_N₂', fontsize=9, labelpad=10)
        self.ax.set_zlabel('Q (mW)', fontsize=9, labelpad=10)
        self.ax.set_title(f'{GAS_DATA[gas_key]["symbol"]} — Heat Flow vs Pressure & Accommodation',
                          fontsize=11, color=COLORS['text_bright'], pad=15)
        self.fig.colorbar(surf, ax=self.ax, shrink=0.5, aspect=15, label='Q (mW)')

    def _plot_pressure_temperature(self, cfg, gas_key, cmap_name, wireframe):
        P = np.logspace(-1, 4, 36)
        T = np.linspace(313, 573, 36)
        PP, TT = np.meshgrid(np.log10(P), T)
        Z = np.zeros_like(PP)

        for i in range(len(T)):
            cfg_t = dict(cfg)
            cfg_t['T1'] = T[i]
            Qs, _, _ = calc_heat_flow_vec(gas_key, cfg_t, P, 0.6)
            Z[i, :] = Qs * 1000

        surf = self.ax.plot_surface(PP, TT, Z, cmap=cmap_name, alpha=0.85)
        if wireframe:
            self.ax.plot_wireframe(PP, TT, Z, color=COLORS['text_dim'], linewidth=0.3, alpha=0.3)

        self.ax.set_xlabel('log₁₀(P / Pa)', fontsize=9, labelpad=10)
        self.ax.set_ylabel('T₁ (K)', fontsize=9, labelpad=10)
        self.ax.set_zlabel('Q (mW)', fontsize=9, labelpad=10)
        self.ax.set_title(f'{GAS_DATA[gas_key]["symbol"]} — Heat Flow vs Pressure & Wire Temperature',
                          fontsize=11, color=COLORS['text_bright'], pad=15)
        self.fig.colorbar(surf, ax=self.ax, shrink=0.5, aspect=15, label='Q (mW)')

    def _plot_pressure_gap(self, cfg, gas_key, cmap_name, wireframe):
        P = np.logspace(-1, 4, 36)
        if cfg['geometry'] in ('cylindrical', 'square_cavity'):
            G = np.linspace(2, 30, 36)  # enclosure radius in mm
        else:
            G = np.linspace(0.002, 5, 36)  # gap in mm
        PP, GG = np.meshgrid(np.log10(P), G)
        Z = np.zeros_like(PP)

        for i in range(len(G)):
            cfg_g = dict(cfg)
            if cfg['geometry'] in ('cylindrical', 'square_cavity'):
                cfg_g['enc_r'] = G[i] * 1e-3
            else:
                cfg_g['gap'] = G[i] * 1e-3
            Qs, _, _ = calc_heat_flow_vec(gas_key, cfg_g, P, 0.6)
            Z[i, :] = Qs * 1000

        surf = self.ax.plot_surface(PP, GG, Z, cmap=cmap_name, alpha=0.85)
        if wireframe:
            self.ax.plot_wireframe(PP, GG, Z, color=COLORS['text_dim'], linewidth=0.3, alpha=0.3)

        gap_label = 'Chamber Half-Size (mm)' if cfg['geometry'] in ('cylindrical', 'square_cavity') else 'Plate Gap (mm)'
        self.ax.set_xlabel('log₁₀(P / Pa)', fontsize=9, labelpad=10)
        self.ax.set_ylabel(gap_label, fontsize=9, labelpad=10)
        self.ax.set_zlabel('Q (mW)', fontsize=9, labelpad=10)
        self.ax.set_title(f'{GAS_DATA[gas_key]["symbol"]} — Heat Flow vs Pressure & Geometry',
                          fontsize=11, color=COLORS['text_bright'], pad=15)
        self.fig.colorbar(surf, ax=self.ax, shrink=0.5, aspect=15, label='Q (mW)')

    def _plot_gas_comparison(self, cfg, cmap_name):
        gases = list(GAS_DATA.keys())
        P = np.logspace(-1, 4, 60)
        Z = np.zeros((len(gases), len(P)))

        Qn2_all, _, _ = calc_heat_flow_vec('N2', cfg, P, 0.6)
        for i, gk in enumerate(gases):
            Qx_all, _, _ = calc_heat_flow_vec(gk, cfg, P, 0.6)
            safe = np.where(Qx_all > 0, Qx_all, 1e-30)
            Z[i, :] = np.where(Qx_all > 0, Qn2_all / safe, 0)

        X, Y = np.meshgrid(np.log10(P), np.arange(len(gases)))
        colors = [GAS_DATA[g]['color'] for g in gases]

        for i, gk in enumerate(gases):
            self.ax.plot(np.log10(P), [i]*len(P), Z[i, :],
                         color=GAS_DATA[gk]['color'], linewidth=2.5, label=GAS_DATA[gk]['symbol'])

        self.ax.set_xlabel('log₁₀(P / Pa)', fontsize=9, labelpad=10)
        self.ax.set_ylabel('Gas Species', fontsize=9, labelpad=10)
        self.ax.set_zlabel('CF_X/N₂', fontsize=9, labelpad=10)
        self.ax.set_yticks(range(len(gases)))
        self.ax.set_yticklabels([GAS_DATA[g]['symbol'] for g in gases], fontsize=7)
        self.ax.set_title('Correction Factors — All Gases vs Pressure',
                          fontsize=11, color=COLORS['text_bright'], pad=15)
        self.ax.legend(fontsize=7, loc='upper left')

    def _plot_mol_vs_visc(self, cfg, gas_key, cmap_name, wireframe):
        P = np.logspace(-1, 5, 80)
        gas = GAS_DATA[gas_key]
        Q_arr, Qm_arr, Qv_arr = calc_heat_flow_vec(gas_key, cfg, P, 0.6)
        Qs = (Q_arr * 1000).tolist()
        Qms = (Qm_arr * 1000).tolist()
        Qvs = (Qv_arr * 1000).tolist()

        logP = np.log10(P)
        zeros = np.zeros_like(logP)

        self.ax.plot(logP, zeros, Qms, color='#4fc3f7', linewidth=2.5, label='Molecular (Q_mol)')
        self.ax.plot(logP, np.ones_like(logP), Qvs, color='#e67e22', linewidth=2.5, label='Viscous (Q_visc)')
        self.ax.plot(logP, np.ones_like(logP)*2, Qs, color='#6bcb77', linewidth=3, label='Combined')

        # Fill surfaces
        for i in range(len(logP)-1):
            verts = [[logP[i], 0, 0], [logP[i], 0, Qms[i]],
                     [logP[i+1], 0, Qms[i+1]], [logP[i+1], 0, 0]]
            self.ax.plot_surface(
                np.array([[logP[i], logP[i+1]], [logP[i], logP[i+1]]]),
                np.array([[0, 0], [0, 0]]),
                np.array([[0, 0], [Qms[i], Qms[i+1]]]),
                color='#4fc3f7', alpha=0.1
            )

        self.ax.set_xlabel('log₁₀(P / Pa)', fontsize=9, labelpad=10)
        self.ax.set_ylabel('Regime', fontsize=9, labelpad=10)
        self.ax.set_zlabel('Q (mW)', fontsize=9, labelpad=10)
        self.ax.set_yticks([0, 1, 2])
        self.ax.set_yticklabels(['Molecular', 'Viscous', 'Combined'], fontsize=7)
        self.ax.set_title(f'{gas["symbol"]} — Molecular vs Viscous Regime Breakdown',
                          fontsize=11, color=COLORS['text_bright'], pad=15)
        self.ax.legend(fontsize=8, loc='upper left')


# ══════════════════════════════════════════════════════════════════════════════
#  TAB: CORRECTION FACTORS
# ══════════════════════════════════════════════════════════════════════════════

class CorrectionFactorsTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self._build_ui()
        self._update_plot()

    def _build_ui(self):
        self._scroll_panel = ScrollableControlPanel(self, width=270)
        self._scroll_panel.pack(side='left', fill='y', padx=(6, 0), pady=6)
        ctrl_frame = self._scroll_panel.inner

        mode_frame = ttk.LabelFrame(ctrl_frame, text=' View Mode ')
        mode_frame.pack(fill='x', pady=(0, 8))

        self.cf_mode = tk.StringVar(value='bar')
        modes = [
            ('bar', 'Bar Chart Comparison'),
            ('true_vs_ind', 'True vs Indicated (Fig 4)'),
            ('theory_vs_exp', 'Theory vs Experiment'),
            ('spread', 'Gauge-to-Gauge Spread'),
        ]
        for val, text in modes:
            ttk.Radiobutton(mode_frame, text=text, variable=self.cf_mode,
                            value=val, command=self._update_plot,
                            style='TCheckbutton').pack(anchor='w', padx=8, pady=2)

        # Data table
        table_frame = ttk.LabelFrame(ctrl_frame, text=' Correction Factor Data (Table IX) ')
        table_frame.pack(fill='x', pady=(0, 8))

        cols = ('gas', 'range', 'mean', 'spread')
        self.tree = ttk.Treeview(table_frame, columns=cols, show='headings', height=9)
        self.tree.heading('gas', text='Gas')
        self.tree.heading('range', text='Range (Pa)')
        self.tree.heading('mean', text='Mean CF')
        self.tree.heading('spread', text='σ')
        self.tree.column('gas', width=50)
        self.tree.column('range', width=80)
        self.tree.column('mean', width=60)
        self.tree.column('spread', width=50)
        self.tree.pack(padx=4, pady=4)

        for key, d in EXPERIMENTAL_CF.items():
            self.tree.insert('', 'end', values=(
                GAS_DATA[key]['symbol'], d['range'], f"{d['mean']:.2f}", f"±{d['spread']:.2f}"))

        # Plot area
        plot_frame = ttk.Frame(self)
        plot_frame.pack(side='right', fill='both', expand=True, padx=8, pady=8)

        with plt.rc_context(MPL_STYLE):
            self.fig = Figure(figsize=(9, 6), dpi=100)
            self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()
        toolbar.pack(fill='x')

    def on_theme_changed(self):
        self.fig.patch.set_facecolor(COLORS['bg_card'])

    def _update_plot(self, *args):
        with plt.rc_context(MPL_STYLE):
            self.ax.clear()
            self.ax.set_facecolor(COLORS['bg_input'])
            self.fig.patch.set_facecolor(COLORS['bg_card'])
            mode = self.cf_mode.get()

            if mode == 'bar':
                self._plot_bar()
            elif mode == 'true_vs_ind':
                self._plot_true_vs_indicated()
            elif mode == 'theory_vs_exp':
                self._plot_theory_vs_exp()
            elif mode == 'spread':
                self._plot_spread()

            self.ax.tick_params(colors=COLORS['text_dim'])
            for spine in self.ax.spines.values():
                spine.set_color(COLORS['border'])

            leg = self.ax.get_legend()
            if leg is not None:
                leg.get_frame().set_facecolor(COLORS['bg_card'])
                leg.get_frame().set_edgecolor(COLORS['border'])
                for txt in leg.get_texts():
                    txt.set_color(COLORS['text'])

            self.fig.tight_layout()
            self.canvas.draw_idle()

    def _plot_bar(self):
        gases = [k for k in EXPERIMENTAL_CF if k != 'N2']
        means = [EXPERIMENTAL_CF[k]['mean'] for k in gases]
        vm3s = [EXPERIMENTAL_CF[k]['vm3'] for k in gases]
        spreads = [EXPERIMENTAL_CF[k]['spread'] for k in gases]
        colors = [GAS_DATA[k]['color'] for k in gases]
        labels = [GAS_DATA[k]['symbol'] for k in gases]

        x = np.arange(len(gases))
        w = 0.35
        bars1 = self.ax.bar(x - w/2, means, w, color=colors, alpha=0.85, label='Mean (all gauges)',
                            yerr=spreads, capsize=4, error_kw={'color': COLORS['text_dim'], 'linewidth': 1})
        bars2 = self.ax.bar(x + w/2, vm3s, w, color=colors, alpha=0.4, label='VM3 (MEMS)')

        self.ax.axhline(y=1.0, color=COLORS['accent'], linestyle='--', alpha=0.5, label='N₂ = 1.0')
        self.ax.set_xticks(x)
        self.ax.set_xticklabels(labels)
        self.ax.set_ylabel('CF_X/N₂', fontsize=11)
        self.ax.set_title('Normalized Correction Factors', fontsize=12, color=COLORS['text_bright'])
        self.ax.legend(fontsize=9)
        self.ax.grid(True, axis='y', alpha=0.3)

    def _plot_true_vs_indicated(self):
        pressures = np.logspace(-1, 5, 200)
        for key, d in EXPERIMENTAL_CF.items():
            cf = d['mean']
            p_ind = pressures / cf
            self.ax.loglog(p_ind, pressures, color=GAS_DATA[key]['color'],
                           linewidth=2 if key == 'N2' else 1.5,
                           linestyle='-' if key == 'N2' else '--',
                           label=GAS_DATA[key]['symbol'])

        self.ax.set_xlabel('Indicated Pressure (Pa)', fontsize=11)
        self.ax.set_ylabel('True Pressure (Pa)', fontsize=11)
        self.ax.set_title('True vs Indicated Pressure (cf. Fig. 4)', fontsize=12,
                          color=COLORS['text_bright'])
        self.ax.legend(fontsize=8, ncol=3, loc='upper left')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlim(0.01, 1e5)
        self.ax.set_ylim(0.1, 1e4)

        # Annotation regions
        self.ax.fill_between([0.01, 1e5], [0.01, 1e5], [0.1, 1e6],
                             alpha=0.03, color=COLORS['accent'])
        self.ax.text(50, 5000, 'p_ind < p_true\n(underread)', fontsize=8,
                     color=COLORS['text_dim'], ha='center')
        self.ax.text(5000, 50, 'p_ind > p_true\n(overread)', fontsize=8,
                     color=COLORS['text_dim'], ha='center')

    def _plot_theory_vs_exp(self):
        gases = [k for k in EXPERIMENTAL_CF if k != 'N2']
        exp = [EXPERIMENTAL_CF[k]['mean'] for k in gases]
        theory_mol = [calc_correction_factor_theory(k, 0.6, 'W') for k in gases]
        colors = [GAS_DATA[k]['color'] for k in gases]
        labels = [GAS_DATA[k]['symbol'] for k in gases]

        self.ax.scatter(exp, theory_mol, c=colors, s=120, zorder=5,
                edgecolors=COLORS['text_bright'], linewidths=1.5)
        for i, lbl in enumerate(labels):
            self.ax.annotate(lbl, (exp[i], theory_mol[i]), textcoords='offset points',
                             xytext=(8, 8), fontsize=9, color=colors[i])

        lim = [0, max(max(exp), max(theory_mol)) * 1.1 + 0.2]
        self.ax.plot(lim, lim, '--', color=COLORS['text_dim'], alpha=0.5, label='Perfect agreement')
        self.ax.set_xlabel('Experimental CF (mean)', fontsize=11)
        self.ax.set_ylabel('Theoretical CF (Eq. 11, a_N₂=0.6)', fontsize=11)
        self.ax.set_title('Theory vs Experiment — Correction Factors', fontsize=12,
                          color=COLORS['text_bright'])
        self.ax.legend(fontsize=9)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')

    def _plot_spread(self):
        gases = [k for k in EXPERIMENTAL_CF if k != 'N2']
        labels = [GAS_DATA[k]['symbol'] for k in gases]
        colors = [GAS_DATA[k]['color'] for k in gases]

        for i, k in enumerate(gases):
            d = EXPERIMENTAL_CF[k]
            self.ax.barh(i, d['cfMax'] - d['cfMin'], left=d['cfMin'],
                         color=GAS_DATA[k]['color'], alpha=0.6, height=0.6)
            self.ax.plot(d['mean'], i, 'o', color=COLORS['text_bright'], markersize=8, zorder=5)
            self.ax.plot(d['vm3'], i, 's', color=GAS_DATA[k]['color'], markersize=8,
                         markeredgecolor=COLORS['text_bright'], markeredgewidth=1.5, zorder=5)

        self.ax.set_yticks(range(len(gases)))
        self.ax.set_yticklabels(labels)
        self.ax.set_xlabel('CF_X/N₂', fontsize=11)
        self.ax.set_title('Gauge-to-Gauge Spread of Correction Factors', fontsize=12,
                          color=COLORS['text_bright'])
        self.ax.axvline(x=1.0, color=COLORS['accent'], linestyle='--', alpha=0.5)
        self.ax.grid(True, axis='x', alpha=0.3)

        from matplotlib.lines import Line2D
        legend_elements = [
             Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['text_bright'],
                 markeredgecolor=COLORS['text_bright'], markersize=8, label='Mean (all gauges)'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['accent'], markersize=8,
                 markeredgecolor=COLORS['text_bright'], label='VM3 (MEMS)'),
        ]
        self.ax.legend(handles=legend_elements, fontsize=9, loc='lower right')


# ══════════════════════════════════════════════════════════════════════════════
#  TAB: ACCOMMODATION COEFFICIENTS
# ══════════════════════════════════════════════════════════════════════════════

class AccommodationTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self._build_ui()
        self._update_plot()

    def _build_ui(self):
        self._scroll_panel = ScrollableControlPanel(self, width=270)
        self._scroll_panel.pack(side='left', fill='y', padx=(6, 0), pady=6)
        ctrl_frame = self._scroll_panel.inner

        surf_frame = ttk.LabelFrame(ctrl_frame, text=' Surface Material ')
        surf_frame.pack(fill='x', pady=(0, 8))

        self.surface_var = tk.StringVar(value='W')
        ttk.Radiobutton(surf_frame, text='Tungsten (oxidized)', variable=self.surface_var,
                        value='W', command=self._update_plot, style='TCheckbutton').pack(anchor='w', padx=8, pady=2)
        ttk.Radiobutton(surf_frame, text='Silicon (MEMS)', variable=self.surface_var,
                        value='Si', command=self._update_plot, style='TCheckbutton').pack(anchor='w', padx=8, pady=2)

        param_frame = ttk.LabelFrame(ctrl_frame, text=' Reference a_N₂ ')
        param_frame.pack(fill='x', pady=(0, 8))

        self.sl_aN2_accom = LabeledSlider(param_frame, 'a_N₂', 0.1, 1.0, 0.6,
                                          fmt='{:.2f}', command=lambda v: self._update_plot())
        self.sl_aN2_accom.pack(fill='x', padx=6, pady=2)

        info = ttk.Label(param_frame, text='Upper limit (W): 0.86\nUpper limit (Si): 0.76',
                         style='Dim.TLabel', wraplength=200)
        info.pack(padx=8, pady=4)

        # Table
        table_frame = ttk.LabelFrame(ctrl_frame, text=' Accommodation Coefficients ')
        table_frame.pack(fill='x', pady=(0, 8))

        cols = ('gas', 'ratio', 'aE')
        self.accom_tree = ttk.Treeview(table_frame, columns=cols, show='headings', height=9)
        self.accom_tree.heading('gas', text='Gas')
        self.accom_tree.heading('ratio', text='aX/aN₂')
        self.accom_tree.heading('aE', text='aE')
        self.accom_tree.column('gas', width=50)
        self.accom_tree.column('ratio', width=70)
        self.accom_tree.column('aE', width=70)
        self.accom_tree.pack(padx=4, pady=4)

        plot_frame = ttk.Frame(self)
        plot_frame.pack(side='right', fill='both', expand=True, padx=8, pady=8)

        with plt.rc_context(MPL_STYLE):
            self.fig = Figure(figsize=(9, 6), dpi=100)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()
        toolbar.pack(fill='x')

    def on_theme_changed(self):
        self.fig.patch.set_facecolor(COLORS['bg_card'])

    def _update_plot(self, *args):
        surface = self.surface_var.get()
        aN2 = self.sl_aN2_accom.get()
        accom = ACCOM_RATIOS_W if surface == 'W' else ACCOM_RATIOS_Si

        # Update table
        for item in self.accom_tree.get_children():
            self.accom_tree.delete(item)
        for key, ratio in accom.items():
            aE = aN2 * ratio
            flag = ' ⚠' if aE > 1.0 else ''
            self.accom_tree.insert('', 'end', values=(
                GAS_DATA[key]['symbol'], f"{ratio:.2f}", f"{aE:.3f}{flag}"))

        with plt.rc_context(MPL_STYLE):
            self.fig.clear()
            self.fig.patch.set_facecolor(COLORS['bg_card'])
            ax1 = self.fig.add_subplot(121)
            ax2 = self.fig.add_subplot(122)

            gases = list(accom.keys())
            ratios = [accom[k] for k in gases]
            aEs = [min(aN2 * accom[k], 1.0) for k in gases]
            colors = [GAS_DATA[k]['color'] for k in gases]
            labels = [GAS_DATA[k]['symbol'] for k in gases]
            over1 = [aN2 * accom[k] > 1.0 for k in gases]

            # Ratio plot
            bars = ax1.barh(range(len(gases)), ratios, color=colors, alpha=0.7, height=0.6)
            ax1.axvline(x=1.0, color=COLORS['error'], linestyle='--', alpha=0.5, label='Ratio = 1')
            ax1.set_yticks(range(len(gases)))
            ax1.set_yticklabels(labels)
            ax1.set_xlabel('a_X / a_N₂', fontsize=10)
            ax1.set_title('Relative Ratios', fontsize=11, color=COLORS['text_bright'])
            ax1.grid(True, axis='x', alpha=0.3)

            # Absolute values
            bar_colors = [COLORS['error'] if o else c for o, c in zip(over1, colors)]
            ax2.barh(range(len(gases)), aEs, color=bar_colors, alpha=0.7, height=0.6)
            ax2.axvline(x=1.0, color=COLORS['error'], linestyle='--', alpha=0.5, label='Physical limit')
            ax2.set_yticks(range(len(gases)))
            ax2.set_yticklabels(labels)
            ax2.set_xlabel(f'a_E (a_N₂ = {aN2:.2f})', fontsize=10)
            ax2.set_title(f'Absolute Values ({("Tungsten" if surface == "W" else "Silicon")})',
                          fontsize=11, color=COLORS['text_bright'])
            ax2.grid(True, axis='x', alpha=0.3)
            ax2.legend(fontsize=8)

            self.fig.tight_layout()
            self.canvas.draw_idle()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB: GAS EXPLORER
# ══════════════════════════════════════════════════════════════════════════════

class GasExplorerTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self._build_ui()
        self._update_plot()

    def _build_ui(self):
        self._scroll_panel = ScrollableControlPanel(self, width=270)
        self._scroll_panel.pack(side='left', fill='y', padx=(6, 0), pady=6)
        ctrl_frame = self._scroll_panel.inner

        prop_frame = ttk.LabelFrame(ctrl_frame, text=' Sort/Compare By ')
        prop_frame.pack(fill='x', pady=(0, 8))

        self.prop_var = tk.StringVar(value='m')
        props = [('m', 'Molecular Mass (amu)'), ('f', 'Degrees of Freedom'),
                 ('cbar', 'Mean Velocity (m/s)'), ('plbar', 'p·λ̄ Product'),
                 ('gamma', 'γ (Cp/Cv)')]
        for val, text in props:
            ttk.Radiobutton(prop_frame, text=text, variable=self.prop_var,
                            value=val, command=self._update_plot,
                            style='TCheckbutton').pack(anchor='w', padx=8, pady=2)

        # Data table
        table_frame = ttk.LabelFrame(ctrl_frame, text=' Gas Properties (Table I) ')
        table_frame.pack(fill='both', expand=True, pady=(0, 8))

        cols = ('gas', 'm', 'f', 'gamma', 'cbar', 'plbar')
        self.tree = ttk.Treeview(table_frame, columns=cols, show='headings', height=9)
        headers = {'gas': 'Gas', 'm': 'Mass', 'f': 'DOF', 'gamma': 'γ', 'cbar': 'c̄', 'plbar': 'p·λ̄'}
        for c in cols:
            self.tree.heading(c, text=headers[c])
            self.tree.column(c, width=55)
        self.tree.pack(padx=4, pady=4, fill='both', expand=True)

        for key, g in GAS_DATA.items():
            self.tree.insert('', 'end', values=(
                g['symbol'], g['m'], g['f'], g['gamma'],
                g['cbar'], f"{g['plbar']*1000:.1f}"))

        plot_frame = ttk.Frame(self)
        plot_frame.pack(side='right', fill='both', expand=True, padx=8, pady=8)

        with plt.rc_context(MPL_STYLE):
            self.fig = Figure(figsize=(9, 6), dpi=100)
            self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()
        toolbar.pack(fill='x')

    def on_theme_changed(self):
        self.fig.patch.set_facecolor(COLORS['bg_card'])

    def _update_plot(self, *args):
        with plt.rc_context(MPL_STYLE):
            self.ax.clear()
            self.fig.patch.set_facecolor(COLORS['bg_card'])
            prop = self.prop_var.get()

            names = {'m': 'Molecular Mass (amu)', 'f': 'Degrees of Freedom',
                     'cbar': 'Mean Thermal Velocity (m/s)', 'plbar': 'p·λ̄ Product (10⁻³ m·Pa)',
                     'gamma': 'γ = Cp/Cv'}

            gases = sorted(GAS_DATA.keys(), key=lambda k: GAS_DATA[k][prop])
            values = [GAS_DATA[k][prop] * (1000 if prop == 'plbar' else 1) for k in gases]
            colors = [GAS_DATA[k]['color'] for k in gases]
            labels = [GAS_DATA[k]['symbol'] for k in gases]

            bars = self.ax.bar(range(len(gases)), values, color=colors, alpha=0.85,
                               edgecolor=[c for c in colors], linewidth=1.5)
            self.ax.set_xticks(range(len(gases)))
            self.ax.set_xticklabels(labels, fontsize=10)
            self.ax.set_ylabel(names.get(prop, prop), fontsize=11)
            self.ax.set_title(f'Gas Species Comparison — {names.get(prop, prop)}',
                              fontsize=12, color=COLORS['text_bright'])
            self.ax.grid(True, axis='y', alpha=0.3)

            # Add value labels on bars
            for bar, val in zip(bars, values):
                self.ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02,
                             f'{val:.1f}' if prop != 'f' else f'{val:.0f}',
                             ha='center', fontsize=8, color=COLORS['text_dim'])

            self.fig.tight_layout()
            self.canvas.draw_idle()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB: CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════

class CalculatorTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        # Two-column layout
        left = ttk.Frame(self)
        left.pack(side='left', fill='both', expand=True, padx=8, pady=8)

        right = ttk.Frame(self)
        right.pack(side='right', fill='both', expand=True, padx=8, pady=8)

        # ── Pressure Correction ──
        cf_frame = ttk.LabelFrame(left, text=' Pressure Correction Calculator ')
        cf_frame.pack(fill='x', pady=(0, 12))

        row1 = ttk.Frame(cf_frame)
        row1.pack(fill='x', padx=8, pady=4)
        ttk.Label(row1, text='Gas Species:').pack(side='left')
        self.calc_gas = tk.StringVar(value='Ar')
        gas_combo = ttk.Combobox(row1, textvariable=self.calc_gas,
                                 values=[f"{GAS_DATA[k]['symbol']} ({k})" for k in GAS_DATA],
                                 state='readonly', width=20)
        gas_combo.pack(side='right')
        gas_combo.bind('<<ComboboxSelected>>', self._calc_correction)

        row2 = ttk.Frame(cf_frame)
        row2.pack(fill='x', padx=8, pady=4)
        self.pind_label = ttk.Label(row2, text='Indicated Pressure:')
        self.pind_label.pack(side='left')
        self.calc_pind = tk.StringVar(value='1')
        e_pind = ttk.Entry(row2, textvariable=self.calc_pind, width=12)
        e_pind.pack(side='right')
        e_pind.bind('<KeyRelease>', self._calc_correction)

        ttk.Button(cf_frame, text='Calculate True Pressure', command=self._calc_correction).pack(padx=8, pady=4)

        self.result_frame = ttk.Frame(cf_frame, style='Card.TFrame')
        self.result_frame.pack(fill='x', padx=8, pady=4)

        self.result_label = ttk.Label(self.result_frame, text='', style='Big.TLabel')
        self.result_label.pack(pady=4)
        self.result_detail = ttk.Label(self.result_frame, text='', style='Dim.TLabel',
                                       wraplength=350)
        self.result_detail.pack(pady=(0, 4))

        # ── Heat Transfer Calculator ──
        ht_frame = ttk.LabelFrame(left, text=' Heat Transfer Calculator ')
        ht_frame.pack(fill='x', pady=(0, 12))

        t_unit = get_temperature_unit()
        t_lbl = TEMPERATURE_UNITS[t_unit]['label']
        T1_display = convert_temperature(393, 'K', t_unit)
        T2_display = convert_temperature(296, 'K', t_unit)
        params = [
            ('Gas:', 'ht_gas', 'N2'),
            ('Pressure:', 'ht_p', '0.1'),
            ('a_E:', 'ht_aE', '0.6'),
            (f'T\u2081 wire ({t_lbl}):', 'ht_T1', f'{T1_display:.1f}'),
            (f'T\u2082 enclosure ({t_lbl}):', 'ht_T2', f'{T2_display:.1f}'),
        ]

        self.ht_vars = {}
        self.ht_labels = {}
        for label, var_name, default in params:
            row = ttk.Frame(ht_frame)
            row.pack(fill='x', padx=8, pady=2)
            lbl = ttk.Label(row, text=label)
            lbl.pack(side='left')
            self.ht_labels[var_name] = lbl
            var = tk.StringVar(value=default)
            self.ht_vars[var_name] = var
            if var_name == 'ht_gas':
                combo = ttk.Combobox(row, textvariable=var,
                                     values=list(GAS_DATA.keys()), state='readonly', width=10)
                combo.pack(side='right')
            else:
                ttk.Entry(row, textvariable=var, width=10).pack(side='right')

        ttk.Button(ht_frame, text='Calculate Heat Flow', command=self._calc_heat).pack(padx=8, pady=4)

        self.ht_result = ttk.Label(ht_frame, text='', style='Card.TLabel', wraplength=350)
        self.ht_result.pack(padx=8, pady=4)

        # ── Quick Reference Cards ──
        self.ref_frame = ttk.LabelFrame(right, text=' Quick Reference — All Correction Factors ')
        self.ref_frame.pack(fill='both', expand=True, pady=(0, 8))

        self.ref_cards = []
        for key, d in EXPERIMENTAL_CF.items():
            gas = GAS_DATA[key]
            card = ttk.Frame(self.ref_frame, style='Card.TFrame')
            card.pack(fill='x', padx=6, pady=3)

            ttk.Label(card, text=f"  {gas['symbol']}", style='Accent.TLabel',
                      width=6).pack(side='left', padx=(4, 8))
            ttk.Label(card, text=f"CF = {d['mean']:.2f}  (±{d['spread']:.2f})",
                      style='Card.TLabel').pack(side='left')

            direction = '↑ underreads' if d['mean'] > 1 else ('↓ overreads' if d['mean'] < 1 else '— reference')
            range_lbl = ttk.Label(card, text='', style='Dim.TLabel')
            range_lbl.pack(side='right', padx=4)
            self.ref_cards.append((key, direction, range_lbl))

        # Initial unit sync
        self._sync_unit_labels()

    def _sync_unit_labels(self):
        """Update all labels and reference cards to reflect current units."""
        unit = get_pressure_unit()
        ulbl = PRESSURE_UNITS[unit]['label']
        self.pind_label.config(text=f'Indicated Pressure ({ulbl}):')
        self.ht_labels['ht_p'].config(text=f'Pressure ({ulbl}):')

        t_unit = get_temperature_unit()
        t_lbl = TEMPERATURE_UNITS[t_unit]['label']
        self.ht_labels['ht_T1'].config(text=f'T\u2081 wire ({t_lbl}):')
        self.ht_labels['ht_T2'].config(text=f'T\u2082 enclosure ({t_lbl}):')

        for key, direction, range_lbl in self.ref_cards:
            d = EXPERIMENTAL_CF[key]
            # Parse the range string (values are in Pa in the paper)
            try:
                parts = d['range'].split('-')
                lo_pa = float(parts[0])
                hi_pa = float(parts[1])
                lo = convert_pressure(lo_pa, unit)
                hi = convert_pressure(hi_pa, unit)
                range_str = f"{lo:.3g}–{hi:.3g} {ulbl}"
            except (ValueError, IndexError):
                range_str = f"{d['range']} Pa"
            range_lbl.config(text=f"  {direction}  |  Range: {range_str}")

    def _get_gas_key(self):
        val = self.calc_gas.get()
        for key in GAS_DATA:
            if key in val:
                return key
        return 'N2'

    def _calc_correction(self, *args):
        self._sync_unit_labels()
        try:
            key = self._get_gas_key()
            p_ind_user = float(self.calc_pind.get())
            unit = get_pressure_unit()
            ulbl = PRESSURE_UNITS[unit]['label']
            cf = EXPERIMENTAL_CF[key]['mean']
            spread = EXPERIMENTAL_CF[key]['spread']
            p_true = p_ind_user * cf
            p_min = p_ind_user * (cf - spread)
            p_max = p_ind_user * (cf + spread)

            self.result_label.config(text=f"  {p_true:.4g} {ulbl}  ")
            self.result_detail.config(
                text=f"CF = {cf:.2f} ± {spread:.2f}\n"
                     f"Range: {p_min:.4g} – {p_max:.4g} {ulbl}\n"
                     f"Formula: p_true = {p_ind_user} × {cf:.2f} = {p_true:.4g} {ulbl}")
        except (ValueError, KeyError):
            self.result_label.config(text="  Enter valid values  ")
            self.result_detail.config(text="")

    def _calc_heat(self, *args):
        self._sync_unit_labels()
        try:
            gas_key = self.ht_vars['ht_gas'].get()
            unit = get_pressure_unit()
            ulbl = PRESSURE_UNITS[unit]['label']
            factor = PRESSURE_UNITS[unit]['factor']
            p_user = float(self.ht_vars['ht_p'].get())
            p_pa = p_user / factor   # convert user units → Pa
            aE = float(self.ht_vars['ht_aE'].get())
            t_unit = get_temperature_unit()
            T1 = convert_temperature(float(self.ht_vars['ht_T1'].get()), t_unit, 'K')
            T2 = convert_temperature(float(self.ht_vars['ht_T2'].get()), t_unit, 'K')
            gas = GAS_DATA[gas_key]

            cfg = dict(GAUGE_CONFIGS['jousten_wire'])
            cfg['T1'] = T1
            cfg['T2'] = T2

            Q_combined, Q_mol, Q_visc = calc_heat_flow(gas_key, cfg, p_pa, aE)

            self.ht_result.config(
                text=f"  {gas['symbol']} at {p_user:.4g} {ulbl}:\n"
                     f"  Q_molecular = {Q_mol*1000:.4g} mW\n"
                     f"  Q_viscous   = {Q_visc*1000:.4g} mW\n"
                     f"  Q_combined  = {Q_combined*1000:.4g} mW\n"
                     f"  Dominant regime: {'Molecular' if Q_mol < Q_visc else 'Viscous'}")
        except (ValueError, KeyError) as e:
            self.ht_result.config(text=f"  Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB: HOW IT WORKS (EDUCATIONAL)
# ══════════════════════════════════════════════════════════════════════════════

class LearnTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self._canvas = None
        self._text_widgets = []
        self._build_ui()

    def _build_ui(self):
        self._canvas = tk.Canvas(self, bg=COLORS['bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient='vertical', command=self._canvas.yview)
        scroll_frame = ttk.Frame(self._canvas)

        scroll_frame.bind('<Configure>', lambda e: self._canvas.configure(scrollregion=self._canvas.bbox('all')))
        self._content_window_id = self._canvas.create_window((0, 0), window=scroll_frame, anchor='nw')
        self._canvas.configure(yscrollcommand=scrollbar.set)

        def _on_canvas_resize(event):
            self._canvas.itemconfigure(self._content_window_id, width=max(event.width - 2, 200))
        self._canvas.bind('<Configure>', _on_canvas_resize)

        scrollbar.pack(side='right', fill='y')
        self._canvas.pack(side='left', fill='both', expand=True)

        # Bind mousewheel
        def _on_mousewheel(event):
            self._canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        self._canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Content

        sections = [
            ("What is a Pirani Gauge?",
             "A Pirani gauge is a thermal conductivity vacuum gauge that measures gas pressure by "
             "detecting how much heat is carried away from a heated element (typically a thin wire) by "
             "the surrounding gas. As pressure decreases, fewer gas molecules are available to transport "
             "heat, so the heated element retains more energy.\n\n"
             "In the constant-temperature Pirani gauge studied in this paper, a Wheatstone bridge keeps "
             "the wire at a fixed temperature (~120°C for conventional, ~60°C for MEMS). The electrical "
             "power needed to maintain this temperature directly indicates gas pressure.\n\n"
             "Typical measurement range: 0.1 Pa to 10⁴ Pa, sometimes up to atmospheric pressure."),

            ("The Energy Balance (Eq. 1)",
             "Q̇_el = Q̇_gas + Q̇_supp + Q̇_rad + Q̇_conv\n\n"
             "• Q̇_gas — Heat carried by gas molecules (PRESSURE-DEPENDENT signal)\n"
             "• Q̇_supp — Heat conducted through wire supports (fixed offset)\n"
             "• Q̇_rad — Thermal radiation from the hot wire (fixed offset)\n"
             "• Q̇_conv — Convection (significant only at high pressure)\n\n"
             "The 'zero offset' p₀ = Q̇_supp + Q̇_rad is set during gauge calibration."),

            ("Molecular Regime — Low Pressure (Eq. 3)",
             "Q̇_gas,mol = aE · (f+1)/8 · c̄ · A · (T₁−T₂)/Tx · p\n\n"
             "Heat transfer is LINEAR with pressure — this is the useful range.\n"
             "Each molecule independently carries energy between wire and wall.\n"
             "Key parameters: accommodation coefficient (aE), degrees of freedom (f),\n"
             "mean velocity (c̄), wire area (A), temperatures (T₁, T₂)."),

            ("Viscous Regime — High Pressure (Eqs. 6-7)",
             "In the viscous regime, molecule-molecule collisions dominate.\n"
             "Heat transfer becomes INDEPENDENT of pressure → gauge saturates.\n"
             "The mean free path λ becomes smaller than gauge dimensions.\n\n"
             "Q̇_gas,visc ∝ (9γ−5)/4 · λ̄/c̄ · f·k/(2m) · geometry factor\n\n"
             "This sets the upper pressure limit of the Pirani gauge."),

            ("Combined Formula (Eqs. 8-9)",
             "1/Q̇_gas = 1/Q̇_mol + 1/Q̇_visc  (like resistances in series)\n\n"
             "This gives: Q̇_gas = αp / (1 + gp)\n\n"
             "• At low p (gp ≪ 1): Q̇_gas ≈ αp (linear, molecular)\n"
             "• At high p (gp ≫ 1): Q̇_gas ≈ α/g (saturated, viscous)\n\n"
             "This is the classic Pirani calibration curve shape."),

            ("Why Gas Species Matters (Eqs. 11-12)",
             "Different gases have different:\n"
             "• Accommodation coefficients (aE) — energy transfer efficiency at surfaces\n"
             "• Degrees of freedom (f) — monatomic: 3, diatomic: 5, polyatomic: 6+\n"
             "• Mean velocity (c̄) — lighter = faster (H₂: 1764 m/s vs Xe: 219 m/s)\n\n"
             "The molecular regime ratio Q̇_N₂/Q̇_X gives the correction factor.\n"
             "KEY FINDING: Correction factors vary 10-20% between gauges because\n"
             "the accommodation coefficient depends on BOTH gas AND surface condition."),

            ("The Paper's Key Results",
             "• Four Pirani gauges (VM1-VM4) tested with nine gases\n"
             "• Gas correction factors show considerable gauge-to-gauge spread\n"
             "• MEMS gauge (VM3, silicon) had broader usable range\n"
             "• Molecular regime theory (Eq. 11) matches experiment much better\n"
             "  than viscous theory (Eq. 12)\n"
             "• Upper limits of accommodation coefficients determined:\n"
             "  Tungsten: a_N₂ ≤ 0.86, Silicon: a_N₂ ≤ 0.76\n"
             "• Most probable values: a_N₂ = 0.4-0.6 (W), 0.6-0.76 (Si)"),
        ]

        for title, body in sections:
            frame = ttk.Frame(scroll_frame, style='Card.TFrame')
            frame.pack(fill='x', padx=20, pady=6)

            ttk.Label(frame, text=title, style='Title.TLabel').pack(anchor='w', padx=12, pady=(8, 4))

            text_widget = tk.Text(frame, wrap='word', bg=COLORS['bg_card'], fg=COLORS['text'],
                                  font=('Consolas', 10), relief='flat', height=body.count('\n') + 3,
                                  padx=12, pady=8, highlightthickness=0, borderwidth=0)
            text_widget.insert('1.0', body)
            text_widget.config(state='disabled')
            text_widget.pack(fill='x', padx=8, pady=(0, 8))
            self._text_widgets.append(text_widget)

    def on_theme_changed(self):
        if self._canvas is not None:
            self._canvas.configure(bg=COLORS['bg'])
        for text_widget in self._text_widgets:
            text_widget.configure(
                bg=COLORS['bg_card'],
                fg=COLORS['text'],
                insertbackground=COLORS['text'],
                selectbackground=COLORS['accent'],
                selectforeground='#FFFFFF',
            )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB: CROSS-SECTION / GEOMETRY VIEWER
# ══════════════════════════════════════════════════════════════════════════════

class GeometryViewerTab(ttk.Frame):
    """Interactive 2D cross-section view of gauge configurations."""
    def __init__(self, parent):
        super().__init__(parent)
        self._update_job = None
        self._build_ui()
        self._sync_pressure_slider_display()
        self._update_plot()

    def _build_ui(self):
        self._scroll_panel = ScrollableControlPanel(self, width=270)
        self._scroll_panel.pack(side='left', fill='y', padx=(6, 0), pady=6)
        ctrl_frame = self._scroll_panel.inner

        cfg_frame = ttk.LabelFrame(ctrl_frame, text=' Configuration ')
        cfg_frame.pack(fill='x', pady=(0, 8))

        self.geo_config = tk.StringVar(value='psg55x')
        for key, cfg in GAUGE_CONFIGS.items():
            ttk.Radiobutton(cfg_frame, text=cfg['name'],
                            variable=self.geo_config, value=key,
                            command=self._schedule_update,
                            style='TCheckbutton').pack(anchor='w', padx=8, pady=2)

        # Description
        self.desc_label = ttk.Label(ctrl_frame, text='', style='Dim.TLabel', wraplength=250)
        self.desc_label.pack(padx=8, pady=8)

        # Visualization options
        vis_frame = ttk.LabelFrame(ctrl_frame, text=' Visualization ')
        vis_frame.pack(fill='x', pady=(0, 8))

        self.show_molecules = tk.BooleanVar(value=True)
        ttk.Checkbutton(vis_frame, text='Show gas molecules', variable=self.show_molecules,
                        command=self._schedule_update).pack(anchor='w', padx=8, pady=2)

        self.show_heat_arrows = tk.BooleanVar(value=True)
        ttk.Checkbutton(vis_frame, text='Show heat flow arrows', variable=self.show_heat_arrows,
                        command=self._schedule_update).pack(anchor='w', padx=8, pady=2)

        self.show_knudsen = tk.BooleanVar(value=True)
        ttk.Checkbutton(vis_frame, text='Show Knudsen number info', variable=self.show_knudsen,
                        command=self._schedule_update).pack(anchor='w', padx=8, pady=2)

        self.sl_pressure_geo = LabeledSlider(ctrl_frame, 'Pressure', -2, 5, 1,
                                             fmt='{:.1f}',
                                             command=lambda v: self._schedule_update())
        self.sl_pressure_geo.pack(fill='x', padx=6, pady=8)

        plot_frame = ttk.Frame(self)
        plot_frame.pack(side='right', fill='both', expand=True, padx=8, pady=8)

        with plt.rc_context(MPL_STYLE):
            self.fig = Figure(figsize=(9, 7), dpi=100)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

    def _sync_pressure_slider_display(self):
        unit = get_pressure_unit()
        self.sl_pressure_geo.set_value_formatter(
            lambda v: format_pressure(10 ** v, unit, fmt='{:.3g}')
        )

    def on_global_units_changed(self):
        self._sync_pressure_slider_display()
        self._schedule_update()

    def on_theme_changed(self):
        self.fig.patch.set_facecolor(COLORS['bg_card'])

    def _schedule_update(self, *args):
        if self._update_job is not None:
            self.after_cancel(self._update_job)
        self._update_job = self.after(70, self._update_plot)

    def _update_plot(self, *args):
        self._update_job = None
        cfg_key = self.geo_config.get()
        cfg = GAUGE_CONFIGS[cfg_key]
        self.desc_label.config(text=cfg['desc'])

        pressure = 10 ** self.sl_pressure_geo.get()

        with plt.rc_context(MPL_STYLE):
            self.fig.clear()
            self.fig.patch.set_facecolor(COLORS['bg_card'])

            if cfg['geometry'] == 'cylindrical':
                self._draw_cylindrical(pressure, cfg)
            else:
                self._draw_plates(pressure, cfg)

            self.fig.tight_layout()
            self.canvas.draw()

    def _draw_cylindrical(self, pressure, cfg):
        ax = self.fig.add_subplot(121)
        ax3 = self.fig.add_subplot(122, projection='3d')

        # 2D cross-section
        enc_r = 8  # mm visual scale
        wire_r = 0.5

        theta = np.linspace(0, 2*np.pi, 100)
        is_square = cfg.get('geometry') == 'square_cavity'
        if is_square:
            sq = np.array([
                [-enc_r, -enc_r],
                [ enc_r, -enc_r],
                [ enc_r,  enc_r],
                [-enc_r,  enc_r],
                [-enc_r, -enc_r],
            ])
            ax.plot(sq[:, 0], sq[:, 1], color=COLORS['accent'], linewidth=2.5)
            ax.fill(sq[:, 0], sq[:, 1], alpha=0.05, color=COLORS['accent'])
        else:
            ax.plot(enc_r * np.cos(theta), enc_r * np.sin(theta), color=COLORS['accent'], linewidth=2.5)
            ax.fill(enc_r * np.cos(theta), enc_r * np.sin(theta), alpha=0.05, color=COLORS['accent'])

        ax.plot(wire_r * np.cos(theta), wire_r * np.sin(theta), color='#ff6b35', linewidth=2.5)
        ax.fill(wire_r * np.cos(theta), wire_r * np.sin(theta), alpha=0.5, color='#ff6b35')

        ax.text(0, 0, 'T₁', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
        ax.text(enc_r * 0.75, enc_r * 0.75, 'T₂', ha='center', fontsize=9, color=COLORS['accent'])

        if self.show_molecules.get():
            np.random.seed(42)
            n_mol = min(int(pressure / 5) + 5, 80)
            for _ in range(n_mol):
                if is_square:
                    x = np.random.uniform(-enc_r + 0.6, enc_r - 0.6)
                    y = np.random.uniform(-enc_r + 0.6, enc_r - 0.6)
                    if (x * x + y * y) < (wire_r + 0.4) ** 2:
                        continue
                else:
                    r = np.random.uniform(wire_r + 0.5, enc_r - 0.5)
                    angle = np.random.uniform(0, 2*np.pi)
                    x, y = r * np.cos(angle), r * np.sin(angle)
                ax.plot(x, y, 'o', color=COLORS['accent'], markersize=2, alpha=0.5)

        if self.show_heat_arrows.get():
            for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                dx = np.cos(angle)
                dy = np.sin(angle)
                ax.annotate('', xy=(enc_r*0.8*dx, enc_r*0.8*dy),
                            xytext=(wire_r*2*dx, wire_r*2*dy),
                            arrowprops=dict(arrowstyle='->', color='#ff6b3588', lw=1.5))

        ax.set_xlim(-enc_r*1.3, enc_r*1.3)
        ax.set_ylim(-enc_r*1.3, enc_r*1.3)
        ax.set_aspect('equal')
        ax.set_title('Cross-Section View', fontsize=11, color=COLORS['text_bright'])
        orient_note = 'horizontal wire (stronger convection mixing)' if cfg.get('orientation', 'vertical') == 'horizontal' else 'vertical wire'
        ax.text(0, enc_r * 1.15, f'Orientation: {orient_note}',
            ha='center', fontsize=8, color=COLORS['text_dim'])

        if self.show_knudsen.get():
            lbar_n2 = GAS_DATA['N2']['plbar'] / pressure if pressure > 0 else 999
            d_char = 0.008  # 8mm characteristic dimension
            Kn = lbar_n2 / d_char
            regime = 'Molecular' if Kn > 1 else ('Transition' if Kn > 0.01 else 'Viscous')
            unit = get_pressure_unit()
            ax.text(0, -enc_r*1.15,
                f'Kn = {Kn:.2g}  ({regime} regime)\nλ = {lbar_n2*1000:.2g} mm   p = {format_pressure(pressure, unit, fmt="{:.3g}")}',
                    ha='center', fontsize=8, color=COLORS['text_dim'])

        # 3D cylindrical view
        z = np.linspace(0, 5, 30)
        theta_3d = np.linspace(0, 2*np.pi, 40)
        Z, Theta = np.meshgrid(z, theta_3d)

        if is_square:
            yv = np.linspace(-enc_r, enc_r, 2)
            zv = np.linspace(0, 5, 30)
            Yf, Zf = np.meshgrid(yv, zv)
            Xp = np.full_like(Yf, enc_r)
            Xm = np.full_like(Yf, -enc_r)
            xv = np.linspace(-enc_r, enc_r, 2)
            Xf, Zs = np.meshgrid(xv, zv)
            Yp = np.full_like(Xf, enc_r)
            Ym = np.full_like(Xf, -enc_r)
            for Xs, Ys, Zsrf in [(Xp, Yf, Zf), (Xm, Yf, Zf), (Xf, Yp, Zs), (Xf, Ym, Zs)]:
                ax3.plot_surface(Xs, Ys, Zsrf, alpha=0.12, color=COLORS['accent'])
        else:
            X_enc = enc_r * np.cos(Theta)
            Y_enc = enc_r * np.sin(Theta)
            ax3.plot_surface(X_enc, Y_enc, Z, alpha=0.15, color=COLORS['accent'])

        X_wire = wire_r * np.cos(Theta)
        Y_wire = wire_r * np.sin(Theta)
        ax3.plot_surface(X_wire, Y_wire, Z, alpha=0.8, color='#ff6b35')

        ax3.set_title('3D Gauge Geometry', fontsize=11, color=COLORS['text_bright'])
        ax3.set_xlabel('x (mm)', fontsize=8)
        ax3.set_ylabel('y (mm)', fontsize=8)
        ax3.set_zlabel('z (mm)', fontsize=8)
        try:
            ax3.xaxis.pane.fill = False
            ax3.yaxis.pane.fill = False
            ax3.zaxis.pane.fill = False
        except:
            pass

    def _draw_plates(self, pressure, cfg):
        ax = self.fig.add_subplot(121)
        ax3 = self.fig.add_subplot(122, projection='3d')

        # 2D side view of parallel plates
        plate_w = 10
        gap = 3

        # Bottom plate (enclosure, T2)
        ax.fill_between([-plate_w/2, plate_w/2], [-gap/2 - 0.3, -gap/2 - 0.3],
                        [-gap/2, -gap/2], color=COLORS['accent'], alpha=0.6)
        ax.text(0, -gap/2 - 0.7, 'Enclosure (T₂)', ha='center', fontsize=9, color=COLORS['accent'])

        # Top plate (heated, T1)
        ax.fill_between([-plate_w/2, plate_w/2], [gap/2, gap/2],
                        [gap/2 + 0.3, gap/2 + 0.3], color='#ff6b35', alpha=0.6)
        ax.text(0, gap/2 + 0.7, 'Heated Sheet (T₁)', ha='center', fontsize=9, color='#ff6b35')

        if self.show_molecules.get():
            np.random.seed(42)
            n_mol = min(int(pressure / 2) + 5, 60)
            for _ in range(n_mol):
                x = np.random.uniform(-plate_w/2 + 0.5, plate_w/2 - 0.5)
                y = np.random.uniform(-gap/2 + 0.2, gap/2 - 0.2)
                ax.plot(x, y, 'o', color=COLORS['accent'], markersize=2, alpha=0.4)

        if self.show_heat_arrows.get():
            for x in np.linspace(-plate_w/2 + 1, plate_w/2 - 1, 6):
                ax.annotate('', xy=(x, -gap/2 + 0.2), xytext=(x, gap/2 - 0.2),
                            arrowprops=dict(arrowstyle='->', color='#ff6b3588', lw=1.5))

        ax.set_xlim(-plate_w, plate_w)
        ax.set_ylim(-gap*1.5, gap*1.5)
        ax.set_aspect('equal')
        ax.set_title('Side View — Parallel Plates', fontsize=11, color=COLORS['text_bright'])

        if self.show_knudsen.get():
            gap_m = float(cfg.get('gap', 2e-6))
            lbar_n2 = GAS_DATA['N2']['plbar'] / pressure if pressure > 0 else 999
            Kn = lbar_n2 / gap_m
            regime = 'Molecular' if Kn > 1 else ('Transition' if Kn > 0.01 else 'Viscous')
            unit = get_pressure_unit()
            ax.text(0, -gap*1.3,
                f'Kn = {Kn:.2g}  ({regime})\ngap: {gap_m*1e6:.1f} μm   p = {format_pressure(pressure, unit, fmt="{:.3g}")}',
                    ha='center', fontsize=8, color=COLORS['text_dim'])

        # 3D plate visualization
        x_3d = np.linspace(-5, 5, 20)
        y_3d = np.linspace(-5, 5, 20)
        X, Y = np.meshgrid(x_3d, y_3d)

        Z_top = np.ones_like(X) * 1.5
        Z_bot = np.ones_like(X) * -1.5

        ax3.plot_surface(X, Y, Z_top, alpha=0.6, color='#ff6b35')
        ax3.plot_surface(X, Y, Z_bot, alpha=0.4, color=COLORS['accent'])

        ax3.set_title('3D Plate Geometry', fontsize=11, color=COLORS['text_bright'])
        ax3.set_xlabel('x', fontsize=8)
        ax3.set_ylabel('y', fontsize=8)
        ax3.set_zlabel('z', fontsize=8)
        try:
            ax3.xaxis.pane.fill = False
            ax3.yaxis.pane.fill = False
            ax3.zaxis.pane.fill = False
        except:
            pass


# ══════════════════════════════════════════════════════════════════════════════
#  GAS MIXTURE PRESETS
# ══════════════════════════════════════════════════════════════════════════════

GAS_MIXTURE_PRESETS = {
    'pure_n2':      {'name': 'Pure N₂',           'desc': 'Reference calibration gas',
                     'mix': {'N2': 100}},
    'pure_ar':      {'name': 'Pure Ar',            'desc': 'Common sputter / etch gas',
                     'mix': {'Ar': 100}},
    'pure_he':      {'name': 'Pure He',            'desc': 'Leak detection gas',
                     'mix': {'He': 100}},
    'pure_h2':      {'name': 'Pure H₂',           'desc': 'Lightest & fastest molecule',
                     'mix': {'H2': 100}},
    'pure_xe':      {'name': 'Pure Xe',            'desc': 'Heaviest noble gas',
                     'mix': {'Xe': 100}},
    'air':          {'name': 'Dry Air',            'desc': 'N₂ 78 %  O₂ 21 %  Ar 0.93 %  CO₂ 0.04 %  — real atmosphere',
                     'mix': {'N2': 78, 'O2': 21, 'Ar': 1}},
    'he_leak':      {'name': 'He Leak Check',      'desc': '10 % He tracer in N₂ background',
                     'mix': {'He': 10, 'N2': 90}},
    'residual':     {'name': 'Residual Gas',       'desc': 'Typical UHV residual atmosphere',
                     'mix': {'N2': 40, 'H2': 30, 'CO': 15, 'CO2': 10, 'Ar': 5}},
    'sputter':      {'name': 'Sputter Ar / N₂',   'desc': 'Reactive sputter process gas',
                     'mix': {'Ar': 80, 'N2': 20}},
    'noble_mix':    {'name': 'Noble Gas Mix',      'desc': 'Five noble gases combined',
                     'mix': {'He': 40, 'Ne': 20, 'Ar': 20, 'Kr': 10, 'Xe': 10}},
    'light':        {'name': 'Light Gases',        'desc': 'H₂ + He — fast, high thermal transport',
                     'mix': {'H2': 60, 'He': 40}},
    'custom':       {'name': 'Custom Mixture',     'desc': 'Edit ratios below for any combination',
                     'mix': {}},
}


# ══════════════════════════════════════════════════════════════════════════════
#  TAB: 3D MOLECULAR COLLISION SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

class MolecularSimTab(ttk.Frame):
    """
    Interactive 3D simulation of gas molecules colliding with the Pirani
    filament. Molecules are rendered as 3D balls with size proportional to
    molecular mass. On collision, kinetic energy is transferred from the hot
    filament to the gas molecule, cooling that segment of the filament.

    Filament color indicates local temperature:
        RED  = hot (equilibrium Joule-heated temperature)
        BLUE = cooled (energy removed by molecular collision)

    Between collisions each segment recovers toward equilibrium via:
        - Joule heating (electrical power input)
        - Thermal conduction to neighbouring segments

    Energy transfer per collision scales with:
        - Accommodation coefficient αE (surface & gas dependent)
        - Degrees of freedom f of the gas molecule
        - Temperature difference (T_wire − T_gas)
    """

    # ── Visual geometry (display units) ──────────────────────────────────────
    R_WIRE = 0.4           # filament display radius
    L_WIRE = 10.0          # filament display length (along z)
    R_ENC = 4.5            # enclosure display radius
    N_SEG = 24             # filament segments for temperature tracking
    N_THETA = 12           # angular resolution of wire cylinder surface

    # ── Physics (tuned for visual clarity) ───────────────────────────────────
    T_HOT = 393.0          # equilibrium wire temperature (K)
    T_COLD = 296.0         # enclosure / gas temperature (K)
    HEATING_TAU = 0.55     # Joule-heating recovery time constant (visual s)
    DIFFUSION_K = 4.0      # thermal diffusion along the wire
    COOL_BASE = 16.0       # base ΔT per collision (K), scaled by αE & f

    # ── Animation ────────────────────────────────────────────────────────────
    DT = 0.04              # simulation timestep (s)
    INTERVAL = 33          # ms between frames (~30 fps)
    DRAW_EVERY = 3         # draw every N physics steps

    # ── Molecule visuals ─────────────────────────────────────────────────────
    MOL_SPEED_BASE = 4.0   # display speed for N₂ (units / s)
    MOL_SIZE_BASE = 60     # scatter marker size for N₂
    MOLECULES_PER_PARTICLE = 3.0e16  # real molecules represented by one visual particle
    TRAIL_LENGTH = 12      # stored positions per particle for trajectory trails
    MOL_COLLISION_RADIUS = 0.22   # display-units proximity for molecule collisions
    MOL_MM_RELAX = 0.28           # temperature exchange fraction per molecule collision
    PIRANI_AVG_WINDOW = 50        # default rolling average window (measurements)
    GRAVITY_ACCEL = -0.30         # display-units/s², downward along global z
    BUOYANCY_ACCEL = 1.10         # max upward accel scale for hot molecules
    CONVECTION_P_ON_MBAR = 70.0   # pressure where convection forcing activates
    CONVECTION_TRANSITION_N = 1.3 # sharpness of pressure activation

    # ── Molecule radii (pm) for realistic relative sizing ────────────────────
    MOL_RADII = {
        'H2': 289, 'He': 260, 'Ne': 275, 'CO': 376, 'N2': 364,
        'O2': 346, 'Ar': 340, 'CO2': 330, 'Kr': 360, 'Xe': 396,
    }

    def __init__(self, parent):
        super().__init__(parent)
        # Make T_HOT / T_COLD mutable instance attributes (override class defaults)
        self.T_HOT = float(MolecularSimTab.T_HOT)
        self.T_COLD = float(MolecularSimTab.T_COLD)
        self.running = False
        self._was_running_before_hide = False   # for tab-switch auto-pause
        self.anim_id = None
        self.collision_count = 0
        self.total_energy_transferred = 0.0
        self.frame_count = 0
        self._last_draw_frame = 0
        self.collision_per_gas = {}  # gas_key -> count
        self._step_cooling_accum = 0.0
        self.collision_signal_ema = 0.0
        self.sensor_samples = 0
        self.sensor_gain_q = None
        self._gain_fast_adapt = 0
        self._substep_remainder = 0.0
        self._last_anim_tick = None
        self._session_bias = float(np.random.default_rng().normal(0, 0.4))
        self.gas_ambient_temp_k = self.T_COLD
        self.wall_coupling_ema = 0.0

        # Electro-thermal Pirani model parameters (constant-bias bridge style).
        self.sensor_t_ref_k = 293.15
        self.sensor_r0_ohm = 10000.0
        self.sensor_tcr_per_k = 0.00385
        self.bridge_v_bias = 2.4
        self.bridge_r1_ohm = 10000.0
        self.bridge_r2_ohm = 10000.0
        self.bridge_v_sensor = 1.2
        self.sensor_emissivity = 0.20
        self.sensor_support_lambda_wmk = 70.0
        self.sensor_support_w_m = 25e-6
        self.sensor_support_t_m = 2.0e-6
        self.sensor_support_l_m = 450e-6
        self.sensor_extra_support_g_wpk = 2.0e-6
        self.sensor_enable_bridge_noise = False

        self.seg_temps = np.full(self.N_SEG, self.T_HOT, dtype=np.float64)
        self.mol_pos = np.zeros((0, 3))
        self.mol_vel = np.zeros((0, 3))
        self.mol_gas_keys = []  # per-molecule gas species key
        # Precomputed per-molecule physics arrays (set in _init_molecules)
        self.mol_aE = np.zeros(0)
        self.mol_f_plus1 = np.zeros(0)
        self.mol_thermal_speed = np.zeros(0)  # Rayleigh scale at T_COLD
        self.molecules_per_particle = self.MOLECULES_PER_PARTICLE
        self.mol_trail_pos = np.zeros((0, self.TRAIL_LENGTH, 3))
        self.mol_trail_temp = np.zeros((0, self.TRAIL_LENGTH))
        self.pirani_readings_pa = deque(maxlen=self.PIRANI_AVG_WINDOW)

        self._elev = 20
        self._azim = -60
        self._camera_preset_pending = False  # flag to skip reading axes on next draw
        self._pending_reinit_job = None
        self._pending_reinit_draw = False

        self._build_ui()
        self._on_config_change()
        self._apply_preset('pure_n2')
        self._init_molecules()
        self._draw_scene()

    # ── UI Construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        # ---------- left: scrollable controls ----------
        self._scroll_panel = ScrollableControlPanel(self, width=278)
        self._scroll_panel.pack(side='left', fill='y', padx=(6, 0), pady=6)
        ctrl = self._scroll_panel.inner

        # ── Gas Mixture selector ──
        mix_frame = ttk.LabelFrame(ctrl, text='  Gas Mixture  ')
        mix_frame.pack(fill='x', pady=(0, 6))

        # Preset combobox
        preset_row = ttk.Frame(mix_frame, style='Card.TFrame')
        preset_row.pack(fill='x', padx=8, pady=(6, 2))
        ttk.Label(preset_row, text='Preset', style='Card.TLabel').pack(side='left')
        self.preset_var = tk.StringVar(value='pure_n2')
        preset_names = [v['name'] for v in GAS_MIXTURE_PRESETS.values()]
        self.preset_combo = ttk.Combobox(preset_row, textvariable=self.preset_var,
                                         values=list(GAS_MIXTURE_PRESETS.keys()),
                                         state='readonly', width=16)
        self.preset_combo.pack(side='right')
        self.preset_combo.bind('<<ComboboxSelected>>', self._on_preset_change)

        self.preset_desc = ttk.Label(mix_frame, text='', style='Dim.TLabel',
                                     wraplength=240)
        self.preset_desc.pack(padx=8, pady=(0, 4))

        # Per-gas ratio entries
        self.gas_pct_vars = {}
        self.gas_pct_entries = {}
        self._gas_dot_canvases = []  # tk.Canvas dots for per-gas color indicators
        gas_grid = ttk.Frame(mix_frame, style='Card.TFrame')
        gas_grid.pack(fill='x', padx=6, pady=(0, 4))

        for i, (key, gas) in enumerate(GAS_DATA.items()):
            row = ttk.Frame(gas_grid, style='Card.TFrame')
            row.pack(fill='x', pady=1)

            # Color dot via a small canvas
            dot = tk.Canvas(row, width=10, height=10, bg=COLORS['bg_card'],
                            highlightthickness=0)
            dot.create_oval(1, 1, 9, 9, fill=gas['color'], outline='')
            dot.pack(side='left', padx=(4, 4))
            self._gas_dot_canvases.append(dot)

            ttk.Label(row, text=f"{gas['symbol']}", style='Card.TLabel',
                      width=4).pack(side='left')

            var = tk.StringVar(value='0')
            entry = ttk.Entry(row, textvariable=var, width=5, justify='right')
            entry.pack(side='right', padx=(0, 4))
            entry.bind('<KeyRelease>', self._on_ratio_change)
            ttk.Label(row, text='%', style='Dim.TLabel').pack(side='right')

            self.gas_pct_vars[key] = var
            self.gas_pct_entries[key] = entry

        # Total & normalize
        total_row = ttk.Frame(mix_frame, style='Card.TFrame')
        total_row.pack(fill='x', padx=8, pady=(2, 6))
        self.total_label = ttk.Label(total_row, text='Total: 0 %',
                                     style='Accent.TLabel')
        self.total_label.pack(side='left')
        ttk.Button(total_row, text='Normalize',
                   command=self._normalize_ratios).pack(side='right')

        # Composition bar (tiny matplotlib)
        self.comp_fig = Figure(figsize=(2.6, 0.3), dpi=100)
        self.comp_fig.patch.set_facecolor(COLORS['bg_card'])
        self.comp_canvas = FigureCanvasTkAgg(self.comp_fig, master=mix_frame)
        self.comp_canvas.get_tk_widget().pack(fill='x', padx=6, pady=(0, 6))

        # Gauge configuration
        cfg_frame = ttk.LabelFrame(ctrl, text=' Gauge ')
        cfg_frame.pack(fill='x', pady=(0, 6))
        self.cfg_var = tk.StringVar(value='jousten_wire')
        for key, cfg_info in GAUGE_CONFIGS.items():
            ttk.Radiobutton(cfg_frame, text=cfg_info['name'],
                            variable=self.cfg_var, value=key,
                            command=self._on_config_change,
                            style='TCheckbutton').pack(anchor='w', padx=8, pady=2)

        # Display toggles
        vis_frame = ttk.LabelFrame(ctrl, text=' Display ')
        vis_frame.pack(fill='x', pady=(0, 6))
        self.show_enclosure = tk.BooleanVar(value=True)
        ttk.Checkbutton(vis_frame, text='Show enclosure',
                        variable=self.show_enclosure,
                        command=lambda: self._draw_scene()).pack(anchor='w', padx=8, pady=1)
        self.show_vectors = tk.BooleanVar(value=False)
        ttk.Checkbutton(vis_frame, text='Show velocity vectors',
                        variable=self.show_vectors,
                        command=lambda: self._draw_scene()).pack(anchor='w', padx=8, pady=1)
        self.show_temp_trails = tk.BooleanVar(value=False)
        ttk.Checkbutton(vis_frame, text='Show particle temp trails',
                variable=self.show_temp_trails,
                command=lambda: self._draw_scene()).pack(anchor='w', padx=8, pady=1)

        avg_frame = ttk.LabelFrame(ctrl, text=' Pressure Averaging ')
        avg_frame.pack(fill='x', pady=(0, 6))
        self.sl_avg_samples = LabeledSlider(
            avg_frame,
            'Avg Measurements',
            5,
            200,
            self.PIRANI_AVG_WINDOW,
            fmt='{:.0f}',
            command=lambda v: self._on_avg_window_change(),
        )
        self.sl_avg_samples.pack(fill='x', padx=6, pady=(4, 4))
        self.sl_avg_samples.set_value_formatter(lambda v: f'{int(round(v))} samples')

        # Filament color range
        color_frame = ttk.LabelFrame(ctrl, text=' Filament Color Range ')
        color_frame.pack(fill='x', pady=(0, 6))

        ttk.Label(color_frame, text='Colormap:', style='Card.TLabel').pack(anchor='w', padx=8, pady=(4, 0))
        self.filament_cmap_var = tk.StringVar(value='coolwarm')
        filament_cmaps = ['coolwarm', 'inferno', 'plasma', 'viridis', 'magma',
                          'hot', 'YlOrRd', 'RdYlBu_r', 'Spectral_r', 'turbo']
        cmap_combo = ttk.Combobox(color_frame, textvariable=self.filament_cmap_var,
                                  values=filament_cmaps, state='readonly', width=14)
        cmap_combo.pack(padx=8, pady=2, anchor='w')
        cmap_combo.bind('<<ComboboxSelected>>', lambda e: self._on_color_range_change())

        self.sl_color_min = LabeledSlider(color_frame, 'Color Min Temp (K)', 200, 500, 388,
                                          fmt='{:.0f}', unit='K',
                                          command=lambda v: self._on_color_range_change())
        self.sl_color_min.pack(fill='x', padx=6, pady=2)

        self.sl_color_max = LabeledSlider(color_frame, 'Color Max Temp (K)', 300, 700, 394,
                                          fmt='{:.0f}', unit='K',
                                          command=lambda v: self._on_color_range_change())
        self.sl_color_max.pack(fill='x', padx=6, pady=2)

        self.auto_color_scale_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            color_frame,
            text='Auto-scale to gauge temperatures',
            variable=self.auto_color_scale_var,
            command=lambda: self._auto_update_color_range(force=True),
            style='TCheckbutton',
        ).pack(anchor='w', padx=8, pady=(0, 2))

        ttk.Button(color_frame, text='Reset to Defaults',
                   command=self._reset_color_range).pack(padx=8, pady=(2, 6))

        # Camera angle presets
        cam_frame = ttk.LabelFrame(ctrl, text=' Camera Angle ')
        cam_frame.pack(fill='x', pady=(0, 6))
        CAMERA_PRESETS = [
            ('Default',   20, -60),
            ('Front',      0,   0),
            ('Side',       0, -90),
            ('Top',       90, -90),
            ('Bottom',   -90, -90),
            ('Iso 45°',   30, -45),
            ('Rear',       0, 180),
        ]
        cam_grid = ttk.Frame(cam_frame, style='Card.TFrame')
        cam_grid.pack(fill='x', padx=6, pady=4)
        for idx, (name, elev, azim) in enumerate(CAMERA_PRESETS):
            btn = ttk.Button(
                cam_grid, text=name, width=8,
                command=lambda e=elev, a=azim: self._set_camera(e, a))
            btn.grid(row=idx // 4, column=idx % 4, padx=2, pady=2, sticky='ew')
        for c in range(4):
            cam_grid.columnconfigure(c, weight=1)

        # ---------- right: info panel (read-only displays) ----------
        self._info_panel = ScrollableControlPanel(self, width=260)
        self._info_panel.pack(side='right', fill='y', padx=(0, 6), pady=6)
        info = self._info_panel.inner

        # Statistics
        stats_frame = ttk.LabelFrame(info, text=' Statistics ')
        stats_frame.pack(fill='x', pady=(0, 6))
        self.stats_label = ttk.Label(stats_frame, text='', style='Dim.TLabel',
                                     wraplength=230, justify='left')
        self.stats_label.pack(padx=8, pady=4)

        # Temperature legend
        self.legend_frame = ttk.LabelFrame(info, text=' Filament Temperature ')
        self.legend_frame.pack(fill='x', pady=(0, 6))
        self._build_temp_legend(self.legend_frame)

        # Live controls (moved to right info section)
        quick = ttk.LabelFrame(info, text=' Live Controls ')
        quick.pack(fill='x', pady=(0, 6))

        self.sim_time_label = ttk.Label(
            quick, text='Simulation Time: 0.0 s', style='Dim.TLabel', anchor='center'
        )
        self.sim_time_label.pack(fill='x', padx=6, pady=(4, 2))

        quick_actions = ttk.Frame(quick, style='Card.TFrame')
        quick_actions.pack(fill='x', padx=6, pady=(4, 4))

        self.btn_play = ttk.Button(quick_actions, text='▶  Start Simulation',
                                   command=self._toggle_play)
        self.btn_play.pack(side='left', padx=(0, 6))
        ttk.Button(quick_actions, text='↺  Reset',
                   command=self._reset).pack(side='left')

        quick_units = ttk.Frame(quick, style='Card.TFrame')
        quick_units.pack(fill='x', padx=6, pady=(0, 4))
        ttk.Label(quick_units, text='Temperature Unit:', style='Dim.TLabel').pack(anchor='w')
        self.temp_unit_combo = ttk.Combobox(
            quick_units,
            textvariable=APP_STATE.get('temperature_unit'),
            values=list(TEMPERATURE_UNITS.keys()),
            state='readonly',
            width=6,
        )
        self.temp_unit_combo.pack(anchor='w')

        self.sl_speed = LabeledSlider(
            quick,
            'Sub-steps / Frame',
            0.5,
            6,
            1,
            fmt='{:.1f}',
            command=lambda v: None,
        )
        self.sl_speed.pack(fill='x', padx=6, pady=(0, 2))

        self.sl_env_temp = LabeledSlider(
            quick,
            'Environment Temp',
            200,
            350,
            self.T_COLD,
            fmt='{:.0f}',
            unit='K',
            command=lambda v: self._on_env_temp_change(),
        )
        self.sl_env_temp.pack(fill='x', padx=6, pady=(0, 2))

        self.sl_wire_temp = LabeledSlider(
            quick,
            'Filament Temp',
            313,
            673,
            self.T_HOT,
            fmt='{:.0f}',
            unit='K',
            command=lambda v: self._on_wire_temp_change(),
        )
        self.sl_wire_temp.pack(fill='x', padx=6, pady=(0, 2))

        # ---------- center: 3D canvas ----------
        plot_frame = ttk.Frame(self)
        plot_frame.pack(fill='both', expand=True, padx=8, pady=8)

        pressure_frame = ttk.LabelFrame(plot_frame, text=' Live Pressure ')
        pressure_frame.pack(fill='x', pady=(0, 6))

        pressure_row = ttk.Frame(pressure_frame, style='Card.TFrame')
        pressure_row.pack(fill='x', padx=6, pady=(2, 2))
        ttk.Label(pressure_row, text='Pressure Unit:', style='Dim.TLabel').pack(side='left', padx=(0, 4))
        PressureUnitSelector(pressure_row,
                             on_change=self._on_pressure_unit_change).pack(side='left')

        self.sl_pressure = LabeledSlider(
            pressure_frame,
            'Pressure (log scale)',
            np.log10(1e-8),
            np.log10(2e5),
            np.log10(100),
            fmt='{:.1f}',
            command=lambda v: self._on_pressure_change(),
        )
        self.sl_pressure.pack(fill='x', padx=6, pady=(0, 2))

        self.live_pressure_label = ttk.Label(
            pressure_frame, text='', style='Big.TLabel', anchor='center')
        self.live_pressure_label.pack(fill='x', padx=6, pady=(0, 2))
        self.live_pressure_detail = ttk.Label(
            pressure_frame, text='', style='Dim.TLabel', anchor='center',
            wraplength=520)
        self.live_pressure_detail.pack(fill='x', padx=6, pady=(0, 6))

        with plt.rc_context(MPL_STYLE):
            self.fig = Figure(figsize=(9, 7), dpi=100)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.pack(fill='x')
        NavigationToolbar2Tk(self.canvas, toolbar_frame)

        self._sync_pressure_slider_display()
        self._sync_temperature_unit_display()

    def _build_temp_legend(self, parent):
        """Small horizontal colour-bar using the user-selected colormap and range."""
        t_unit = get_temperature_unit()
        # Use user-configured color range if sliders exist, else defaults
        if hasattr(self, 'sl_color_min'):
            color_min_k = self.sl_color_min.get()
            color_max_k = self.sl_color_max.get()
            if color_min_k >= color_max_k:
                color_max_k = color_min_k + 1.0
        else:
            color_min_k = self.T_COLD - 5
            color_max_k = self.T_HOT + 5
        cmap_name = self.filament_cmap_var.get() if hasattr(self, 'filament_cmap_var') else 'coolwarm'
        t_cold = convert_temperature(color_min_k, 'K', t_unit)
        t_hot = convert_temperature(color_max_k, 'K', t_unit)
        t_mid = (t_cold + t_hot) / 2.0
        with plt.rc_context(MPL_STYLE):
            fig_leg = Figure(figsize=(2.6, 0.5), dpi=100)
            fig_leg.patch.set_facecolor(COLORS['bg_card'])
            ax = fig_leg.add_axes([0.08, 0.55, 0.84, 0.3])
            gradient = np.linspace(0, 1, 256).reshape(1, -1)
            ax.imshow(gradient, aspect='auto', cmap=cmap_name,
                      extent=[t_cold, t_hot, 0, 1])
            ax.set_yticks([])
            ax.set_xticks([t_cold, t_mid, t_hot])
            ax.set_xticklabels([f'{t_cold:.0f} {TEMPERATURE_UNITS[t_unit]["label"]}\ncooled',
                                f'{t_mid:.0f} {TEMPERATURE_UNITS[t_unit]["label"]}',
                                f'{t_hot:.0f} {TEMPERATURE_UNITS[t_unit]["label"]}\nhot'],
                               fontsize=7)
            ax.tick_params(axis='x', length=2, pad=1)
        c = FigureCanvasTkAgg(fig_leg, master=parent)
        c.get_tk_widget().pack(fill='x', padx=4, pady=2)
        c.draw()

    # ── Molecule helpers ─────────────────────────────────────────────────────

    def _get_gas_key(self):
        """Return the dominant gas in the current mixture."""
        fracs = self._get_mixture_fractions()
        if fracs:
            return max(fracs, key=fracs.get)
        return 'N2'

    def _get_gas(self):
        return GAS_DATA[self._get_gas_key()]

    def _mol_count(self):
        """Number of visual molecules, proportional to pressure."""
        p_pa = 10 ** self.sl_pressure.get()
        p_mbar = convert_pressure(p_pa, 'mbar')
        lo_log = -6.0
        hi_log = np.log10(2000.0)
        frac = (np.log10(max(p_mbar, 1e-12)) - lo_log) / (hi_log - lo_log)
        frac = float(np.clip(frac, 0.0, 1.0))
        n_min, n_max = 6, 520
        shaped = frac ** 1.15
        return int(round(n_min + shaped * (n_max - n_min)))

    def _get_accommodation_for(self, gas_key):
        """Effective accommodation coefficient αE for a specific gas."""
        surface = GAUGE_CONFIGS[self.cfg_var.get()]['surface']
        table = ACCOM_RATIOS_W if surface == 'W' else ACCOM_RATIOS_Si
        return min(0.6 * table.get(gas_key, 1.0), 1.0)

    # ── Mixture management ───────────────────────────────────────────────────

    def _get_mixture_fractions(self):
        """Return normalised dict {gas_key: fraction 0-1} from entry fields."""
        raw = {}
        for key, var in self.gas_pct_vars.items():
            try:
                v = float(var.get())
            except (ValueError, tk.TclError):
                v = 0.0
            if v > 0:
                raw[key] = v
        total = sum(raw.values())
        if total <= 0:
            return {'N2': 1.0}
        return {k: v / total for k, v in raw.items()}

    def _apply_preset(self, preset_key):
        """Fill entry fields from a preset."""
        preset = GAS_MIXTURE_PRESETS.get(preset_key)
        if not preset:
            return
        self.preset_var.set(preset_key)
        self.preset_desc.config(text=preset['desc'])
        mix = preset['mix']
        for key in GAS_DATA:
            self.gas_pct_vars[key].set(str(mix.get(key, 0)))
        self._update_composition_display()

    def _on_preset_change(self, event=None):
        key = self.preset_var.get()
        self._apply_preset(key)
        self._schedule_molecule_reinit(draw_if_idle=True, delay_ms=30)

    def _on_ratio_change(self, event=None):
        self._update_composition_display()
        # Switch preset label to Custom if user edits
        self.preset_var.set('custom')
        self.preset_desc.config(text=GAS_MIXTURE_PRESETS['custom']['desc'])
        self._schedule_molecule_reinit(draw_if_idle=True, delay_ms=150)

    def _normalize_ratios(self):
        """Scale entries so they sum to 100 %."""
        fracs = self._get_mixture_fractions()
        for key in GAS_DATA:
            pct = fracs.get(key, 0.0) * 100
            self.gas_pct_vars[key].set(f'{pct:.1f}' if pct > 0 else '0')
        self._update_composition_display()

    def _update_composition_display(self):
        """Refresh the total label and the stacked composition bar."""
        total = 0.0
        for var in self.gas_pct_vars.values():
            try:
                total += float(var.get())
            except (ValueError, tk.TclError):
                pass
        c = COLORS['accent2'] if abs(total - 100) < 0.5 else COLORS['warn']
        self.total_label.config(text=f'Total: {total:.1f} %')

        # Stacked bar
        self.comp_fig.clear()
        self.comp_fig.patch.set_facecolor(COLORS['bg_card'])
        ax = self.comp_fig.add_axes([0.02, 0.2, 0.96, 0.6])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        fracs = self._get_mixture_fractions()
        left = 0.0
        for key, frac in sorted(fracs.items(), key=lambda x: -x[1]):
            gas = GAS_DATA[key]
            ax.barh(0.5, frac, left=left, height=0.8, color=gas['color'],
                    edgecolor=COLORS['bg_card'], linewidth=0.5)
            if frac > 0.08:
                ax.text(left + frac / 2, 0.5, gas['symbol'],
                        ha='center', va='center', fontsize=7,
                        color='white', fontweight='bold')
            left += frac
        self.comp_canvas.draw_idle()

    # ── Molecule initialisation ──────────────────────────────────────────────

    def _init_molecules(self):
        """Populate molecules with random positions & Maxwell–Boltzmann speeds.
        Each molecule is assigned a gas species based on the current mixture."""
        fracs = self._get_mixture_fractions()
        n_total = self._mol_count()
        geometry = GAUGE_CONFIGS[self.cfg_var.get()]['geometry']
        is_plate = geometry == 'plates'
        is_square = geometry == 'square_cavity'

        all_pos, all_vel, all_keys = [], [], []

        for gas_key, frac in fracs.items():
            n = max(1, int(round(n_total * frac)))
            if n == 0:
                continue
            gas = GAS_DATA[gas_key]
            speed = self.MOL_SPEED_BASE * (gas['cbar'] / GAS_DATA['N2']['cbar'])

            pos = np.zeros((n, 3))
            if is_plate:
                pos[:, 0] = np.random.uniform(-self.R_ENC + 0.3, self.R_ENC - 0.3, n)
                pos[:, 1] = np.random.uniform(-self.R_ENC + 0.3, self.R_ENC - 0.3, n)
                pos[:, 2] = np.random.uniform(0.5, self.L_WIRE - 0.5, n)
            elif is_square:
                pos[:, 0] = np.random.uniform(-self.R_ENC + 0.3, self.R_ENC - 0.3, n)
                pos[:, 1] = np.random.uniform(-self.R_ENC + 0.3, self.R_ENC - 0.3, n)
                pos[:, 2] = np.random.uniform(0.5, self.L_WIRE - 0.5, n)
                # Keep particles out of the wire body at initialization.
                r_xy = np.sqrt(pos[:, 0] * pos[:, 0] + pos[:, 1] * pos[:, 1])
                near_wire = r_xy < (self.R_WIRE + 0.2)
                if np.any(near_wire):
                    theta = np.random.uniform(0, 2 * np.pi, int(np.count_nonzero(near_wire)))
                    rr = np.random.uniform(self.R_WIRE + 0.25, self.R_WIRE + 0.6, int(np.count_nonzero(near_wire)))
                    pos[near_wire, 0] = rr * np.cos(theta)
                    pos[near_wire, 1] = rr * np.sin(theta)
            else:
                r = np.random.uniform(self.R_WIRE + 0.3, self.R_ENC - 0.3, n)
                theta = np.random.uniform(0, 2 * np.pi, n)
                pos[:, 0] = r * np.cos(theta)
                pos[:, 1] = r * np.sin(theta)
                pos[:, 2] = np.random.uniform(0.5, self.L_WIRE - 0.5, n)

            vel = np.random.randn(n, 3)
            norms = np.linalg.norm(vel, axis=1, keepdims=True)
            norms[norms < 1e-8] = 1.0
            # Rayleigh scale matches mol_thermal_speed: σ ∝ √T_COLD
            sigma = speed * 0.65 * math.sqrt(self.T_COLD / 296.0)
            vel = vel / norms * np.random.rayleigh(sigma, (n, 1))

            all_pos.append(pos)
            all_vel.append(vel)
            all_keys.extend([gas_key] * n)

        if all_pos:
            self.mol_pos = np.vstack(all_pos)
            self.mol_vel = np.vstack(all_vel)
        else:
            self.mol_pos = np.zeros((0, 3))
            self.mol_vel = np.zeros((0, 3))
        self.mol_gas_keys = all_keys

        # Precompute per-molecule accommodation & (f+1) for vectorised _step
        n = len(self.mol_gas_keys)
        self.mol_aE = np.empty(n, dtype=np.float64)
        self.mol_f_plus1 = np.empty(n, dtype=np.float64)
        for j, gk in enumerate(self.mol_gas_keys):
            self.mol_aE[j] = self._get_accommodation_for(gk)
            self.mol_f_plus1[j] = GAS_DATA[gk]['f'] + 1

        # Thermal speed scale at T_COLD for wall re-thermalization
        self.mol_thermal_speed = np.empty(n, dtype=np.float64)
        for j, gk in enumerate(self.mol_gas_keys):
            gas = GAS_DATA[gk]
            speed = self.MOL_SPEED_BASE * (gas['cbar'] / GAS_DATA['N2']['cbar'])
            self.mol_thermal_speed[j] = speed * 0.65 * math.sqrt(self.T_COLD / 296.0)

        if n > 0:
            p_target = 10 ** self.sl_pressure.get()
            V = self._get_defined_volume_m3()
            self.molecules_per_particle = max(
                (p_target * V) / (kB * self.T_COLD * n),
                1.0,
            )
        else:
            self.molecules_per_particle = self.MOLECULES_PER_PARTICLE

        if n > 0:
            temp_now = self._estimate_particle_temperatures_k()
            self.mol_trail_pos = np.repeat(self.mol_pos[:, np.newaxis, :], self.TRAIL_LENGTH, axis=1)
            self.mol_trail_temp = np.repeat(temp_now[:, np.newaxis], self.TRAIL_LENGTH, axis=1)
        else:
            self.mol_trail_pos = np.zeros((0, self.TRAIL_LENGTH, 3))
            self.mol_trail_temp = np.zeros((0, self.TRAIL_LENGTH))
        # Gas starts in thermal equilibrium with the enclosure walls
        self.gas_ambient_temp_k = self.T_COLD
        self.wall_coupling_ema = 0.0

        self._step_cooling_accum = 0.0
        self.collision_signal_ema = 0.0
        self.sensor_samples = 0
        self.sensor_gain_q = None          # force recalibration for new mixture
        self._gain_fast_adapt = 12         # fast-adapt counter after gas change
        self.pirani_readings_pa.clear()
        if hasattr(self, '_session_bias'):
            del self._session_bias          # will be re-seeded on next reading

    def _cancel_pending_reinit(self):
        """Cancel any queued molecule reinitialization callback."""
        if self._pending_reinit_job is not None:
            try:
                self.after_cancel(self._pending_reinit_job)
            except Exception:
                pass
            self._pending_reinit_job = None

    def _schedule_molecule_reinit(self, draw_if_idle=True, delay_ms=120):
        """Coalesce rapid UI edits into one expensive molecule rebuild."""
        self._pending_reinit_draw = self._pending_reinit_draw or draw_if_idle
        self._cancel_pending_reinit()
        self._pending_reinit_job = self.after(delay_ms, self._run_scheduled_molecule_reinit)

    def _run_scheduled_molecule_reinit(self):
        """Execute deferred molecule reinitialization after controls settle."""
        self._pending_reinit_job = None
        draw_if_idle = self._pending_reinit_draw
        self._pending_reinit_draw = False
        self._init_molecules()
        if draw_if_idle and not self.running:
            self._draw_scene()

    def _flush_pending_molecule_reinit(self):
        """Run queued molecule rebuild immediately when starting the simulation."""
        if self._pending_reinit_job is None:
            return
        self._cancel_pending_reinit()
        self._run_scheduled_molecule_reinit()

    # ── Physics step (fully vectorised) ───────────────────────────────────

    def _step(self):
        """Advance one simulation timestep using vectorised NumPy operations
        for position integration, boundary reflection, and collision handling."""
        dt = self.DT
        geometry = GAUGE_CONFIGS[self.cfg_var.get()]['geometry']
        is_plate = geometry == 'plates'
        is_square = geometry == 'square_cavity'
        seg_dz = self.L_WIRE / self.N_SEG
        n = len(self.mol_pos)
        if n == 0:
            self.frame_count += 1
            return

        pos = self.mol_pos
        vel = self.mol_vel

        # ── Gravity/buoyancy forcing (global z convection direction) ──
        p_mbar = convert_pressure(10 ** self.sl_pressure.get(), 'mbar')
        p_on = max(self.CONVECTION_P_ON_MBAR, 1e-9)
        x_conv = (p_mbar / p_on) ** self.CONVECTION_TRANSITION_N
        conv_activation = x_conv / (1.0 + x_conv)
        if conv_activation > 1e-6:
            g_axis = 0 if self._is_horizontal_wire_mode() else 2
            t_mol = self._estimate_particle_temperatures_k()
            temp_drive = np.clip((t_mol - self.T_COLD) / max(self.T_HOT - self.T_COLD, 1.0), 0.0, 2.0)
            buoy = conv_activation * self.BUOYANCY_ACCEL * temp_drive
            vel[:, g_axis] += (self.GRAVITY_ACCEL + buoy) * dt
            # Keep forcing numerically stable at high pressures/speeds.
            vcap = np.maximum(2.5 * self.mol_thermal_speed[:n], 0.8)
            vel[:, g_axis] = np.clip(vel[:, g_axis], -vcap, vcap)

        # ── Vectorised Verlet-style position update ──
        pos += vel * dt

        if is_plate:
            wall_hits = np.zeros(n, dtype=bool)

            # — X / Y enclosure walls (cold, at T_COLD) —
            for dim in (0, 1):
                lo = pos[:, dim] < -self.R_ENC
                hi = pos[:, dim] > self.R_ENC
                pos[lo, dim] = -self.R_ENC + 0.05
                vel[lo, dim] = np.abs(vel[lo, dim])
                pos[hi, dim] = self.R_ENC - 0.05
                vel[hi, dim] = -np.abs(vel[hi, dim])
                wall_hits |= lo | hi

            # — Bottom plate (cold, at T_COLD) —
            bot = pos[:, 2] < 0
            pos[bot, 2] = 0.05
            vel[bot, 2] = np.abs(vel[bot, 2])
            wall_hits |= bot

            # Thermalize molecules hitting cold walls to T_COLD
            self._thermalize_at_wall(wall_hits)

            # — Top plate (hot filament) — energy transfer
            top = pos[:, 2] > self.L_WIRE
            if np.any(top):
                pos[top, 2] = self.L_WIRE - 0.05
                vel[top, 2] = -np.abs(vel[top, 2])
                seg_x = (pos[top, 0] + self.R_ENC) / (2.0 * self.R_ENC)
                seg_idx = np.clip((seg_x * self.N_SEG).astype(np.intp),
                                  0, self.N_SEG - 1)
                self._batch_collide(np.where(top)[0], seg_idx)
        elif is_square:
            # — Square-cavity geometry —
            wall_hits = np.zeros(n, dtype=bool)

            # Square side walls reflection (cold walls at T_COLD)
            for dim in (0, 1):
                lo = pos[:, dim] < -self.R_ENC
                hi = pos[:, dim] > self.R_ENC
                pos[lo, dim] = -self.R_ENC + 0.05
                vel[lo, dim] = np.abs(vel[lo, dim])
                pos[hi, dim] = self.R_ENC - 0.05
                vel[hi, dim] = -np.abs(vel[hi, dim])
                wall_hits |= lo | hi

            # Wire surface collision (still cylindrical wire in center)
            x = pos[:, 0]
            y = pos[:, 1]
            z = pos[:, 2]
            r_xy = np.sqrt(x * x + y * y)
            wire = (r_xy <= self.R_WIRE + 0.08) & (z >= 0) & (z <= self.L_WIRE)
            if np.any(wire):
                r_safe = np.maximum(r_xy[wire], 1e-6)
                nx = x[wire] / r_safe
                ny = y[wire] / r_safe
                vn = vel[wire, 0] * nx + vel[wire, 1] * ny
                inv = vn < 0
                vel[np.where(wire)[0][inv], 0] -= 2.0 * vn[inv] * nx[inv]
                vel[np.where(wire)[0][inv], 1] -= 2.0 * vn[inv] * ny[inv]
                pos[wire, 0] = (self.R_WIRE + 0.15) * nx
                pos[wire, 1] = (self.R_WIRE + 0.15) * ny
                seg_idx = np.clip((z[wire] / seg_dz).astype(np.intp),
                                  0, self.N_SEG - 1)
                self._batch_collide(np.where(wire)[0], seg_idx)

            # End caps (cold walls at T_COLD)
            lo_z = pos[:, 2] < 0
            hi_z = pos[:, 2] > self.L_WIRE
            pos[lo_z, 2] = 0.05
            vel[lo_z, 2] = np.abs(vel[lo_z, 2])
            pos[hi_z, 2] = self.L_WIRE - 0.05
            vel[hi_z, 2] = -np.abs(vel[hi_z, 2])
            wall_hits |= lo_z | hi_z

            # Thermalize molecules hitting cold walls to T_COLD
            self._thermalize_at_wall(wall_hits)
        else:
            # — Cylindrical geometry —
            wall_hits = np.zeros(n, dtype=bool)
            x = pos[:, 0]
            y = pos[:, 1]
            z = pos[:, 2]
            r_xy = np.sqrt(x * x + y * y)

            # Enclosure wall reflection (cold wall at T_COLD)
            enc = r_xy >= self.R_ENC
            if np.any(enc):
                r_safe = np.maximum(r_xy[enc], 1e-9)
                nx = x[enc] / r_safe
                ny = y[enc] / r_safe
                vn = vel[enc, 0] * nx + vel[enc, 1] * ny
                out = vn > 0
                vel[np.where(enc)[0][out], 0] -= 2.0 * vn[out] * nx[out]
                vel[np.where(enc)[0][out], 1] -= 2.0 * vn[out] * ny[out]
                pos[enc, 0] = (self.R_ENC - 0.06) * nx
                pos[enc, 1] = (self.R_ENC - 0.06) * ny
                wall_hits |= enc

            # Re-read after clamp
            x = pos[:, 0]; y = pos[:, 1]; z = pos[:, 2]
            r_xy = np.sqrt(x * x + y * y)

            # Wire surface collision (hot filament — energy transfer)
            wire = (r_xy <= self.R_WIRE + 0.08) & (z >= 0) & (z <= self.L_WIRE)
            if np.any(wire):
                r_safe = np.maximum(r_xy[wire], 1e-6)
                nx = x[wire] / r_safe
                ny = y[wire] / r_safe
                vn = vel[wire, 0] * nx + vel[wire, 1] * ny
                inv = vn < 0
                vel[np.where(wire)[0][inv], 0] -= 2.0 * vn[inv] * nx[inv]
                vel[np.where(wire)[0][inv], 1] -= 2.0 * vn[inv] * ny[inv]
                pos[wire, 0] = (self.R_WIRE + 0.15) * nx
                pos[wire, 1] = (self.R_WIRE + 0.15) * ny
                seg_idx = np.clip((z[wire] / seg_dz).astype(np.intp),
                                  0, self.N_SEG - 1)
                self._batch_collide(np.where(wire)[0], seg_idx)

            # End caps (cold walls at T_COLD)
            lo_z = pos[:, 2] < 0
            hi_z = pos[:, 2] > self.L_WIRE
            pos[lo_z, 2] = 0.05
            vel[lo_z, 2] = np.abs(vel[lo_z, 2])
            pos[hi_z, 2] = self.L_WIRE - 0.05
            vel[hi_z, 2] = -np.abs(vel[hi_z, 2])
            wall_hits |= lo_z | hi_z

            # Thermalize molecules hitting cold walls to T_COLD
            self._thermalize_at_wall(wall_hits)

        self._molecule_molecule_collisions()

        # ── Joule-heating recovery (exponential toward T_HOT) ──
        recovery = 1.0 - math.exp(-dt / self.HEATING_TAU)
        self.seg_temps += (self.T_HOT - self.seg_temps) * recovery

        # ── Thermal diffusion along filament ──
        if self.N_SEG > 2:
            T = self.seg_temps.copy()
            dc = min(self.DIFFUSION_K * dt / (seg_dz ** 2), 0.4)
            self.seg_temps[1:-1] += dc * (T[:-2] + T[2:] - 2 * T[1:-1])
            self.seg_temps[0] += dc * (T[1] - T[0])
            self.seg_temps[-1] += dc * (T[-2] - T[-1])

        inst_signal = self._step_cooling_accum / max(dt, 1e-12)
        if self.frame_count == 0:
            self.collision_signal_ema = inst_signal
        else:
            alpha = 0.03   # slow EMA — collision signal is a minor perturbation
            self.collision_signal_ema = (1.0 - alpha) * self.collision_signal_ema + alpha * inst_signal
        if inst_signal > 0:
            self.sensor_samples += 1
        self._step_cooling_accum = 0.0

        self._update_gas_ambient_temperature(wall_hits)
        self._update_particle_trails()

        self.seg_temps = np.clip(self.seg_temps, self.T_COLD, self.T_HOT + 5)
        self.frame_count += 1

    def _estimate_particle_temperatures_k(self):
        """Estimate per-particle kinetic temperature proxy from molecule speed."""
        n = len(self.mol_pos)
        if n == 0:
            return np.zeros(0)
        speeds = np.linalg.norm(self.mol_vel[:n], axis=1)
        if len(self.mol_thermal_speed) >= n:
            scale = np.maximum(self.mol_thermal_speed[:n], 1e-9)
        else:
            scale = np.full(n, self.MOL_SPEED_BASE * 0.65)
        # Rayleigh scale σ: <v²>=2σ², so T = T_COLD·v²/(2σ²) gives <T>=T_COLD
        t_est = 0.5 * self.T_COLD * (speeds / scale) ** 2
        return np.clip(t_est, self.T_COLD, self.T_HOT + 140.0)

    def _update_gas_ambient_temperature(self, wall_hits):
        """Update gas ambient with outer walls as thermal reservoir."""
        n = len(self.mol_pos)
        if n <= 0:
            self.gas_ambient_temp_k = self.T_COLD
            self.wall_coupling_ema = 0.0
            return

        wall_frac = float(np.count_nonzero(wall_hits)) / float(n)
        self.wall_coupling_ema = 0.88 * self.wall_coupling_ema + 0.12 * wall_frac

        t_kin = float(np.mean(self._estimate_particle_temperatures_k()))
        coupling = np.clip(0.25 + 1.8 * self.wall_coupling_ema, 0.25, 1.0)
        target = coupling * self.T_COLD + (1.0 - coupling) * t_kin

        relax = 0.06 + 0.20 * coupling
        self.gas_ambient_temp_k += relax * (target - self.gas_ambient_temp_k)
        # Gas can never be colder than the enclosure walls (coldest surface)
        self.gas_ambient_temp_k = float(np.clip(self.gas_ambient_temp_k, self.T_COLD, self.T_HOT + 140.0))

    def _get_gas_ambient_temperature_k(self):
        """Current gas ambient temperature inside chamber."""
        if len(self.mol_pos) <= 0:
            return self.T_COLD
        return float(np.clip(self.gas_ambient_temp_k, self.T_COLD, self.T_HOT + 140.0))

    def _update_particle_trails(self):
        """Append the latest molecule positions and temperatures to trail history."""
        n = len(self.mol_pos)
        if n == 0:
            return
        if self.mol_trail_pos.shape[0] != n:
            temp_now = self._estimate_particle_temperatures_k()
            self.mol_trail_pos = np.repeat(self.mol_pos[:, np.newaxis, :], self.TRAIL_LENGTH, axis=1)
            self.mol_trail_temp = np.repeat(temp_now[:, np.newaxis], self.TRAIL_LENGTH, axis=1)
            return

        self.mol_trail_pos = np.roll(self.mol_trail_pos, -1, axis=1)
        self.mol_trail_pos[:, -1, :] = self.mol_pos
        self.mol_trail_temp = np.roll(self.mol_trail_temp, -1, axis=1)
        self.mol_trail_temp[:, -1] = self._estimate_particle_temperatures_k()

    def _is_horizontal_wire_mode(self):
        """True when the active cylindrical gauge is configured as horizontal wire."""
        cfg = GAUGE_CONFIGS.get(self.cfg_var.get(), {})
        return cfg.get('geometry') == 'cylindrical' and cfg.get('orientation') == 'horizontal'

    def _render_coords(self, x, y, z):
        """Map internal coordinates to display coordinates.

        Physics is solved in a canonical cylindrical frame with wire axis along z.
        For horizontal-wire mode we rotate the rendered scene so the wire axis
        appears along x.
        """
        if not self._is_horizontal_wire_mode():
            return x, y, z
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        z_arr = np.asarray(z)
        xr = z_arr - 0.5 * self.L_WIRE
        yr = y_arr
        zr = x_arr
        return xr, yr, zr

    def _draw_temperature_trails(self, ax, cmap, tnorm):
        """Draw color-mapped trajectory trails for each particle."""
        n = len(self.mol_pos)
        if n == 0 or self.mol_trail_pos.shape[0] != n or self.TRAIL_LENGTH < 2:
            return

        segments = []
        colors = []
        denom = max(self.TRAIL_LENGTH - 1, 1)

        for j in range(n):
            pts = self.mol_trail_pos[j]
            temps = self.mol_trail_temp[j]
            for i in range(1, self.TRAIL_LENGTH):
                p0 = pts[i - 1]
                p1 = pts[i]
                if not (np.all(np.isfinite(p0)) and np.all(np.isfinite(p1))):
                    continue
                p0x, p0y, p0z = self._render_coords(p0[0], p0[1], p0[2])
                p1x, p1y, p1z = self._render_coords(p1[0], p1[1], p1[2])
                segments.append([[float(p0x), float(p0y), float(p0z)],
                                 [float(p1x), float(p1y), float(p1z)]])
                rgba = list(cmap(tnorm(float(temps[i]))))
                age = i / denom
                rgba[3] = 0.10 + 0.50 * age
                colors.append(tuple(rgba))

        if not segments:
            return

        trail_collection = art3d.Line3DCollection(
            segments,
            colors=colors,
            linewidths=0.9,
            zorder=4,
        )
        ax.add_collection3d(trail_collection)

    def _batch_collide(self, mol_indices, seg_indices):
        """Process collisions for all molecules in *mol_indices* at once.
        Uses precomputed mol_aE / mol_f_plus1 arrays.  Fully vectorised."""
        if len(mol_indices) == 0:
            return
        cool_coeff = self.COOL_BASE / 6.0
        dT_range = self.T_HOT - self.T_COLD + 1e-9

        mi = np.asarray(mol_indices)
        si = np.asarray(seg_indices)

        T_segs = self.seg_temps[si]
        speeds = np.linalg.norm(self.mol_vel[mi], axis=1)
        speed_ref = np.maximum(self.mol_thermal_speed[mi], 1e-9)
        # Rayleigh scale σ: <v²>=2σ², so T = T_COLD·v²/(2σ²) gives <T>=T_COLD
        T_mol = np.clip(0.5 * self.T_COLD * (speeds / speed_ref) ** 2,
                        self.T_COLD,
                        self.T_HOT + 200.0)

        # Bidirectional exchange: if molecule hotter than wire, wire heats up;
        # if wire hotter than molecule, wire cools down.
        temp_diff = T_segs - T_mol
        exchange = self.mol_aE[mi] * self.mol_f_plus1[mi] * cool_coeff * (temp_diff / dT_range)
        exchange = np.clip(exchange, -2.0 * cool_coeff, 2.0 * cool_coeff)

        # Apply wire temperature change per segment using np.add.at for duplicates
        np.add.at(self.seg_temps, si, -exchange)
        np.clip(self.seg_temps, self.T_COLD, None, out=self.seg_temps)

        # Molecule-side update: move kinetic temperature toward local segment temp
        # while preserving direction (speed magnitude scaling only).
        relax = np.clip(0.35 * self.mol_aE[mi], 0.05, 0.70)
        T_mol_new = np.clip(T_mol + relax * temp_diff,
                            self.T_COLD,
                            self.T_HOT + 200.0)
        speed_scale = np.sqrt(np.maximum(T_mol_new, 1e-9) / np.maximum(T_mol, 1e-9))
        self.mol_vel[mi] *= speed_scale[:, np.newaxis]

        total_cool = exchange.sum()
        n_collisions = len(mi)
        self.collision_count += n_collisions
        self.total_energy_transferred += total_cool
        self._step_cooling_accum += total_cool

        # Count per-gas collisions
        for gk in set(self.mol_gas_keys[j] for j in mi):
            count = int(np.sum(np.array([self.mol_gas_keys[j] for j in mi]) == gk))
            self.collision_per_gas[gk] = self.collision_per_gas.get(gk, 0) + count

    def _thermalize_at_wall(self, mask):
        """Re-sample molecule speeds to Maxwell–Boltzmann at T_COLD.

        When a molecule hits a cold enclosure wall it thermalises: the
        reflected direction is preserved but the speed magnitude is redrawn
        from a Rayleigh distribution at the wall temperature (T_COLD).
        This closes the energy-transport cycle in the Pirani gauge:
            hot filament → molecule picks up energy → cold wall absorbs it.
        """
        if not np.any(mask):
            return
        indices = np.where(mask)[0]
        new_speeds = np.random.rayleigh(self.mol_thermal_speed[indices])
        current_speeds = np.linalg.norm(self.mol_vel[indices], axis=1)
        current_speeds = np.maximum(current_speeds, 1e-8)
        scale = new_speeds / current_speeds
        self.mol_vel[indices] *= scale[:, np.newaxis]

    def _molecule_molecule_collisions(self):
        """Apply temperature-dependent molecule-molecule speed exchange.

        Molecules are randomly paired each step. For close pairs, kinetic
        temperatures relax toward each other, with stronger transfer when
        temperature difference is larger.
        """
        n = len(self.mol_pos)
        if n < 2:
            return

        pair_count = n // 2
        if pair_count <= 0:
            return
        perm = np.random.permutation(n)
        pairs = perm[:2 * pair_count].reshape(pair_count, 2)
        a = pairs[:, 0]
        b = pairs[:, 1]

        da = self.mol_pos[a] - self.mol_pos[b]
        close = np.einsum('ij,ij->i', da, da) <= (self.MOL_COLLISION_RADIUS ** 2)
        if not np.any(close):
            return

        ia = a[close]
        ib = b[close]

        va = self.mol_vel[ia]
        vb = self.mol_vel[ib]
        sa = np.maximum(np.linalg.norm(va, axis=1), 1e-9)
        sb = np.maximum(np.linalg.norm(vb, axis=1), 1e-9)

        ref_a = np.maximum(self.mol_thermal_speed[ia], 1e-9)
        ref_b = np.maximum(self.mol_thermal_speed[ib], 1e-9)
        # Rayleigh scale σ: <v²>=2σ², so T = T_COLD·v²/(2σ²) gives <T>=T_COLD
        Ta = np.clip(0.5 * self.T_COLD * (sa / ref_a) ** 2, self.T_COLD, self.T_HOT + 200.0)
        Tb = np.clip(0.5 * self.T_COLD * (sb / ref_b) ** 2, self.T_COLD, self.T_HOT + 200.0)

        rel = np.clip(np.abs(Ta - Tb) / max(self.T_HOT - self.T_COLD, 1.0), 0.0, 2.0)
        relax = np.clip(self.MOL_MM_RELAX * (0.5 + 0.5 * rel), 0.08, 0.65)

        Ta_new = Ta + relax * (Tb - Ta)
        Tb_new = Tb + relax * (Ta - Tb)

        sa_new = sa * np.sqrt(np.maximum(Ta_new, 1e-9) / np.maximum(Ta, 1e-9))
        sb_new = sb * np.sqrt(np.maximum(Tb_new, 1e-9) / np.maximum(Tb, 1e-9))

        self.mol_vel[ia] *= (sa_new / sa)[:, np.newaxis]
        self.mol_vel[ib] *= (sb_new / sb)[:, np.newaxis]

    # ── Drawing ──────────────────────────────────────────────────────────────

    def _draw_scene(self):
        """Render the complete 3D scene (filament + enclosure + molecules)."""
        # Preserve camera angle across redraws (skip if a preset was just set)
        if self._camera_preset_pending:
            self._camera_preset_pending = False
        elif hasattr(self, 'ax3d') and self.ax3d is not None:
            try:
                self._elev = self.ax3d.elev
                self._azim = self.ax3d.azim
            except Exception:
                pass

        self.fig.clear()
        with plt.rc_context(MPL_STYLE):
            self.fig.patch.set_facecolor(COLORS['bg_card'])
            self.ax3d = self.fig.add_subplot(111, projection='3d')
            ax = self.ax3d
            ax.set_facecolor(COLORS['bg_input'])

            cmap = matplotlib.colormaps[self.filament_cmap_var.get()]
            color_min = self.sl_color_min.get()
            color_max = self.sl_color_max.get()
            if color_min >= color_max:
                color_max = color_min + 1.0
            tnorm = mcolors.Normalize(vmin=color_min, vmax=color_max)
            is_plate = GAUGE_CONFIGS[self.cfg_var.get()]['geometry'] == 'plates'

            if is_plate:
                self._draw_plate(ax, cmap, tnorm)
            else:
                self._draw_wire(ax, cmap, tnorm)

            # Molecules — grouped by gas species for colour & size
            if len(self.mol_pos) > 0:
                if self.show_temp_trails.get():
                    self._draw_temperature_trails(ax, cmap, tnorm)

                gas_groups = {}
                for j, gk in enumerate(self.mol_gas_keys):
                    gas_groups.setdefault(gk, []).append(j)

                for gk, indices in gas_groups.items():
                    gas = GAS_DATA[gk]
                    idx = np.array(indices)
                    sz = self.MOL_SIZE_BASE * (self.MOL_RADII[gk] / self.MOL_RADII['N2']) ** 2
                    mx, my, mz = self._render_coords(self.mol_pos[idx, 0],
                                                     self.mol_pos[idx, 1],
                                                     self.mol_pos[idx, 2])
                    ax.scatter(mx, my, mz,
                               s=sz, c=gas['color'], alpha=0.85,
                               edgecolors='white', linewidths=0.4,
                               depthshade=True, zorder=5, label=gas['symbol'])

                if self.show_vectors.get():
                    for j in range(len(self.mol_pos)):
                        gk = self.mol_gas_keys[j] if j < len(self.mol_gas_keys) else 'N2'
                        p = self.mol_pos[j]
                        v = self.mol_vel[j] * 0.25
                        p0x, p0y, p0z = self._render_coords(p[0], p[1], p[2])
                        p1x, p1y, p1z = self._render_coords(p[0] + v[0], p[1] + v[1], p[2] + v[2])
                        ax.plot([float(p0x), float(p1x)], [float(p0y), float(p1y)],
                                [float(p0z), float(p1z)], color=GAS_DATA[gk]['color'],
                                alpha=0.45, linewidth=0.7)

            # Axes setup
            lim = self.R_ENC * 1.1
            if is_plate:
                ax.set_xlim(-lim, lim)
                ax.set_ylim(-lim, lim)
                ax.set_zlim(-0.5, self.L_WIRE + 0.5)
            elif self._is_horizontal_wire_mode():
                ax.set_xlim(-0.5 * self.L_WIRE - 0.5, 0.5 * self.L_WIRE + 0.5)
                ax.set_ylim(-lim, lim)
                ax.set_zlim(-lim, lim)
            else:
                ax.set_xlim(-lim, lim)
                ax.set_ylim(-lim, lim)
                ax.set_zlim(-0.5, self.L_WIRE + 0.5)

            # Build title from mixture
            fracs = self._get_mixture_fractions()
            if len(fracs) == 1:
                gk = list(fracs.keys())[0]
                title = f"Molecular Energy Transfer — {GAS_DATA[gk]['name']} ({GAS_DATA[gk]['symbol']})"
            else:
                parts = [f"{GAS_DATA[k]['symbol']} {v*100:.0f}%" for k, v in
                         sorted(fracs.items(), key=lambda x: -x[1])[:4]]
                title = 'Molecular Energy Transfer — ' + '  +  '.join(parts)
                if len(fracs) > 4:
                    title += '  + …'
            ax.set_title(title, fontsize=10, color=COLORS['text_bright'], pad=10)
            ax.set_xlabel('x', fontsize=8, labelpad=-2)
            ax.set_ylabel('y', fontsize=8, labelpad=-2)
            ax.set_zlabel('z', fontsize=8, labelpad=-2)
            ax.tick_params(colors=COLORS['text_dim'])
            try:
                ax.xaxis.pane.fill = False
                ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False
                ax.xaxis.pane.set_edgecolor(COLORS['border'])
                ax.yaxis.pane.set_edgecolor(COLORS['border'])
                ax.zaxis.pane.set_edgecolor(COLORS['border'])
            except Exception:
                pass

            regime, kn, lambda_m = self._get_flow_regime_info(self._calc_real_pressure_pa())
            regime_text = (
                f"Flow Regime: {regime}\n"
                f"Knudsen #: {kn:.3g}\n"
                f"Mean Free Path: {lambda_m*1e3:.3g} mm\n"
                "Bands: Mol>1 | Trans 0.01-1 | Visc<0.01"
            )
            ax.text2D(
                0.998,
                0.985,
                regime_text,
                transform=ax.transAxes,
                ha='right',
                va='top',
                fontsize=7.5,
                color=COLORS['text_bright'],
                bbox=dict(
                    boxstyle='round,pad=0.28',
                    facecolor=COLORS['bg_card'],
                    edgecolor=COLORS['border'],
                    alpha=0.9,
                ),
            )

            ax.view_init(elev=self._elev, azim=self._azim)

        self.fig.tight_layout()
        self.canvas.draw_idle()
        self._update_stats()

    def _draw_wire(self, ax, cmap, tnorm):
        """Render the cylindrical filament with per-segment temperature colour."""
        cfg = GAUGE_CONFIGS.get(self.cfg_var.get(), {})
        is_square = cfg.get('geometry') == 'square_cavity'
        theta = np.linspace(0, 2 * np.pi, self.N_THETA)
        z_edges = np.linspace(0, self.L_WIRE, self.N_SEG + 1)
        Theta, Z = np.meshgrid(theta, z_edges)
        X = self.R_WIRE * np.cos(Theta)
        Y = self.R_WIRE * np.sin(Theta)
        Xr, Yr, Zr = self._render_coords(X, Y, Z)

        fc = np.zeros((self.N_SEG, self.N_THETA - 1, 4))
        for i in range(self.N_SEG):
            fc[i, :] = cmap(tnorm(self.seg_temps[i]))
        ax.plot_surface(Xr, Yr, Zr, facecolors=fc, shade=False,
                        rstride=1, cstride=1, antialiased=False)

        # Wire end-caps (small discs)
        t = np.linspace(0, 2 * np.pi, self.N_THETA)
        r = np.linspace(0, self.R_WIRE, 4)
        T2d, R2d = np.meshgrid(t, r)
        Xc = R2d * np.cos(T2d)
        Yc = R2d * np.sin(T2d)
        # bottom cap
        Zb = np.zeros_like(Xc)
        fc_bot = np.full((*Zb.shape, 4), cmap(tnorm(self.seg_temps[0])))
        Xbr, Ybr, Zbr = self._render_coords(Xc, Yc, Zb)
        ax.plot_surface(Xbr, Ybr, Zbr, facecolors=fc_bot, shade=False, antialiased=False)
        # top cap
        Zt = np.full_like(Xc, self.L_WIRE)
        fc_top = np.full((*Zt.shape, 4), cmap(tnorm(self.seg_temps[-1])))
        Xtr, Ytr, Ztr = self._render_coords(Xc, Yc, Zt)
        ax.plot_surface(Xtr, Ytr, Ztr, facecolors=fc_top, shade=False, antialiased=False)

        # Enclosure wireframe
        if self.show_enclosure.get():
            if is_square:
                e = self.R_ENC
                corners = np.array([
                    [-e, -e],
                    [ e, -e],
                    [ e,  e],
                    [-e,  e],
                    [-e, -e],
                ])
                for zp in (0.0, self.L_WIRE):
                    x = corners[:, 0]
                    y = corners[:, 1]
                    z = np.full_like(x, zp)
                    xr, yr, zr = self._render_coords(x, y, z)
                    ax.plot(xr, yr, zr, color=COLORS['accent'], alpha=0.35, linewidth=0.9)
                for x0, y0 in corners[:-1]:
                    x = np.array([x0, x0], dtype=float)
                    y = np.array([y0, y0], dtype=float)
                    z = np.array([0.0, self.L_WIRE], dtype=float)
                    xr, yr, zr = self._render_coords(x, y, z)
                    ax.plot(xr, yr, zr, color=COLORS['accent'], alpha=0.18, linewidth=0.55)
            else:
                te = np.linspace(0, 2 * np.pi, 40)
                for zp in (0, self.L_WIRE):
                    x = self.R_ENC * np.cos(te)
                    y = self.R_ENC * np.sin(te)
                    z = np.full_like(te, zp)
                    xr, yr, zr = self._render_coords(x, y, z)
                    ax.plot(xr, yr, zr, color=COLORS['accent'], alpha=0.3, linewidth=0.8)
                for a in np.linspace(0, 2 * np.pi, 10, endpoint=False):
                    x = np.array([self.R_ENC * math.cos(a)] * 2)
                    y = np.array([self.R_ENC * math.sin(a)] * 2)
                    z = np.array([0.0, self.L_WIRE])
                    xr, yr, zr = self._render_coords(x, y, z)
                    ax.plot(xr, yr, zr, color=COLORS['accent'], alpha=0.15, linewidth=0.5)

    def _draw_plate(self, ax, cmap, tnorm):
        """Render parallel-plate geometry with per-segment temperature colour."""
        x_edges = np.linspace(-self.R_ENC, self.R_ENC, self.N_SEG + 1)
        y_edges = np.array([-self.R_ENC, self.R_ENC])
        X, Yg = np.meshgrid(x_edges, y_edges)
        Z_top = np.full_like(X, self.L_WIRE)

        fc = np.zeros((1, self.N_SEG, 4))
        for i in range(self.N_SEG):
            fc[0, i] = cmap(tnorm(self.seg_temps[i]))
        ax.plot_surface(X, Yg, Z_top, facecolors=fc, shade=False,
                        rstride=1, cstride=1, antialiased=False)

        # Cold bottom plate
        Z_bot = np.zeros_like(X)
        ax.plot_surface(X, Yg, Z_bot, alpha=0.35, color=COLORS['accent'])

        # Box outline
        if self.show_enclosure.get():
            e = self.R_ENC
            corners = [(-e, -e), (e, -e), (e, e), (-e, e), (-e, -e)]
            for (x1, y1), (x2, y2) in zip(corners[:-1], corners[1:]):
                ax.plot([x1, x2], [y1, y2], [0, 0],
                        color=COLORS['accent'], alpha=0.3, lw=0.8)
                ax.plot([x1, x2], [y1, y2], [self.L_WIRE] * 2,
                        color='#ff6b35', alpha=0.3, lw=0.8)
                ax.plot([x1, x1], [y1, y1], [0, self.L_WIRE],
                        color=COLORS['text_dim'], alpha=0.15, lw=0.5)

    # ── Stats display ────────────────────────────────────────────────────────

    def _update_stats(self):
        fracs = self._get_mixture_fractions()
        T_min = self.seg_temps.min()
        T_max = self.seg_temps.max()
        T_avg = self.seg_temps.mean()
        t_unit = get_temperature_unit()
        unit = get_pressure_unit()

        # Update live pressure readout
        p_set, p_real, p_ind, vol_m3 = self._update_live_pressure()
        p_ind_avg = self._get_avg_pirani_reading_pa()
        self._update_sim_clock()

        if p_ind_avg is not None:
            p_err_abs = p_ind_avg - p_real
            p_err_pct = (p_err_abs / max(p_real, 1e-20)) * 100.0
            p_err_text = f"{format_pressure(p_err_abs, unit, fmt='{:+.3g}')} ({p_err_pct:+.1f}%)"
        else:
            p_err_text = '—'

        # Collision rate (per frame)
        col_rate = self.collision_count / max(self.frame_count, 1)

        # Mixture summary
        mix_parts = []
        for gk, fr in sorted(fracs.items(), key=lambda x: -x[1]):
            gas = GAS_DATA[gk]
            hits = self.collision_per_gas.get(gk, 0)
            mix_parts.append(f"  {gas['symbol']:>4s} {fr*100:5.1f}%  αE={self._get_accommodation_for(gk):.2f}  hits={hits}")

        represented_molecules = len(self.mol_pos) * self.molecules_per_particle

        # Expected molecular-regime correction factor for current mixture
        cf_mix = self._get_empirical_cf_mix(fracs)
        acc_spec_label = self._pirani_accuracy_label(p_real)

        self.stats_label.config(text=(
            f"Setpoint p: {format_pressure(p_set, unit)}\n"
            f"Real p (N,V,T): {format_pressure(p_real, unit)}\n"
            f"Pirani p_ind (N₂ cal): {format_pressure(p_ind, unit)}\n"
            f"Pirani avg ({len(self.pirani_readings_pa)}): {format_pressure(p_ind_avg, unit) if p_ind_avg is not None else '—'}\n"
            f"Avg error vs real: {p_err_text}\n"
            f"Correction factor: {cf_mix:.3f}  (accuracy {acc_spec_label})\n"
            f"Molecules: {len(self.mol_pos)}  (represents {represented_molecules:.2e})\n"
            f"Defined volume: {vol_m3 * 1e6:.3g} cm³\n"
            f"Ambient gas: {format_temperature(self._get_gas_ambient_temperature_k(), t_unit)}  (walls: {format_temperature(self.T_COLD, t_unit)})\n"
            f"───────────────────\n"
            + '\n'.join(mix_parts) + '\n'
            f"───────────────────\n"
            f"Wire: {convert_temperature(T_min, 'K', t_unit):.1f}–{convert_temperature(T_max, 'K', t_unit):.1f} {TEMPERATURE_UNITS[t_unit]['label']}  (eq {convert_temperature(self.T_HOT, 'K', t_unit):.0f})\n"
            f"Mean T: {convert_temperature(T_avg, 'K', t_unit):.1f} {TEMPERATURE_UNITS[t_unit]['label']}\n"
            f"Collisions: {self.collision_count}\n"
            f"Rate: {col_rate:.1f} /frame\n"
            f"ΔT total: {self.total_energy_transferred:.1f} K"
        ))

    # ── Animation loop ───────────────────────────────────────────────────────

    def _animate(self):
        if not self.running:
            return

        try:
            now = time.perf_counter()
            if self._last_anim_tick is None:
                self._last_anim_tick = now
            elapsed = max(0.0, min(now - self._last_anim_tick, 0.2))
            self._last_anim_tick = now

            nominal = max(self.INTERVAL / 1000.0, 1e-6)
            speed = max(0.1, float(self.sl_speed.get()))
            self._substep_remainder += speed * (elapsed / nominal)
            n_sub = int(self._substep_remainder)
            if n_sub <= 0:
                n_sub = 1
            n_sub = min(n_sub, 24)
            self._substep_remainder = max(0.0, self._substep_remainder - n_sub)

            for _ in range(n_sub):
                self._step()

            draw_every = max(1, int(round(self.DRAW_EVERY * max(1.0, speed / 2.0))))
            if self.frame_count - self._last_draw_frame >= draw_every:
                self._draw_scene()
                self._last_draw_frame = self.frame_count
            self.anim_id = self.after(self.INTERVAL, self._animate)
        except Exception:
            # Keep UI recoverable if a runtime callback ever throws.
            traceback.print_exc()
            self._stop_simulation(set_start_label=False)
            self.btn_play.config(text='▶  Resume')

    def _stop_simulation(self, set_start_label=False):
        """Stop animation and cancel pending callback safely."""
        self.running = False
        if self.anim_id is not None:
            try:
                self.after_cancel(self.anim_id)
            except Exception:
                pass
            self.anim_id = None
        self._last_anim_tick = None
        self._substep_remainder = 0.0
        if set_start_label:
            self.btn_play.config(text='▶  Start Simulation')
        else:
            self.btn_play.config(text='▶  Resume')

    def _start_simulation(self):
        """Start animation from a clean callback state."""
        # Prevent stale callbacks from previous runs from interfering.
        if self.anim_id is not None:
            try:
                self.after_cancel(self.anim_id)
            except Exception:
                pass
            self.anim_id = None
        self._flush_pending_molecule_reinit()
        self.running = True
        self._last_anim_tick = time.perf_counter()
        self.btn_play.config(text='⏸  Pause')
        self._animate()

    def _toggle_play(self):
        if self.running:
            self._stop_simulation(set_start_label=False)
            return
        try:
            self._start_simulation()
        except Exception:
            traceback.print_exc()
            self._stop_simulation(set_start_label=False)

    def _reset(self):
        self._cancel_pending_reinit()
        self._pending_reinit_draw = False
        self._stop_simulation(set_start_label=True)
        self.seg_temps = np.full(self.N_SEG, self.T_HOT, dtype=np.float64)
        self.collision_count = 0
        self.total_energy_transferred = 0.0
        self.collision_per_gas = {}
        self.frame_count = 0
        self._step_cooling_accum = 0.0
        self.collision_signal_ema = 0.0
        self.sensor_samples = 0
        self.sensor_gain_q = None
        self._gain_fast_adapt = 0
        self._substep_remainder = 0.0
        self._last_anim_tick = None
        self._session_bias = float(np.random.default_rng().normal(0, 0.4))
        self._init_molecules()
        self._draw_scene()
        self.btn_play.config(text='▶  Start Simulation')

    def _set_camera(self, elev, azim):
        """Set the 3D camera to a preset angle and redraw."""
        self._elev = elev
        self._azim = azim
        self._camera_preset_pending = True
        self._draw_scene()

    # ── Control callbacks ────────────────────────────────────────────────────

    def _on_config_change(self):
        cfg = GAUGE_CONFIGS.get(self.cfg_var.get(), {})
        # Load sensor/bridge parameters from the selected gauge
        sensor = cfg.get('sensor', {})
        if sensor:
            self.sensor_r0_ohm = sensor.get('r0_ohm', self.sensor_r0_ohm)
            self.sensor_tcr_per_k = sensor.get('tcr_per_k', self.sensor_tcr_per_k)
            self.sensor_emissivity = sensor.get('emissivity', self.sensor_emissivity)
            self.bridge_v_bias = sensor.get('bridge_v_bias', self.bridge_v_bias)
            self.bridge_v_sensor = sensor.get('bridge_v_sensor', self.bridge_v_sensor)
            self.sensor_support_lambda_wmk = sensor.get('support_lambda_wmk', self.sensor_support_lambda_wmk)
            self.sensor_support_w_m = sensor.get('support_w_m', self.sensor_support_w_m)
            self.sensor_support_t_m = sensor.get('support_t_m', self.sensor_support_t_m)
            self.sensor_support_l_m = sensor.get('support_l_m', self.sensor_support_l_m)
            self.sensor_extra_support_g_wpk = sensor.get('extra_support_g_wpk', self.sensor_extra_support_g_wpk)
        # Update hot/cold temperatures from config
        t_hot_lo, t_hot_hi = cfg.get('t_hot_range_k', (313.0, 673.0))
        t_cold_lo, t_cold_hi = cfg.get('t_cold_range_k', (200.0, 350.0))
        self.sl_wire_temp.set_range(t_hot_lo, t_hot_hi)
        self.sl_env_temp.set_range(t_cold_lo, t_cold_hi)
        new_t_hot = float(cfg.get('T1', self.T_HOT))
        new_t_cold = float(cfg.get('T2', self.T_COLD))
        if new_t_hot != self.T_HOT:
            self.T_HOT = float(new_t_hot)
            self.sl_wire_temp.set(self.T_HOT)
        if new_t_cold != self.T_COLD:
            self.T_COLD = float(new_t_cold)
            self.gas_ambient_temp_k = self.T_COLD
            self.sl_env_temp.set(self.T_COLD)
        self.seg_temps = np.full(self.N_SEG, self.T_HOT, dtype=np.float64)
        self._auto_update_color_range()
        self._schedule_molecule_reinit(draw_if_idle=True, delay_ms=60)

    def _on_pressure_change(self):
        self._schedule_molecule_reinit(draw_if_idle=True, delay_ms=90)
        self._update_live_pressure()

    def _on_avg_window_change(self):
        """Called when rolling Pirani average window size is changed."""
        new_n = max(5, int(round(self.sl_avg_samples.get())))
        old_vals = list(self.pirani_readings_pa)
        self.pirani_readings_pa = deque(old_vals[-new_n:], maxlen=new_n)
        if not self.running:
            self._update_stats()

    def _on_env_temp_change(self):
        """Called when user adjusts the external environment temperature."""
        new_temp = self.sl_env_temp.get()
        if new_temp >= self.T_HOT:
            new_temp = self.T_HOT - 1
            self.sl_env_temp.set(new_temp)
        self.T_COLD = float(new_temp)
        # Gas equilibrates with the walls (coldest thermal reservoir)
        self.gas_ambient_temp_k = self.T_COLD
        # Clamp wire segments — can't be below the new wall temperature
        self.seg_temps = np.clip(self.seg_temps, self.T_COLD, self.T_HOT + 5)
        self._auto_update_color_range()
        self._sync_temperature_unit_display()
        # Rebuild molecules after slider interaction settles for smoother UI.
        self._schedule_molecule_reinit(draw_if_idle=True, delay_ms=120)
        self._update_live_pressure()

    def _on_wire_temp_change(self):
        """Called when user adjusts the filament equilibrium temperature."""
        new_temp = self.sl_wire_temp.get()
        if new_temp <= self.T_COLD:
            new_temp = self.T_COLD + 1
            self.sl_wire_temp.set(new_temp)
        self.T_HOT = float(new_temp)
        self._sync_temperature_unit_display()
        # Reset filament to new equilibrium and re-init molecules
        self.seg_temps = np.full(self.N_SEG, self.T_HOT, dtype=np.float64)
        self._auto_update_color_range()
        self._schedule_molecule_reinit(draw_if_idle=True, delay_ms=120)
        self._update_live_pressure()

    def _on_pressure_unit_change(self):
        """Called when the user toggles the pressure unit."""
        self._sync_pressure_slider_display()
        self._update_live_pressure()
        if not self.running:
            self._update_stats()

    def _on_color_range_change(self):
        """Called when user changes filament colormap or temperature bounds."""
        # Rebuild the temperature legend with new settings
        for child in self.legend_frame.winfo_children():
            child.destroy()
        self._build_temp_legend(self.legend_frame)
        if not self.running:
            self._draw_scene()

    def _get_default_color_window(self):
        """Return a robust default color window tied to current cold/hot temperatures."""
        span = max(self.T_HOT - self.T_COLD, 8.0)
        cmin = self.T_COLD - max(4.0, 0.15 * span)
        cmax = self.T_HOT + max(5.0, 0.25 * span)
        cmin = float(np.clip(cmin, 180.0, 780.0))
        cmax = float(np.clip(cmax, cmin + 2.0, 800.0))
        return cmin, cmax

    def _auto_update_color_range(self, force=False):
        """Auto-apply color min/max from active gauge temperatures when enabled."""
        if not force and not self.auto_color_scale_var.get():
            return
        cmin, cmax = self._get_default_color_window()
        self.sl_color_min.set(cmin)
        self.sl_color_max.set(cmax)
        self._on_color_range_change()

    def _reset_color_range(self):
        """Reset filament color range to defaults based on T_COLD/T_HOT."""
        self.filament_cmap_var.set('coolwarm')
        self.auto_color_scale_var.set(True)
        self._auto_update_color_range(force=True)

    # ── Tab visibility (auto-pause when user leaves this tab) ────────────

    def on_tab_hidden(self):
        """Called when user switches away from the Molecular Sim tab."""
        if self.running:
            self._was_running_before_hide = True
            self._stop_simulation(set_start_label=False)
        else:
            self._was_running_before_hide = False

    def on_tab_shown(self):
        """Called when user returns to the Molecular Sim tab."""
        if self._was_running_before_hide:
            self._was_running_before_hide = False
            try:
                self._start_simulation()
            except Exception:
                traceback.print_exc()
                self._stop_simulation(set_start_label=False)

    def on_global_units_changed(self):
        self._sync_pressure_slider_display()
        self._sync_temperature_unit_display()
        self._update_live_pressure()
        for child in self.legend_frame.winfo_children():
            child.destroy()
        self._build_temp_legend(self.legend_frame)
        if not self.running:
            self._draw_scene()

    def on_theme_changed(self):
        """Refresh all native Tk widgets that don't respond to ttk style changes."""
        # Gas-color dot canvases
        for dot in self._gas_dot_canvases:
            dot.configure(bg=COLORS['bg_card'])

        # Composition bar figure background
        self.comp_fig.patch.set_facecolor(COLORS['bg_card'])
        self._update_composition_display()

        # Temperature legend (rebuilt with current colors)
        for child in self.legend_frame.winfo_children():
            child.destroy()
        self._build_temp_legend(self.legend_frame)

        # Stats are refreshed implicitly by _draw_scene

    def _sync_pressure_slider_display(self):
        unit = get_pressure_unit()
        self.sl_pressure.set_value_formatter(
            lambda v: format_pressure(10 ** v, unit, fmt='{:.4g}')
        )

    def _sync_temperature_unit_display(self):
        """Update Molecular Sim temperature slider labels to selected unit."""
        t_unit = get_temperature_unit()
        self.sl_env_temp.set_value_formatter(
            lambda v: format_temperature(v, t_unit, fmt='{:.1f}')
        )
        self.sl_wire_temp.set_value_formatter(
            lambda v: format_temperature(v, t_unit, fmt='{:.1f}')
        )
        if hasattr(self, 'sl_color_min'):
            self.sl_color_min.set_value_formatter(
                lambda v: format_temperature(v, t_unit, fmt='{:.1f}')
            )
        if hasattr(self, 'sl_color_max'):
            self.sl_color_max.set_value_formatter(
                lambda v: format_temperature(v, t_unit, fmt='{:.1f}')
            )

    def _get_active_sim_config(self):
        """Return active gauge config with current temperatures.

        T1 = hot filament, T2 = enclosure wall (T_COLD) per Jousten (2008).
        """
        cfg = dict(GAUGE_CONFIGS[self.cfg_var.get()])
        cfg['T1'] = self.T_HOT
        cfg['T2'] = self.T_COLD
        return cfg

    def _get_defined_volume_m3(self):
        """Physical gas volume used for real-pressure estimate."""
        cfg = self._get_active_sim_config()
        if cfg['geometry'] in ('cylindrical', 'square_cavity'):
            r1 = cfg['wire_r']
            r2 = cfg['enc_r']
            L = cfg['wire_L']
            if cfg['geometry'] == 'square_cavity':
                return max(((2.0 * r2) ** 2 - np.pi * r1 * r1) * L, 1e-18)
            return max(np.pi * (r2 * r2 - r1 * r1) * L, 1e-18)
        area = cfg['plate_area']
        gap = cfg['gap']
        return max(area * gap, 1e-18)

    def _get_flow_regime_info(self, p_real_pa=None):
        """Return (label, Kn, lambda_m) using current gas and geometry scale."""
        if p_real_pa is None:
            p_real_pa = self._calc_real_pressure_pa()
        p_mbar = convert_pressure(max(p_real_pa, 1e-20), 'mbar')
        gas = self._get_gas()
        lambda_m = gas['plbar'] / max(p_mbar, 1e-20)

        cfg = self._get_active_sim_config()
        if cfg['geometry'] == 'plates':
            d_char = max(cfg['gap'], 1e-12)
        elif cfg['geometry'] == 'square_cavity':
            d_char = max(2.0 * cfg['enc_r'], 1e-12)
        else:
            d_char = max(cfg['enc_r'], 1e-12)

        kn = lambda_m / d_char
        if kn > 1.0:
            regime = 'Molecular'
        elif kn > 0.01:
            regime = 'Transition'
        else:
            regime = 'Viscous'
        return regime, kn, lambda_m

    def _calc_real_pressure_pa(self):
        """Ideal-gas real pressure from molecule count in the defined volume."""
        n_real = len(self.mol_pos) * self.molecules_per_particle
        V = self._get_defined_volume_m3()
        t_gas = self._get_gas_ambient_temperature_k()
        return (n_real * kB * t_gas) / V

    def _calc_theoretical_mixture_heat_flow(self, p_real_pa):
        """Return electro-thermal gas heat-loss term for the current gas mixture."""
        fracs = self._get_mixture_fractions()
        state = self._solve_electro_thermal_state(p_real_pa, fracs)
        return state['q_gas_w']

    def _get_nominal_aN2(self, cfg):
        """Nominal N2 accommodation baseline by sensor surface material."""
        return 0.70 if cfg.get('surface') == 'Si' else 0.60

    def _get_sensor_hot_area_m2(self, cfg):
        """Radiating/heated area used in thermal-loss terms."""
        if cfg['geometry'] == 'plates':
            return max(float(cfg['plate_area']), 1e-18)
        r1 = float(cfg['wire_r'])
        L = float(cfg['wire_L'])
        return max(2.0 * math.pi * r1 * L, 1e-18)

    def _calc_support_conductance_w_per_k(self, cfg):
        """Pressure-independent support conductance Gs (W/K)."""
        g_extra = max(float(self.sensor_extra_support_g_wpk), 0.0)
        if cfg['geometry'] == 'plates':
            n_support = 4.0
            lam = max(float(self.sensor_support_lambda_wmk), 1e-6)
            w = max(float(self.sensor_support_w_m), 1e-9)
            t = max(float(self.sensor_support_t_m), 1e-9)
            Ls = max(float(self.sensor_support_l_m), 1e-9)
            g_s = n_support * lam * (w * t) / Ls
            return g_s + g_extra

        # Wire gauge: conduction through two lead paths along the filament.
        k_wire = 140.0 if cfg.get('surface') == 'W' else 72.0
        r1 = max(float(cfg.get('wire_r', 5e-6)), 1e-9)
        A_wire = math.pi * (r1 ** 2)
        half_len = max(0.5 * float(cfg.get('wire_L', 0.05)), 1e-6)
        g_wire = 2.0 * k_wire * A_wire / half_len
        return g_wire + g_extra

    def _calc_radiation_loss_w(self, t_hot_k, t_cold_k, cfg):
        """Stefan-Boltzmann radiative heat loss from the heated element."""
        A = self._get_sensor_hot_area_m2(cfg)
        eps = float(np.clip(self.sensor_emissivity, 0.01, 0.95))
        t1 = max(float(t_hot_k), 1.0)
        t2 = max(float(t_cold_k), 1.0)
        return eps * SIGMA_SB * A * max(t1 ** 4 - t2 ** 4, 0.0)

    def _calc_convection_loss_w(self, p_pa, t_hot_k, t_cold_k, cfg, fracs):
        """Natural-convection loss, negligible below about 1e4 Pa."""
        p = max(float(p_pa), 0.0)
        dt = max(float(t_hot_k) - float(t_cold_k), 0.0)
        if p < 1e4 or dt <= 0.0:
            return 0.0

        t_film = max(0.5 * (t_hot_k + t_cold_k), 180.0)
        k_mix = 0.0
        mu_mix = 0.0
        cp_mix = 0.0
        m_mix = 0.0
        for gk, x in fracs.items():
            mu_g, k_g, cp_g, _ = _get_gas_transport(gk, t_film)
            k_mix += x * k_g
            mu_mix += x * mu_g
            cp_mix += x * cp_g
            m_mix += x * (GAS_DATA[gk]['m'] / 1000.0)

        k_mix = max(k_mix, 1e-6)
        mu_mix = max(mu_mix, 1e-9)
        cp_mix = max(cp_mix, 1.0)
        m_mix = max(m_mix, 1e-6)
        rho = max((p * m_mix) / (R_UNIV * t_film), 1e-9)
        nu = mu_mix / rho
        alpha = k_mix / max(rho * cp_mix, 1e-12)
        pr = float(np.clip(nu / max(alpha, 1e-12), 0.2, 8.0))

        if cfg['geometry'] == 'plates':
            l_char = max(float(cfg.get('gap', 1e-3)), 1e-6)
        else:
            r2 = float(cfg.get('enc_r', 8e-3))
            r1 = float(cfg.get('wire_r', 5e-6))
            l_char = max(r2 - r1, 1e-6)

        beta = 1.0 / t_film
        ra = G_STD * beta * dt * (l_char ** 3) / max(nu * alpha, 1e-18)
        ra = float(np.clip(ra, 0.0, 1e12))
        nu_nat = 0.68 + (0.67 * (ra ** 0.25)) / ((1.0 + (0.492 / pr) ** (9.0 / 16.0)) ** (4.0 / 9.0))
        h = max((nu_nat * k_mix) / l_char, 0.0)
        A = self._get_sensor_hot_area_m2(cfg)
        return h * A * dt

    def _solve_electro_thermal_state(self, p_pa, fracs, use_n2_only=False):
        """Solve Qel(T)=Qgas(T,P)+Qsupport(T)+Qrad(T)+Qconv(T,P) for filament T."""
        cfg = dict(self._get_active_sim_config())
        cfg['conv_gain'] = 0.0  # convection is accounted separately in Q_conv
        t_cold = float(self.T_COLD)
        p = max(float(p_pa), 1e-20)
        aN2 = self._get_nominal_aN2(cfg)
        g_s = self._calc_support_conductance_w_per_k(cfg)

        mix = {'N2': 1.0} if use_n2_only else dict(fracs)
        if not mix:
            mix = {'N2': 1.0}

        r0 = max(float(self.sensor_r0_ohm), 1.0)
        alpha_tcr = float(self.sensor_tcr_per_k)
        t_ref = float(self.sensor_t_ref_k)
        v_sensor = float(self.bridge_v_sensor)

        def residual(t_hot):
            rs = max(r0 * (1.0 + alpha_tcr * (t_hot - t_ref)), 1e-6)
            q_el = (v_sensor * v_sensor) / rs
            q_gas = 0.0
            for gk, x in mix.items():
                q_g, _, _ = calc_heat_flow(
                    gk,
                    cfg,
                    p,
                    aN2=aN2,
                    t_hot_override=t_hot,
                    t_cold_override=t_cold,
                )
                q_gas += x * q_g
            q_support = g_s * max(t_hot - t_cold, 0.0)
            q_rad = self._calc_radiation_loss_w(t_hot, t_cold, cfg)
            q_conv = self._calc_convection_loss_w(p, t_hot, t_cold, cfg, mix)
            return q_el - (q_gas + q_support + q_rad + q_conv), q_gas, q_support, q_rad, q_conv, rs, q_el

        t_lo = t_cold + 0.05
        t_hi = max(t_cold + 450.0, 900.0)
        f_lo, *_ = residual(t_lo)
        f_hi, *_ = residual(t_hi)

        if f_lo * f_hi > 0.0:
            # Fallback if bracket is imperfect: choose lower-residual endpoint.
            cand = [t_lo, t_hi]
            vals = [abs(f_lo), abs(f_hi)]
            t_star = cand[int(vals[1] < vals[0])]
            f_star, q_gas, q_support, q_rad, q_conv, rs, q_el = residual(t_star)
            return {
                't_hot_k': float(t_star),
                'q_el_w': float(q_el),
                'q_gas_w': float(q_gas),
                'q_support_w': float(q_support),
                'q_radiation_w': float(q_rad),
                'q_convection_w': float(q_conv),
                'r_sensor_ohm': float(rs),
                'residual_w': float(f_star),
            }

        lo, hi = t_lo, t_hi
        state = None
        for _ in range(56):
            mid = 0.5 * (lo + hi)
            f_mid, q_gas, q_support, q_rad, q_conv, rs, q_el = residual(mid)
            state = (mid, f_mid, q_gas, q_support, q_rad, q_conv, rs, q_el)
            if f_mid == 0.0:
                break
            if f_lo * f_mid > 0.0:
                lo = mid
                f_lo = f_mid
            else:
                hi = mid

        t_hot, f_mid, q_gas, q_support, q_rad, q_conv, rs, q_el = state
        return {
            't_hot_k': float(t_hot),
            'q_el_w': float(q_el),
            'q_gas_w': float(q_gas),
            'q_support_w': float(q_support),
            'q_radiation_w': float(q_rad),
            'q_convection_w': float(q_conv),
            'r_sensor_ohm': float(rs),
            'residual_w': float(f_mid),
        }

    def _bridge_output_from_sensor_resistance(self, r_sensor_ohm):
        """Wheatstone bridge output voltage Vout from Rs and reference branch."""
        t_cold = float(self.T_COLD)
        r_ref = self.sensor_r0_ohm * (1.0 + self.sensor_tcr_per_k * (t_cold - self.sensor_t_ref_k))
        r_ref = max(r_ref, 1e-6)
        rs = max(float(r_sensor_ohm), 1e-6)
        vb = float(self.bridge_v_bias)
        r1 = max(float(self.bridge_r1_ohm), 1e-6)
        r2 = max(float(self.bridge_r2_ohm), 1e-6)
        return vb * ((rs / (rs + r_ref)) - (r1 / (r1 + r2)))

    def _bridge_output_for_pressure(self, p_pa, fracs, use_n2_only=False):
        """Bridge output voltage for a pressure and composition."""
        state = self._solve_electro_thermal_state(p_pa, fracs, use_n2_only=use_n2_only)
        v_out = self._bridge_output_from_sensor_resistance(state['r_sensor_ohm'])
        state['v_out_v'] = float(v_out)
        return state

    def _invert_n2_bridge_to_pressure(self, v_target):
        """Invert N2 bridge calibration curve to indicated pressure."""
        p_lo = 1e-10
        p_hi = 2e6
        v_lo = self._bridge_output_for_pressure(p_lo, {'N2': 1.0}, use_n2_only=True)['v_out_v']
        v_hi = self._bridge_output_for_pressure(p_hi, {'N2': 1.0}, use_n2_only=True)['v_out_v']

        if v_target <= min(v_lo, v_hi):
            return p_lo if v_lo <= v_hi else p_hi
        if v_target >= max(v_lo, v_hi):
            return p_hi if v_hi >= v_lo else p_lo

        lo, hi = p_lo, p_hi
        for _ in range(42):
            mid = 0.5 * (lo + hi)
            v_mid = self._bridge_output_for_pressure(mid, {'N2': 1.0}, use_n2_only=True)['v_out_v']
            if (v_mid < v_target and v_lo < v_hi) or (v_mid > v_target and v_lo > v_hi):
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _get_pirani_calibration_pa(self):
        """Return fixed model range and zero baseline in Pa.

        Calibration sliders were removed; bounds now track the active gauge
        saturation target so readout behavior is stable across presets.
        """
        cfg = self._get_active_sim_config()
        sat_pa = max(float(cfg.get('sat_target_mbar', 1000.0)) * 100.0, 1.0)
        p_min = 1e-8
        p_max = max(4.0 * sat_pa, 1e5)
        p_zero = p_min
        return p_min, p_max, p_zero

    def _get_empirical_cf_for_gas(self, gas_key):
        """Return experimental correction factor for one gas in active geometry."""
        d = EXPERIMENTAL_CF.get(gas_key)
        if d is None:
            return 1.0
        cf_key = 'vm3' if GAUGE_CONFIGS[self.cfg_var.get()]['geometry'] == 'plates' else 'mean'
        return float(d.get(cf_key, 1.0))

    def _get_empirical_cf_mix(self, fracs):
        """Return molecular-regime mixture CF with harmonic averaging.

        In molecular flow: p_ind / p_real = sum_i(x_i / CF_i), therefore
        CF_mix = 1 / sum_i(x_i / CF_i).
        """
        denom = 0.0
        for gk, x in fracs.items():
            cf_i = max(self._get_empirical_cf_for_gas(gk), 1e-12)
            denom += x / cf_i
        if denom <= 1e-18:
            return 1.0
        return 1.0 / denom

    # ── Accuracy curves per gauge geometry ────────────────────────────────

    def _pirani_accuracy_band(self, p_real_pa):
        """Return (low, high) fractional accuracy for the active gauge and pressure.

        Accuracy tiers are read directly from the gauge config so each real-world
        gauge model produces the correct spec-sheet accuracy bands.
        """
        p_mbar = convert_pressure(max(p_real_pa, 1e-20), 'mbar')
        cfg_key = self.cfg_var.get()
        cfg = GAUGE_CONFIGS.get(cfg_key, {})
        frac = gauge_accuracy_fraction(cfg, p_mbar)
        return frac, frac

    def _pirani_accuracy_fraction(self, p_real_pa):
        """Return conservative (upper-band) fractional accuracy for error envelope."""
        _, hi = self._pirani_accuracy_band(p_real_pa)
        return hi

    def _pirani_accuracy_label(self, p_real_pa):
        """Return display label using original gauge-spec accuracy band."""
        lo, hi = self._pirani_accuracy_band(p_real_pa)
        lo_pct = int(round(lo * 100.0))
        hi_pct = int(round(hi * 100.0))
        if lo_pct == hi_pct:
            return f"±{hi_pct}%"
        return f"±{lo_pct}-{hi_pct}%"

    def _invert_n2_heat_flow_to_pressure(self, q_target):
        """Return N2-equivalent pressure for a measured Pirani heat-flow signal."""
        cfg = self._get_active_sim_config()
        p_lo = 1e-10
        p_hi = 2e6
        q_lo, _, _ = calc_heat_flow('N2', cfg, p_lo, aN2=0.6)
        q_hi, _, _ = calc_heat_flow('N2', cfg, p_hi, aN2=0.6)

        if q_target <= q_lo:
            return p_lo
        if q_target >= q_hi:
            return p_hi

        lo, hi = p_lo, p_hi
        for _ in range(36):
            mid = 0.5 * (lo + hi)
            q_mid, _, _ = calc_heat_flow('N2', cfg, mid, aN2=0.6)
            if q_mid < q_target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _calc_collision_perturbation_factor(self, p_real_pa):
        """Return a small multiplicative perturbation from the collision simulation.

        The analytical heat-flow model (Jousten 2008) is the authoritative
        physics for the Pirani reading.  The molecular collision simulation
        provides stochastic "sensor noise" — a small bounded modulation of
        the theoretical signal, giving the readout some realistic jitter.

        Returns a factor in [0.92, 1.08] centered on 1.0.
        """
        signal_collision = max(self.collision_signal_ema * self.molecules_per_particle, 0.0)
        if signal_collision <= 1e-18 or self.sensor_samples < 4:
            return 1.0

        q_theory_mix = self._calc_theoretical_mixture_heat_flow(p_real_pa)
        if q_theory_mix <= 1e-18:
            return 1.0

        # Track a smoothed gain so we know the expected collision signal level
        g_new = q_theory_mix / signal_collision
        if self.sensor_gain_q is None:
            self.sensor_gain_q = g_new
        else:
            alpha_g = 0.02   # very slow tracking — we only need the baseline
            self.sensor_gain_q = (1.0 - alpha_g) * self.sensor_gain_q + alpha_g * g_new

        # How far is the current collision signal from the expected level?
        q_coll = signal_collision * self.sensor_gain_q
        ratio = q_coll / q_theory_mix          # ≈1.0 at steady state

        # Clamp to ±8 % perturbation — enough for visual realism, never wild
        return float(np.clip(ratio, 0.92, 1.08))

    def on_app_close(self):
        """Cancel pending animation callbacks before app shutdown."""
        self.running = False
        if self.anim_id:
            try:
                self.after_cancel(self.anim_id)
            except Exception:
                pass
            self.anim_id = None

    def _get_readout_profile(self, cfg):
        """Return merged readout profile controlling range-edge nonideal behavior."""
        defaults = {
            'low_floor_mult': 0.35,
            'low_knee_mult': 12.0,
            'high_knee_frac_sat': 0.25,
            'high_ref_frac_sat': 2.0,
            'high_saturation_bend': 0.35,
            'low_edge_max_err': 0.55,
            'high_edge_max_err': 0.50,
        }
        profile = dict(defaults)
        profile.update(cfg.get('readout_profile', {}))
        return profile

    def _calc_range_edge_severity(self, p_real_pa, cfg):
        """Return (low_edge, high_edge) severities for range-end nonideal behavior."""
        p = max(float(p_real_pa), 1e-20)
        profile = self._get_readout_profile(cfg)
        range_lo_mbar, _ = cfg.get('range_mbar', (1e-4, 1000.0))
        range_lo_pa = max(float(range_lo_mbar) * 100.0, 1e-12)
        sat_pa = max(float(cfg.get('sat_target_mbar', 1000.0)) * 100.0, 1.0)

        # Low-pressure edge: electronics/noise floor dominates near lower limit.
        low_floor = max(float(profile['low_floor_mult']) * range_lo_pa, 1e-12)
        low_knee = max(float(profile['low_knee_mult']) * range_lo_pa, low_floor * 2.0)
        if p >= low_knee:
            low_edge = 0.0
        else:
            denom = max(math.log10(low_knee / low_floor), 1e-6)
            low_edge = np.clip(math.log10(low_knee / max(p, low_floor)) / denom, 0.0, 1.0)

        # High-pressure edge: Pirani heat transfer saturates in viscous regime.
        high_knee = max(float(profile['high_knee_frac_sat']) * sat_pa, range_lo_pa * 100.0)
        high_ref = max(float(profile['high_ref_frac_sat']) * sat_pa, high_knee * 1.2)
        if p <= high_knee:
            high_edge = 0.0
        else:
            denom = max(math.log10(high_ref / high_knee), 1e-6)
            high_edge = np.clip(math.log10(p / high_knee) / denom, 0.0, 1.0)

        return float(low_edge), float(high_edge)

    def _calc_pirani_indicated_pressure_pa(self, p_real_pa):
        """Pirani indicated pressure from N2-calibrated Wheatstone bridge output."""
        fracs = self._get_mixture_fractions()
        mix_state = self._bridge_output_for_pressure(p_real_pa, fracs, use_n2_only=False)
        v_mix = mix_state['v_out_v']
        p_inv = self._invert_n2_bridge_to_pressure(v_mix)

        # Optional small electrical readout noise floor (<0.1 mV typical).
        if self.sensor_enable_bridge_noise:
            noise_v = np.random.normal(0.0, 0.00005)
            p_inv = self._invert_n2_bridge_to_pressure(v_mix + noise_v)

        cfg = self._get_active_sim_config()
        profile = self._get_readout_profile(cfg)
        p_min, p_max, _ = self._get_pirani_calibration_pa()
        sat_pa = max(float(cfg.get('sat_target_mbar', 1000.0)) * 100.0, 1.0)
        range_lo_pa = max(float(cfg.get('range_mbar', (1e-4, 1000.0))[0]) * 100.0, p_min)
        p_real = max(float(p_real_pa), 1e-20)

        low_edge, high_edge = self._calc_range_edge_severity(p_real, cfg)

        # At the low end, bridge/noise floor causes indications to collapse toward a floor.
        if low_edge > 1e-6:
            floor_pa = max(float(profile['low_floor_mult']) * range_lo_pa, p_min)
            phase = 0.13 * self.frame_count + 0.7
            floor_pa *= (1.0 + 0.08 * (0.7 * np.sin(phase) + 0.3 * np.sin(0.31 * phase + 1.2)))
            floor_pa = max(floor_pa, p_min)
            log_meas = math.log10(max(p_inv, p_min))
            log_floor = math.log10(floor_pa)
            p_inv = 10 ** ((1.0 - low_edge) * log_meas + low_edge * log_floor)

        # At the high end, viscous heat transfer saturation flattens indicated pressure.
        if high_edge > 1e-6:
            bend = float(profile['high_saturation_bend'])
            p_sat = sat_pa * (1.0 + bend * math.log10(1.0 + p_real / sat_pa))
            p_sat = float(np.clip(p_sat, p_min, p_max))
            log_meas = math.log10(max(p_inv, p_min))
            log_sat = math.log10(max(p_sat, p_min))
            p_inv = 10 ** ((1.0 - high_edge) * log_meas + high_edge * log_sat)

        p_inv = self._apply_pirani_error_model(p_inv, p_real, fracs)
        return float(np.clip(max(p_inv, p_min), p_min, p_max))

    def _apply_pirani_error_model(self, p_indicated, p_real_pa, fracs):
        """Apply calibration range clamping, accuracy error, and ambient drift."""
        p_min, p_max, _ = self._get_pirani_calibration_pa()
        cfg = self._get_active_sim_config()
        profile = self._get_readout_profile(cfg)

        # Clamp to calibration range (below-range reads at floor)
        p_cal = max(p_indicated, p_min)
        p_cap = 2e5                    # hard cap at 2000 mbar (2e5 Pa)

        # ── Persistent systematic bias (seeded once per simulation session) ──
        if not hasattr(self, '_session_bias'):
            # Fixed random bias representing gauge calibration uncertainty.
            # ~40% of the accuracy band, constant for the whole session.
            self._session_bias = float(np.random.default_rng().normal(0, 0.4))

        err_amp = self._pirani_accuracy_fraction(p_real_pa)

        # Force realistic edge-of-range divergence toward about ±50%.
        low_edge, high_edge = self._calc_range_edge_severity(p_real_pa, cfg)
        edge_amp = max(float(profile['low_edge_max_err']) * low_edge,
                   float(profile['high_edge_max_err']) * high_edge)
        err_amp = max(err_amp, edge_amp)

        # High ambient temperatures degrade Pirani reliability
        ambient_thresh_k = 323.15  # 50 °C
        if self.T_COLD > ambient_thresh_k:
            severity = np.clip((self.T_COLD - ambient_thresh_k) / 30.0, 0.0, 1.0)
            thermal_headroom = max(self.T_HOT - self.T_COLD, 1.0)
            headroom_loss = np.clip((70.0 - thermal_headroom) / 70.0, 0.0, 1.0)
            err_amp *= (1.0 + 2.2 * severity + 0.8 * headroom_loss)

        # Persistent component (doesn't average away)
        bias_err = err_amp * self._session_bias
        # Small oscillating component (measurement noise, does average away)
        det_phase = 0.17 * self.frame_count + 1.3 * len(fracs)
        osc_err = 0.3 * err_amp * (
            0.65 * np.sin(det_phase) + 0.35 * np.sin(0.37 * det_phase + 1.1))
        # Clamp total error so it never exceeds the accuracy spec envelope
        total_err = np.clip(bias_err + osc_err, -err_amp, err_amp)
        p_cal *= (1.0 + total_err)

        # High-ambient drift
        if self.T_COLD > ambient_thresh_k:
            severity = np.clip((self.T_COLD - ambient_thresh_k) / 30.0, 0.0, 1.0)
            thermal_headroom = max(self.T_HOT - self.T_COLD, 1.0)
            headroom_loss = np.clip((70.0 - thermal_headroom) / 70.0, 0.0, 1.0)
            drift = severity * (0.07 + 0.12 * headroom_loss)
            phase = 0.11 * self.frame_count + 0.9
            p_cal *= (1.0 + drift * (
                0.7 * np.sin(phase) + 0.3 * np.sin(0.41 * phase + 1.7)))

        return float(np.clip(p_cal, p_min, p_cap))

    def _update_live_pressure(self):
        """Refresh the prominent real-time pressure readout."""
        p_set = 10 ** self.sl_pressure.get()
        p_real = self._calc_real_pressure_pa()
        p_ind = self._calc_pirani_indicated_pressure_pa(p_real)
        self.pirani_readings_pa.append(float(p_ind))
        p_ind_avg = self._get_avg_pirani_reading_pa()
        vol_m3 = self._get_defined_volume_m3()
        p_min, p_max, p_zero = self._get_pirani_calibration_pa()
        unit = get_pressure_unit()
        self.live_pressure_label.config(
            text=(
                f"Pirani (N₂-cal): {format_pressure(p_ind, unit, fmt='{:.4g}') }"
                f"   •   Avg ({len(self.pirani_readings_pa)}/{self.pirani_readings_pa.maxlen}): "
                f"{format_pressure(p_ind_avg, unit, fmt='{:.4g}') if p_ind_avg is not None else '—'}"
            ))
        self.live_pressure_detail.config(
            text=(
                f"Real (N,V,T): {format_pressure(p_real, unit, fmt='{:.4g}')}"
                f"   •   Setpoint: {format_pressure(p_set, unit, fmt='{:.4g}') }"
                f"   •   Model range: {format_pressure(p_min, unit, fmt='{:.2g}')}..{format_pressure(p_max, unit, fmt='{:.2g}') }"
                f"   •   Baseline: {format_pressure(p_zero, unit, fmt='{:.2g}') }"
                f"   •   V={vol_m3 * 1e6:.3g} cm³"
                f"{'   •   ⚠ Ambient > 50°C: sensor reading degraded' if self.T_COLD > 323.15 else ''}"
            )
        )
        return p_set, p_real, p_ind, vol_m3

    def _get_avg_pirani_reading_pa(self):
        """Return robust rolling-average Pirani reading (spike resistant)."""
        if not self.pirani_readings_pa:
            return None
        vals = np.asarray(self.pirani_readings_pa, dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        vals = vals[vals > 0]
        if vals.size == 0:
            return None

        logv = np.log(vals)
        med = np.median(logv)
        mad = np.median(np.abs(logv - med))
        sigma = max(1.4826 * mad, 0.08)
        keep = np.abs(logv - med) <= 2.5 * sigma
        if np.any(keep):
            log_mean = np.mean(logv[keep])
        else:
            log_mean = med
        return float(np.exp(log_mean))

    def _update_sim_clock(self):
        """Update live simulation time display."""
        t_s = self.frame_count * self.DT
        self.sim_time_label.config(text=f'Simulation Time: {t_s:.1f} s')


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

class PiraniSimulatorApp:
    VARIABLE_KEY_TEXT = (
        "Pirani Simulator - Variable Key\n"
        "\n"
        "Core pressures and temperatures\n"
        "p            = pressure (Pa unless noted)\n"
        "p_real       = real pressure from ideal gas law N*kB*T/V\n"
        "p_ind        = indicated pressure (N2-calibrated readout)\n"
        "T1 / T_hot   = hot filament/sensor temperature (K)\n"
        "T2 / T_cold  = enclosure (wall) temperature (K)\n"
        "Delta T      = temperature difference (T1 - T2)\n"
        "\n"
        "Gas and kinetic parameters\n"
        "alpha_E      = thermal accommodation coefficient (0..1)\n"
        "f            = molecular degrees of freedom\n"
        "gamma        = heat-capacity ratio Cp/Cv\n"
        "c_bar        = mean molecular speed (m/s)\n"
        "lambda       = mean free path (m)\n"
        "Kn           = Knudsen number = lambda / characteristic_length\n"
        "m            = molecular mass (amu in tables, kg in equations)\n"
        "\n"
        "Geometry terms\n"
        "r1           = wire radius (m)\n"
        "r2           = enclosure radius (m)\n"
        "L            = wire length (m)\n"
        "A            = heated surface area (m^2)\n"
        "x / gap      = plate spacing (m)\n"
        "\n"
        "Heat-flow terms\n"
        "Q_mol        = molecular-regime heat transfer (W)\n"
        "Q_visc       = viscous/continuum-regime heat transfer (W)\n"
        "Q_gas        = gas conduction term seen by sensor (W)\n"
        "Q_support    = support/lead conduction loss (W)\n"
        "Q_rad        = radiative loss (W)\n"
        "Q_conv       = natural-convection loss (W)\n"
        "Q_el         = electrical heating power in sensor element (W)\n"
        "\n"
        "Bridge/electronics terms\n"
        "R_s          = sensor resistance (ohm)\n"
        "R0           = sensor resistance at reference temperature (ohm)\n"
        "TCR          = temperature coefficient of resistance (1/K)\n"
        "V_bias       = bridge supply voltage (V)\n"
        "V_sensor     = effective sensor-bias voltage used in model (V)\n"
        "V_out        = Wheatstone bridge output voltage (V)\n"
        "\n"
        "Flow-regime guidance\n"
        "Kn > 1       = molecular flow\n"
        "0.01 < Kn < 1= transition flow\n"
        "Kn < 0.01    = viscous flow\n"
        "\n"
        "Display notes\n"
        "N2-cal       = readout calibrated to nitrogen (N2)\n"
        "CF           = correction factor for non-N2 gases\n"
        "All temperature sliders are in Kelvin internally; display unit can be changed globally.\n"
    )

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Pirani Vacuum Gauge Simulator — Jousten (2008)")
        sw = max(self.root.winfo_screenwidth(), 1024)
        sh = max(self.root.winfo_screenheight(), 700)
        win_w = max(1100, min(1680, int(sw * 0.9)))
        win_h = max(700, min(1020, int(sh * 0.9)))
        pos_x = max((sw - win_w) // 2, 0)
        pos_y = max((sh - win_h) // 2, 0)
        self.root.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")
        self.root.minsize(max(980, int(win_w * 0.75)), max(620, int(win_h * 0.72)))
        self.root.protocol('WM_DELETE_WINDOW', self._on_close)

        # ── Global shared state ──
        APP_STATE['theme_mode'] = tk.StringVar(value='light')
        APP_STATE['pressure_unit'] = tk.StringVar(value='mbar')
        APP_STATE['temperature_unit'] = tk.StringVar(value='C')

        apply_app_theme(self.root, get_theme_mode())

        # ── Header ──
        header = ttk.Frame(self.root)
        header.pack(fill='x', padx=16, pady=(8, 0))

        # Global theme selector (far-left)
        theme_frame = ttk.Frame(header)
        theme_frame.pack(side='left', pady=4)
        ttk.Label(theme_frame, text='Theme:', style='Dim.TLabel').pack(side='left', padx=(0, 6))
        theme_combo = ttk.Combobox(theme_frame, textvariable=APP_STATE['theme_mode'],
                   values=list(THEME_MODES),
                   state='readonly', width=7)
        theme_combo.pack(side='left')

        ttk.Label(header, text="Pirani Vacuum Gauge Simulator",
              style='Header.TLabel').pack(side='left', padx=(12, 0))

        # Global pressure unit selector (right side of header)
        unit_frame = ttk.Frame(header)
        unit_frame.pack(side='right', pady=4)
        ttk.Label(unit_frame, text='Pressure Unit:', style='Dim.TLabel').pack(side='left', padx=(0, 6))
        unit_combo = ttk.Combobox(unit_frame, textvariable=APP_STATE['pressure_unit'],
                                  values=list(PRESSURE_UNITS.keys()),
                                  state='readonly', width=7)
        unit_combo.pack(side='left')

        ttk.Label(unit_frame, text='Temperature Unit:', style='Dim.TLabel').pack(side='left', padx=(10, 6))
        temp_combo = ttk.Combobox(unit_frame, textvariable=APP_STATE['temperature_unit'],
                      values=list(TEMPERATURE_UNITS.keys()),
                      state='readonly', width=5)
        temp_combo.pack(side='left')

        ttk.Button(unit_frame, text='Variable Key', command=self._show_variable_key).pack(side='left', padx=(10, 0))

        ttk.Label(header, text="Jousten (2008)  •  J. Vac. Sci. Technol. A 26, 352–359",
                  style='Dim.TLabel').pack(side='right', padx=(0, 12), pady=6)

        # Notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=12, pady=(8, 12))

        # Create tabs
        tabs = [
            ("💡 How It Works", LearnTab),
            ("🧪 Gas Explorer", GasExplorerTab),
            ("📐 Gauge Geometry", GeometryViewerTab),
            ("🧬 Molecular Sim", MolecularSimTab),
            ("📈 2D Simulator", Simulator2DTab),
            ("🌐 3D Simulator", Simulator3DTab),
            ("🔧 Correction Factors", CorrectionFactorsTab),
            ("⚛ Accommodation", AccommodationTab),
            ("🧮 Calculator", CalculatorTab),
        ]

        self.tabs = []
        for label, TabClass in tabs:
            tab = TabClass(self.notebook)
            self.notebook.add(tab, text=f" {label} ")
            self.tabs.append(tab)

        APP_STATE['pressure_unit'].trace_add('write', self._on_global_units_changed)
        APP_STATE['temperature_unit'].trace_add('write', self._on_global_units_changed)
        APP_STATE['theme_mode'].trace_add('write', self._on_theme_changed)

        # Auto-pause molecular sim when user switches away from its tab
        self.notebook.bind('<<NotebookTabChanged>>', self._on_tab_changed)
        self._mol_sim_tab_id = None
        for i, tab in enumerate(self.tabs):
            if isinstance(tab, MolecularSimTab):
                self._mol_sim_tab_id = i
                break

    def _show_variable_key(self):
        """Show an always-available legend for symbols/variables used in the app."""
        dlg = tk.Toplevel(self.root)
        dlg.title('Variable Key')
        dlg.geometry('760x640')
        dlg.minsize(620, 460)
        dlg.transient(self.root)

        frame = ttk.Frame(dlg)
        frame.pack(fill='both', expand=True, padx=10, pady=10)

        text = tk.Text(frame, wrap='word', font=('Consolas', 10), padx=8, pady=8)
        ysb = ttk.Scrollbar(frame, orient='vertical', command=text.yview)
        text.configure(yscrollcommand=ysb.set)
        text.pack(side='left', fill='both', expand=True)
        ysb.pack(side='right', fill='y')

        text.insert('1.0', self.VARIABLE_KEY_TEXT)
        text.configure(state='disabled')

        btn_row = ttk.Frame(dlg)
        btn_row.pack(fill='x', padx=10, pady=(0, 10))
        ttk.Button(btn_row, text='Close', command=dlg.destroy).pack(side='right')

    def _on_tab_changed(self, event=None):
        """Pause molecular sim when leaving its tab; resume when returning."""
        if self._mol_sim_tab_id is None:
            return
        mol_tab = self.tabs[self._mol_sim_tab_id]
        current = self.notebook.index(self.notebook.select())
        if current == self._mol_sim_tab_id:
            mol_tab.on_tab_shown()
        else:
            mol_tab.on_tab_hidden()

    def _on_global_units_changed(self, *_):
        for tab in self.tabs:
            handler = getattr(tab, 'on_global_units_changed', None)
            if callable(handler):
                handler()

    def _on_theme_changed(self, *_):
        apply_app_theme(self.root, get_theme_mode())

        for tab in self.tabs:
            # Update ALL ScrollableControlPanel instances
            for attr in ('_scroll_panel', '_info_panel'):
                panel = getattr(tab, attr, None)
                if panel is not None and hasattr(panel, 'on_theme_changed'):
                    panel.on_theme_changed()

            # Call dedicated theme handler if present
            theme_fn = getattr(tab, 'on_theme_changed', None)
            if callable(theme_fn):
                theme_fn()

            # Always refresh the plot / scene so matplotlib picks up new colors
            for hook in ('_update_plot', '_draw_scene'):
                fn = getattr(tab, hook, None)
                if callable(fn):
                    fn()
                    break
            else:
                # Fallback: if tab has a canvas but no plot/scene method, redraw it
                canvas = getattr(tab, 'canvas', None)
                if canvas is not None and hasattr(canvas, 'draw_idle'):
                    canvas.draw_idle()

    def _on_close(self):
        """Shutdown cleanly by cancelling callbacks and destroying the Tk root."""
        for tab in getattr(self, 'tabs', []):
            close_fn = getattr(tab, 'on_app_close', None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass

            for attr in ('_update_job', 'anim_id'):
                aid = getattr(tab, attr, None)
                if aid:
                    try:
                        tab.after_cancel(aid)
                    except Exception:
                        pass
                    try:
                        setattr(tab, attr, None)
                    except Exception:
                        pass

        try:
            plt.close('all')
        except Exception:
            pass

        try:
            self.root.quit()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass

    def run(self):
        self.root.mainloop()


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Starting Pirani Vacuum Gauge Simulator...")
    print("Based on: Jousten (2008) J. Vac. Sci. Technol. A 26, 352-359")
    print()
    app = PiraniSimulatorApp()
    app.run()
