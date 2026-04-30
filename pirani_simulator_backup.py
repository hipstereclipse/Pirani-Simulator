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
from matplotlib import cm
import matplotlib.colors as mcolors
import sys
import math

# ══════════════════════════════════════════════════════════════════════════════
#  PHYSICS DATA FROM THE PAPER
# ══════════════════════════════════════════════════════════════════════════════

kB = 1.380649e-23  # Boltzmann constant (J/K)

GAS_DATA = {
    'H2':  {'name': 'Hydrogen',        'symbol': 'H₂',  'm': 2.016,  'f': 5, 'gamma': 1.41, 'cbar': 1764, 'plbar': 12.2e-3, 'color': '#ff6b6b'},
    'He':  {'name': 'Helium',          'symbol': 'He',   'm': 4.003,  'f': 3, 'gamma': 1.67, 'cbar': 1252, 'plbar': 19.0e-3, 'color': '#ffd93d'},
    'Ne':  {'name': 'Neon',            'symbol': 'Ne',   'm': 20.18,  'f': 3, 'gamma': 1.67, 'cbar': 557,  'plbar': 13.6e-3, 'color': '#6bcb77'},
    'CO':  {'name': 'Carbon Monoxide', 'symbol': 'CO',   'm': 28.011, 'f': 5, 'gamma': 1.40, 'cbar': 473,  'plbar': 6.4e-3,  'color': '#4d96ff'},
    'N2':  {'name': 'Nitrogen',        'symbol': 'N₂',   'm': 28.013, 'f': 5, 'gamma': 1.40, 'cbar': 473,  'plbar': 6.4e-3,  'color': '#a0a0a0'},
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
    'Ar':  {'range': '0.1-90',   'mean': 1.62, 'spread': 0.12, 'cfMin': 1.59, 'cfMax': 1.79, 'vm3': 1.51},
    'CO2': {'range': '0.1-30',   'mean': 0.95, 'spread': 0.03, 'cfMin': 0.95, 'cfMax': 0.97, 'vm3': 0.92},
    'Kr':  {'range': '0.5-90',   'mean': 2.22, 'spread': 0.16, 'cfMin': 2.20, 'cfMax': 2.41, 'vm3': 2.03},
    'Xe':  {'range': '0.5-13',   'mean': 2.71, 'spread': 0.20, 'cfMin': 2.70, 'cfMax': 2.95, 'vm3': 2.48},
}

# Accommodation coefficient ratios (Tables VII & VIII)
ACCOM_RATIOS_W = {'H2': 0.46, 'He': 0.57, 'Ne': 0.93, 'CO': 1.02, 'N2': 1.00, 'Ar': 1.08, 'CO2': 1.12, 'Kr': 1.14, 'Xe': 1.16}
ACCOM_RATIOS_Si = {'H2': 0.37, 'He': 0.48, 'Ne': 0.89, 'CO': 1.03, 'N2': 1.00, 'Ar': 1.19, 'CO2': 1.17, 'Kr': 1.28, 'Xe': 1.31}

# ── Shared application state (populated after Tk root is created) ────────────
APP_STATE = {}

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
    return 'Pa'


# Gauge configurations
GAUGE_CONFIGS = {
    'wire_cylinder': {
        'name': 'Wire-in-Cylinder (Conventional)',
        'desc': 'Heated tungsten wire (oxidized) in cylindrical enclosure.\nUsed in VM1, VM2, VM4 from the paper.',
        'T1': 393, 'T2': 296, 'wire_r': 5e-6, 'wire_L': 0.05, 'enc_r': 0.008,
        'surface': 'W', 'geometry': 'cylindrical',
    },
    'mems_plate': {
        'name': 'MEMS Parallel Plate (VM3)',
        'desc': 'Microstructured silicon sheet opposite a plate.\nSmall dimensions increase Knudsen number, extending\nmolecular regime. Low temp reduces radiation.',
        'T1': 333, 'T2': 296, 'plate_area': 1e-6, 'gap': 2e-6,
        'surface': 'Si', 'geometry': 'plates',
    },
    'custom_wire': {
        'name': 'Custom Wire Gauge',
        'desc': 'User-configurable wire-in-cylinder gauge.\nAdjust all parameters freely.',
        'T1': 393, 'T2': 296, 'wire_r': 5e-6, 'wire_L': 0.05, 'enc_r': 0.008,
        'surface': 'W', 'geometry': 'cylindrical',
    },
    'custom_plate': {
        'name': 'Custom Parallel Plate',
        'desc': 'User-configurable parallel plate gauge.\nAdjust all parameters freely.',
        'T1': 353, 'T2': 296, 'plate_area': 1e-4, 'gap': 1e-3,
        'surface': 'Si', 'geometry': 'plates',
    },
}

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
    m = m_amu * 1.6605e-27
    coeff = (9 * gamma - 5) / 4.0
    return coeff * (2 * np.pi * plbar / cbar) * (f * kB / (2 * m)) * L * (T1 - T2) / np.log(r2 / r1)

def calc_Q_visc_plates(gamma, plbar, cbar, f, m_amu, A, T1, T2, x):
    """Viscous regime heat flow for parallel plates (Eq. 6) — pressure-independent."""
    m = m_amu * 1.6605e-27
    coeff = (9 * gamma - 5) / 4.0
    return coeff * (2 * np.pi * plbar / cbar) * (f * kB / (2 * m)) * A * (T1 - T2) / x

def calc_Q_combined(Q_mol, Q_visc):
    """Combined heat flow using series-resistance analogy (Eq. 8)."""
    if Q_mol == 0 or Q_visc == 0:
        return 0.0
    return 1.0 / (1.0 / Q_mol + 1.0 / Q_visc)

def calc_heat_flow(gas_key, config, p, aN2=0.6):
    """Calculate heat flow for a gas at pressure p using a gauge configuration."""
    gas = GAS_DATA[gas_key]
    accom_table = ACCOM_RATIOS_W if config['surface'] == 'W' else ACCOM_RATIOS_Si
    aE_ratio = accom_table.get(gas_key, 1.0)
    aE = min(aN2 * aE_ratio, 1.0)

    T1, T2 = config['T1'], config['T2']

    if config['geometry'] == 'cylindrical':
        r1 = config['wire_r']
        L = config['wire_L']
        r2 = config['enc_r']
        Q_mol = calc_Q_mol_cylinder(aE, gas['f'], gas['cbar'], r1, L, T1, T2, p)
        Q_visc = calc_Q_visc_cylinder(gas['gamma'], gas['plbar'], gas['cbar'], gas['f'], gas['m'], L, T1, T2, r1, r2)
    else:  # plates
        A = config['plate_area']
        x = config['gap']
        aE2 = aE  # assume same on both surfaces for simplicity
        Q_mol = calc_Q_mol_plates(aE, aE2, gas['f'], gas['cbar'], A, T1, T2, p)
        Q_visc = calc_Q_visc_plates(gas['gamma'], gas['plbar'], gas['cbar'], gas['f'], gas['m'], A, T1, T2, x)

    return calc_Q_combined(Q_mol, Q_visc), Q_mol, Q_visc

def calc_correction_factor_theory(gas_key, aN2=0.6, surface='W'):
    """Theoretical correction factor from molecular regime (Eq. 11)."""
    gas = GAS_DATA[gas_key]
    ref = GAS_DATA['N2']
    accom = ACCOM_RATIOS_W if surface == 'W' else ACCOM_RATIOS_Si
    aE_ratio = accom[gas_key] / accom['N2']
    ratio = aE_ratio * ((gas['f'] + 1) / (ref['f'] + 1)) * (gas['cbar'] / ref['cbar'])
    return 1.0 / ratio if ratio != 0 else float('inf')


# ══════════════════════════════════════════════════════════════════════════════
#  DARK THEME STYLING
# ══════════════════════════════════════════════════════════════════════════════

FONT_FAMILY = 'Segoe UI'
FONT_MONO = 'Consolas'

COLORS = {
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
}

MPL_STYLE = {
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


def apply_dark_theme(root):
    """Apply refined dark theme inspired by macOS / Apple HIG."""
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
                 fmt='{:.2f}', command=None, **kwargs):
        super().__init__(parent, style='Card.TFrame')
        self.fmt = fmt
        self.unit = unit
        self.command = command

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
            var = tk.StringVar(value='Pa')
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
        self._build_ui()
        self._update_plot()

    def _build_ui(self):
        # Left panel: scrollable controls
        self._scroll_panel = ScrollableControlPanel(self, width=270)
        self._scroll_panel.pack(side='left', fill='y', padx=(6, 0), pady=6)
        ctrl_frame = self._scroll_panel.inner

        # Configuration selector
        cfg_frame = ttk.LabelFrame(ctrl_frame, text=' Gauge Configuration ')
        cfg_frame.pack(fill='x', pady=(0, 8))

        self.config_var = tk.StringVar(value='wire_cylinder')
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
                                 variable=var, command=self._update_plot,
                                 style='TCheckbutton')
            cb.pack(anchor='w', padx=8, pady=1)
            self.gas_vars[key] = var

        # Parameter sliders
        param_frame = ttk.LabelFrame(ctrl_frame, text=' Parameters ')
        param_frame.pack(fill='x', pady=(0, 8))

        self.sl_T1 = LabeledSlider(param_frame, 'Wire Temp T₁', 313, 573, 393,
                                   unit='K', fmt='{:.0f}', command=lambda v: self._update_plot())
        self.sl_T1.pack(fill='x', padx=6, pady=2)

        self.sl_T2 = LabeledSlider(param_frame, 'Enclosure Temp T₂', 273, 353, 296,
                                   unit='K', fmt='{:.0f}', command=lambda v: self._update_plot())
        self.sl_T2.pack(fill='x', padx=6, pady=2)

        self.sl_aN2 = LabeledSlider(param_frame, 'a_N₂ (accommodation)', 0.1, 1.0, 0.6,
                                    fmt='{:.2f}', command=lambda v: self._update_plot())
        self.sl_aN2.pack(fill='x', padx=6, pady=2)

        self.sl_wire_r = LabeledSlider(param_frame, 'Wire radius', 1, 50, 5,
                                       unit='μm', fmt='{:.1f}', command=lambda v: self._update_plot())
        self.sl_wire_r.pack(fill='x', padx=6, pady=2)

        self.sl_wire_L = LabeledSlider(param_frame, 'Wire length', 0.5, 20, 5,
                                       unit='cm', fmt='{:.1f}', command=lambda v: self._update_plot())
        self.sl_wire_L.pack(fill='x', padx=6, pady=2)

        self.sl_enc_r = LabeledSlider(param_frame, 'Enclosure radius', 2, 30, 8,
                                      unit='mm', fmt='{:.1f}', command=lambda v: self._update_plot())
        self.sl_enc_r.pack(fill='x', padx=6, pady=2)

        # Display options
        opt_frame = ttk.LabelFrame(ctrl_frame, text=' Display ')
        opt_frame.pack(fill='x', pady=(0, 8))

        self.log_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt_frame, text='Logarithmic X-axis', variable=self.log_var,
                        command=self._update_plot).pack(anchor='w', padx=8, pady=2)

        self.show_regimes_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_frame, text='Show mol/visc regimes', variable=self.show_regimes_var,
                        command=self._update_plot).pack(anchor='w', padx=8, pady=2)

        self.show_p0_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_frame, text='Show total Q_el (with p₀)', variable=self.show_p0_var,
                        command=self._update_plot).pack(anchor='w', padx=8, pady=2)

        # Right panel: plot
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
        if cfg['geometry'] == 'cylindrical':
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
        self.sl_T1.set(cfg['T1'])
        self.sl_T2.set(cfg['T2'])
        if cfg['geometry'] == 'cylindrical':
            self.sl_wire_r.set(cfg['wire_r'] * 1e6)
            self.sl_wire_L.set(cfg['wire_L'] * 1e2)
            self.sl_enc_r.set(cfg['enc_r'] * 1e3)
        else:
            self.sl_wire_r.set(25)
            self.sl_wire_L.set(1)
            self.sl_enc_r.set(cfg.get('gap', 1e-3) * 1e3)
        self._update_plot()

    def _update_plot(self, *args):
        with plt.rc_context(MPL_STYLE):
            self.ax.clear()
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
                Qs, Qms, Qvs = [], [], []
                for p in pressures:
                    Q, Qm, Qv = calc_heat_flow(gk, cfg, p, aN2)
                    Qs.append(Q * 1000)
                    Qms.append(Qm * 1000)
                    Qvs.append(Qv * 1000)

                self.ax.plot(p_display, Qs, color=gas['color'], linewidth=2, label=gas['symbol'])

                if show_regimes:
                    self.ax.plot(p_display, Qms, color=gas['color'], linewidth=1, linestyle='--', alpha=0.4)
                    self.ax.plot(p_display, Qvs, color=gas['color'], linewidth=1, linestyle=':', alpha=0.4)

            if show_p0:
                Q_n2 = []
                for p in pressures:
                    Q, _, _ = calc_heat_flow('N2', cfg, p, aN2)
                    Q_n2.append(Q * 1000 + 0.01 * 1000)
                self.ax.plot(p_display, Q_n2, color='white', linewidth=1.5, linestyle='-.',
                             alpha=0.5, label='Q_el (N₂+p₀)')

            if self.log_var.get():
                self.ax.set_xscale('log')

            self.ax.set_xlabel(f'Pressure ({unit})', fontsize=11)
            self.ax.set_ylabel('Heat Flow (mW)', fontsize=11)
            self.ax.set_title(f'Gas Heat Transfer — {GAUGE_CONFIGS[self.config_var.get()]["name"]}',
                              fontsize=12, color=COLORS['text_bright'], pad=10)
            self.ax.legend(fontsize=9, loc='upper left')
            self.ax.grid(True, alpha=0.3)

            if show_regimes:
                self.ax.text(0.98, 0.02, '—— combined  - - molecular  ···· viscous',
                             transform=self.ax.transAxes, ha='right', fontsize=8,
                             color=COLORS['text_dim'])

            self.fig.tight_layout()
            self.canvas.draw()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB: 3D SURFACE SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════

class Simulator3DTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
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
                            value=val, command=self._update_plot,
                            style='TCheckbutton').pack(anchor='w', padx=8, pady=2)

        # Gas selector for 3D
        gas_frame = ttk.LabelFrame(ctrl_frame, text=' Primary Gas ')
        gas_frame.pack(fill='x', pady=(0, 8))

        self.gas_3d_var = tk.StringVar(value='Ar')
        for key, gas in GAS_DATA.items():
            ttk.Radiobutton(gas_frame, text=f"{gas['symbol']}", variable=self.gas_3d_var,
                            value=key, command=self._update_plot,
                            style='TCheckbutton').pack(anchor='w', padx=8, pady=1)

        # Config selector
        cfg_frame = ttk.LabelFrame(ctrl_frame, text=' Configuration ')
        cfg_frame.pack(fill='x', pady=(0, 8))

        self.cfg_3d_var = tk.StringVar(value='wire_cylinder')
        for key, cfg in GAUGE_CONFIGS.items():
            ttk.Radiobutton(cfg_frame, text=cfg['name'], variable=self.cfg_3d_var,
                            value=key, command=self._update_plot,
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
        cm_combo.bind('<<ComboboxSelected>>', lambda e: self._update_plot())

        self.wireframe_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(app_frame, text='Wireframe overlay', variable=self.wireframe_var,
                        command=self._update_plot).pack(anchor='w', padx=8, pady=2)

        ttk.Button(ctrl_frame, text='↻ Refresh Plot', command=self._update_plot).pack(fill='x', pady=4)

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

    def _update_plot(self, *args):
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
            self.canvas.draw()

    def _plot_pressure_accommodation(self, cfg, gas_key, cmap_name, wireframe):
        P = np.logspace(-1, 4, 50)
        A = np.linspace(0.1, 1.0, 50)
        PP, AA = np.meshgrid(np.log10(P), A)
        Z = np.zeros_like(PP)

        for i in range(len(A)):
            for j in range(len(P)):
                Q, _, _ = calc_heat_flow(gas_key, cfg, P[j], A[i])
                Z[i, j] = Q * 1000

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
        P = np.logspace(-1, 4, 50)
        T = np.linspace(313, 573, 50)
        PP, TT = np.meshgrid(np.log10(P), T)
        Z = np.zeros_like(PP)

        for i in range(len(T)):
            cfg_t = dict(cfg)
            cfg_t['T1'] = T[i]
            for j in range(len(P)):
                Q, _, _ = calc_heat_flow(gas_key, cfg_t, P[j], 0.6)
                Z[i, j] = Q * 1000

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
        P = np.logspace(-1, 4, 50)
        if cfg['geometry'] == 'cylindrical':
            G = np.linspace(2, 30, 50)  # enclosure radius in mm
        else:
            G = np.linspace(0.002, 5, 50)  # gap in mm
        PP, GG = np.meshgrid(np.log10(P), G)
        Z = np.zeros_like(PP)

        for i in range(len(G)):
            cfg_g = dict(cfg)
            if cfg['geometry'] == 'cylindrical':
                cfg_g['enc_r'] = G[i] * 1e-3
            else:
                cfg_g['gap'] = G[i] * 1e-3
            for j in range(len(P)):
                Q, _, _ = calc_heat_flow(gas_key, cfg_g, P[j], 0.6)
                Z[i, j] = Q * 1000

        surf = self.ax.plot_surface(PP, GG, Z, cmap=cmap_name, alpha=0.85)
        if wireframe:
            self.ax.plot_wireframe(PP, GG, Z, color=COLORS['text_dim'], linewidth=0.3, alpha=0.3)

        gap_label = 'Enclosure Radius (mm)' if cfg['geometry'] == 'cylindrical' else 'Plate Gap (mm)'
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

        for i, gk in enumerate(gases):
            Q_n2_arr = []
            Q_x_arr = []
            for j, p in enumerate(P):
                Qn2, _, _ = calc_heat_flow('N2', cfg, p, 0.6)
                Qx, _, _ = calc_heat_flow(gk, cfg, p, 0.6)
                if Qx > 0:
                    Z[i, j] = Qn2 / Qx
                else:
                    Z[i, j] = 0

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
        Qs, Qms, Qvs = [], [], []
        for p in P:
            Q, Qm, Qv = calc_heat_flow(gas_key, cfg, p, 0.6)
            Qs.append(Q * 1000)
            Qms.append(Qm * 1000)
            Qvs.append(Qv * 1000)

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

    def _update_plot(self, *args):
        with plt.rc_context(MPL_STYLE):
            self.ax.clear()
            mode = self.cf_mode.get()

            if mode == 'bar':
                self._plot_bar()
            elif mode == 'true_vs_ind':
                self._plot_true_vs_indicated()
            elif mode == 'theory_vs_exp':
                self._plot_theory_vs_exp()
            elif mode == 'spread':
                self._plot_spread()

            self.fig.tight_layout()
            self.canvas.draw()

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

        self.ax.scatter(exp, theory_mol, c=colors, s=120, zorder=5, edgecolors='white', linewidths=1.5)
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
            self.ax.plot(d['mean'], i, 'o', color='white', markersize=8, zorder=5)
            self.ax.plot(d['vm3'], i, 's', color=GAS_DATA[k]['color'], markersize=8,
                         markeredgecolor='white', markeredgewidth=1.5, zorder=5)

        self.ax.set_yticks(range(len(gases)))
        self.ax.set_yticklabels(labels)
        self.ax.set_xlabel('CF_X/N₂', fontsize=11)
        self.ax.set_title('Gauge-to-Gauge Spread of Correction Factors', fontsize=12,
                          color=COLORS['text_bright'])
        self.ax.axvline(x=1.0, color=COLORS['accent'], linestyle='--', alpha=0.5)
        self.ax.grid(True, axis='x', alpha=0.3)

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markersize=8, label='Mean (all gauges)'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['accent'], markersize=8,
                   markeredgecolor='white', label='VM3 (MEMS)'),
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
            self.canvas.draw()


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

    def _update_plot(self, *args):
        with plt.rc_context(MPL_STYLE):
            self.ax.clear()
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
            self.canvas.draw()


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
        ttk.Label(row2, text='Indicated Pressure (Pa):').pack(side='left')
        self.calc_pind = tk.StringVar(value='100')
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

        params = [
            ('Gas:', 'ht_gas', 'N2'),
            ('Pressure (Pa):', 'ht_p', '10'),
            ('a_E:', 'ht_aE', '0.6'),
            ('T₁ wire (K):', 'ht_T1', '393'),
            ('T₂ enclosure (K):', 'ht_T2', '296'),
        ]

        self.ht_vars = {}
        for label, var_name, default in params:
            row = ttk.Frame(ht_frame)
            row.pack(fill='x', padx=8, pady=2)
            ttk.Label(row, text=label).pack(side='left')
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
        ref_frame = ttk.LabelFrame(right, text=' Quick Reference — All Correction Factors ')
        ref_frame.pack(fill='both', expand=True, pady=(0, 8))

        for key, d in EXPERIMENTAL_CF.items():
            gas = GAS_DATA[key]
            card = ttk.Frame(ref_frame, style='Card.TFrame')
            card.pack(fill='x', padx=6, pady=3)

            ttk.Label(card, text=f"  {gas['symbol']}", style='Accent.TLabel',
                      width=6).pack(side='left', padx=(4, 8))
            ttk.Label(card, text=f"CF = {d['mean']:.2f}  (±{d['spread']:.2f})",
                      style='Card.TLabel').pack(side='left')

            direction = '↑ underreads' if d['mean'] > 1 else ('↓ overreads' if d['mean'] < 1 else '— reference')
            ttk.Label(card, text=f"  {direction}  |  Range: {d['range']} Pa",
                      style='Dim.TLabel').pack(side='right', padx=4)

    def _get_gas_key(self):
        val = self.calc_gas.get()
        for key in GAS_DATA:
            if key in val:
                return key
        return 'N2'

    def _calc_correction(self, *args):
        try:
            key = self._get_gas_key()
            p_ind = float(self.calc_pind.get())
            cf = EXPERIMENTAL_CF[key]['mean']
            spread = EXPERIMENTAL_CF[key]['spread']
            p_true = p_ind * cf
            p_min = p_ind * (cf - spread)
            p_max = p_ind * (cf + spread)

            self.result_label.config(text=f"  {p_true:.4g} Pa  ")
            self.result_detail.config(
                text=f"CF = {cf:.2f} ± {spread:.2f}\n"
                     f"Range: {p_min:.4g} – {p_max:.4g} Pa\n"
                     f"Formula: p_true = {p_ind} × {cf:.2f} = {p_true:.4g} Pa")
        except (ValueError, KeyError):
            self.result_label.config(text="  Enter valid values  ")
            self.result_detail.config(text="")

    def _calc_heat(self, *args):
        try:
            gas_key = self.ht_vars['ht_gas'].get()
            p = float(self.ht_vars['ht_p'].get())
            aE = float(self.ht_vars['ht_aE'].get())
            T1 = float(self.ht_vars['ht_T1'].get())
            T2 = float(self.ht_vars['ht_T2'].get())
            gas = GAS_DATA[gas_key]

            cfg = dict(GAUGE_CONFIGS['wire_cylinder'])
            cfg['T1'] = T1
            cfg['T2'] = T2

            Q_combined, Q_mol, Q_visc = calc_heat_flow(gas_key, cfg, p, aE)

            self.ht_result.config(
                text=f"  {gas['symbol']} at {p} Pa:\n"
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
        self._build_ui()

    def _build_ui(self):
        canvas = tk.Canvas(self, bg=COLORS['bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient='vertical', command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=scroll_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)

        # Bind mousewheel
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Content
        content_width = 900

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


# ══════════════════════════════════════════════════════════════════════════════
#  TAB: CROSS-SECTION / GEOMETRY VIEWER
# ══════════════════════════════════════════════════════════════════════════════

class GeometryViewerTab(ttk.Frame):
    """Interactive 2D cross-section view of gauge configurations."""
    def __init__(self, parent):
        super().__init__(parent)
        self._build_ui()
        self._update_plot()

    def _build_ui(self):
        self._scroll_panel = ScrollableControlPanel(self, width=270)
        self._scroll_panel.pack(side='left', fill='y', padx=(6, 0), pady=6)
        ctrl_frame = self._scroll_panel.inner

        cfg_frame = ttk.LabelFrame(ctrl_frame, text=' Configuration ')
        cfg_frame.pack(fill='x', pady=(0, 8))

        self.geo_config = tk.StringVar(value='wire_cylinder')
        for key in ['wire_cylinder', 'mems_plate']:
            ttk.Radiobutton(cfg_frame, text=GAUGE_CONFIGS[key]['name'],
                            variable=self.geo_config, value=key,
                            command=self._update_plot,
                            style='TCheckbutton').pack(anchor='w', padx=8, pady=2)

        # Description
        self.desc_label = ttk.Label(ctrl_frame, text='', style='Dim.TLabel', wraplength=250)
        self.desc_label.pack(padx=8, pady=8)

        # Visualization options
        vis_frame = ttk.LabelFrame(ctrl_frame, text=' Visualization ')
        vis_frame.pack(fill='x', pady=(0, 8))

        self.show_molecules = tk.BooleanVar(value=True)
        ttk.Checkbutton(vis_frame, text='Show gas molecules', variable=self.show_molecules,
                        command=self._update_plot).pack(anchor='w', padx=8, pady=2)

        self.show_heat_arrows = tk.BooleanVar(value=True)
        ttk.Checkbutton(vis_frame, text='Show heat flow arrows', variable=self.show_heat_arrows,
                        command=self._update_plot).pack(anchor='w', padx=8, pady=2)

        self.show_knudsen = tk.BooleanVar(value=True)
        ttk.Checkbutton(vis_frame, text='Show Knudsen number info', variable=self.show_knudsen,
                        command=self._update_plot).pack(anchor='w', padx=8, pady=2)

        self.sl_pressure_geo = LabeledSlider(ctrl_frame, 'Pressure', -2, 5, 1,
                                             unit='(log₁₀ Pa)', fmt='{:.1f}',
                                             command=lambda v: self._update_plot())
        self.sl_pressure_geo.pack(fill='x', padx=6, pady=8)

        plot_frame = ttk.Frame(self)
        plot_frame.pack(side='right', fill='both', expand=True, padx=8, pady=8)

        with plt.rc_context(MPL_STYLE):
            self.fig = Figure(figsize=(9, 7), dpi=100)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

    def _update_plot(self, *args):
        cfg_key = self.geo_config.get()
        cfg = GAUGE_CONFIGS[cfg_key]
        self.desc_label.config(text=cfg['desc'])

        pressure = 10 ** self.sl_pressure_geo.get()

        with plt.rc_context(MPL_STYLE):
            self.fig.clear()

            if cfg['geometry'] == 'cylindrical':
                self._draw_cylindrical(pressure)
            else:
                self._draw_plates(pressure)

            self.fig.tight_layout()
            self.canvas.draw()

    def _draw_cylindrical(self, pressure):
        ax = self.fig.add_subplot(121)
        ax3 = self.fig.add_subplot(122, projection='3d')

        # 2D cross-section
        enc_r = 8  # mm visual scale
        wire_r = 0.5

        theta = np.linspace(0, 2*np.pi, 100)
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

        if self.show_knudsen.get():
            lbar_n2 = GAS_DATA['N2']['plbar'] / pressure if pressure > 0 else 999
            d_char = 0.008  # 8mm characteristic dimension
            Kn = lbar_n2 / d_char
            regime = 'Molecular' if Kn > 1 else ('Transition' if Kn > 0.01 else 'Viscous')
            ax.text(0, -enc_r*1.15, f'Kn = {Kn:.2g}  ({regime} regime)\nλ = {lbar_n2*1000:.2g} mm   p = {pressure:.2g} Pa',
                    ha='center', fontsize=8, color=COLORS['text_dim'])

        # 3D cylindrical view
        z = np.linspace(0, 5, 30)
        theta_3d = np.linspace(0, 2*np.pi, 40)
        Z, Theta = np.meshgrid(z, theta_3d)

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

    def _draw_plates(self, pressure):
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
            gap_m = 2e-6  # MEMS gap
            lbar_n2 = GAS_DATA['N2']['plbar'] / pressure if pressure > 0 else 999
            Kn = lbar_n2 / gap_m
            regime = 'Molecular' if Kn > 1 else ('Transition' if Kn > 0.01 else 'Viscous')
            ax.text(0, -gap*1.3, f'Kn = {Kn:.2g}  ({regime})\nMEMS gap: {gap_m*1e6:.0f} μm   p = {pressure:.2g} Pa',
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
    'air':          {'name': 'Air (simplified)',   'desc': 'N₂ 99 %  +  Ar 1 %  (no O₂ in model)',
                     'mix': {'N2': 99, 'Ar': 1}},
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
    N_SEG = 30             # filament segments for temperature tracking
    N_THETA = 16           # angular resolution of wire cylinder surface

    # ── Physics (tuned for visual clarity) ───────────────────────────────────
    T_HOT = 393.0          # equilibrium wire temperature (K)
    T_COLD = 296.0         # enclosure / gas temperature (K)
    HEATING_TAU = 0.55     # Joule-heating recovery time constant (visual s)
    DIFFUSION_K = 4.0      # thermal diffusion along the wire
    COOL_BASE = 16.0       # base ΔT per collision (K), scaled by αE & f

    # ── Animation ────────────────────────────────────────────────────────────
    DT = 0.04              # simulation timestep (s)
    INTERVAL = 40           # ms between frames (~25 fps)

    # ── Molecule visuals ─────────────────────────────────────────────────────
    MOL_SPEED_BASE = 4.0   # display speed for N₂ (units / s)
    MOL_SIZE_BASE = 60     # scatter marker size for N₂

    # ── Molecule radii (pm) for realistic relative sizing ────────────────────
    MOL_RADII = {
        'H2': 289, 'He': 260, 'Ne': 275, 'CO': 376, 'N2': 364,
        'Ar': 340, 'CO2': 330, 'Kr': 360, 'Xe': 396,
    }

    def __init__(self, parent):
        super().__init__(parent)
        self.running = False
        self.anim_id = None
        self.collision_count = 0
        self.total_energy_transferred = 0.0
        self.frame_count = 0
        self.collision_per_gas = {}  # gas_key -> count

        self.seg_temps = np.full(self.N_SEG, self.T_HOT)
        self.mol_pos = np.zeros((0, 3))
        self.mol_vel = np.zeros((0, 3))
        self.mol_gas_keys = []  # per-molecule gas species key

        self._elev = 20
        self._azim = -60

        self._build_ui()
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
        self.cfg_var = tk.StringVar(value='wire_cylinder')
        for key in ['wire_cylinder', 'mems_plate']:
            label = 'Wire-in-Cylinder' if key == 'wire_cylinder' else 'MEMS Parallel Plate'
            ttk.Radiobutton(cfg_frame, text=label,
                            variable=self.cfg_var, value=key,
                            command=self._on_config_change,
                            style='TCheckbutton').pack(anchor='w', padx=8, pady=2)

        # Pressure control
        pres_frame = ttk.LabelFrame(ctrl, text=' Pressure ')
        pres_frame.pack(fill='x', pady=(0, 6))

        self.sl_pressure = LabeledSlider(pres_frame, 'Pressure', -1, 3, 1,
                                         unit='(log₁₀ Pa)', fmt='{:.1f}',
                                         command=lambda v: self._on_pressure_change())
        self.sl_pressure.pack(fill='x', padx=6, pady=4)

        PressureUnitSelector(pres_frame,
                             on_change=self._on_pressure_unit_change).pack(
            fill='x', padx=6, pady=(0, 4))

        # Live pressure readout
        self.live_pressure_label = ttk.Label(
            pres_frame, text='', style='Big.TLabel', anchor='center')
        self.live_pressure_label.pack(fill='x', padx=6, pady=(0, 2))
        self.live_pressure_detail = ttk.Label(
            pres_frame, text='', style='Dim.TLabel', anchor='center',
            wraplength=250)
        self.live_pressure_detail.pack(fill='x', padx=6, pady=(0, 6))

        # Speed slider
        self.sl_speed = LabeledSlider(ctrl, 'Sim Speed', 0.2, 3.0, 1.0,
                                      fmt='{:.1f}×', command=lambda v: None)
        self.sl_speed.pack(fill='x', padx=6, pady=4)

        # Play / Reset buttons
        btn_frame = ttk.Frame(ctrl, style='Card.TFrame')
        btn_frame.pack(fill='x', pady=6)
        self.btn_play = ttk.Button(btn_frame, text='▶  Start Simulation',
                                   command=self._toggle_play)
        self.btn_play.pack(fill='x', padx=6, pady=2)
        ttk.Button(btn_frame, text='↺  Reset',
                   command=self._reset).pack(fill='x', padx=6, pady=2)

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

        # Statistics
        stats_frame = ttk.LabelFrame(ctrl, text=' Statistics ')
        stats_frame.pack(fill='x', pady=(0, 6))
        self.stats_label = ttk.Label(stats_frame, text='', style='Dim.TLabel',
                                     wraplength=230, justify='left')
        self.stats_label.pack(padx=8, pady=4)

        # Temperature legend
        legend_frame = ttk.LabelFrame(ctrl, text=' Filament Temperature ')
        legend_frame.pack(fill='x', pady=(0, 6))
        self._build_temp_legend(legend_frame)

        # ---------- right: 3D canvas ----------
        plot_frame = ttk.Frame(self)
        plot_frame.pack(side='right', fill='both', expand=True, padx=8, pady=8)

        with plt.rc_context(MPL_STYLE):
            self.fig = Figure(figsize=(9, 7), dpi=100)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.pack(fill='x')
        NavigationToolbar2Tk(self.canvas, toolbar_frame)

    def _build_temp_legend(self, parent):
        """Small horizontal colour-bar showing blue→red temperature scale."""
        with plt.rc_context(MPL_STYLE):
            fig_leg = Figure(figsize=(2.6, 0.5), dpi=100)
            fig_leg.patch.set_facecolor(COLORS['bg_card'])
            ax = fig_leg.add_axes([0.08, 0.55, 0.84, 0.3])
            gradient = np.linspace(0, 1, 256).reshape(1, -1)
            ax.imshow(gradient, aspect='auto', cmap='coolwarm',
                      extent=[self.T_COLD, self.T_HOT, 0, 1])
            ax.set_yticks([])
            mid = (self.T_COLD + self.T_HOT) / 2
            ax.set_xticks([self.T_COLD, mid, self.T_HOT])
            ax.set_xticklabels([f'{self.T_COLD:.0f} K\ncooled',
                                f'{mid:.0f} K', f'{self.T_HOT:.0f} K\nhot'],
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
        p = 10 ** self.sl_pressure.get()
        return max(3, min(80, int(5 + p * 0.5)))

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
        self._init_molecules()
        if not self.running:
            self._draw_scene()

    def _on_ratio_change(self, event=None):
        self._update_composition_display()
        # Switch preset label to Custom if user edits
        self.preset_var.set('custom')
        self.preset_desc.config(text=GAS_MIXTURE_PRESETS['custom']['desc'])
        self._init_molecules()
        if not self.running:
            self._draw_scene()

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
        is_plate = GAUGE_CONFIGS[self.cfg_var.get()]['geometry'] == 'plates'

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
            else:
                r = np.random.uniform(self.R_WIRE + 0.3, self.R_ENC - 0.3, n)
                theta = np.random.uniform(0, 2 * np.pi, n)
                pos[:, 0] = r * np.cos(theta)
                pos[:, 1] = r * np.sin(theta)
                pos[:, 2] = np.random.uniform(0.5, self.L_WIRE - 0.5, n)

            vel = np.random.randn(n, 3)
            norms = np.linalg.norm(vel, axis=1, keepdims=True)
            norms[norms < 1e-8] = 1.0
            vel = vel / norms * np.random.rayleigh(speed * 0.65, (n, 1))

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

    # ── Physics step ─────────────────────────────────────────────────────────

    def _step(self):
        """Advance one simulation timestep: move molecules, detect collisions,
        transfer energy, recover filament temperature."""
        dt = self.DT * self.sl_speed.get()
        is_plate = GAUGE_CONFIGS[self.cfg_var.get()]['geometry'] == 'plates'
        seg_dz = self.L_WIRE / self.N_SEG

        # Move
        self.mol_pos += self.mol_vel * dt

        n = len(self.mol_pos)
        for i in range(n):
            x, y, z = self.mol_pos[i]
            vx, vy, vz = self.mol_vel[i]

            # Per-molecule gas properties
            gk = self.mol_gas_keys[i] if i < len(self.mol_gas_keys) else 'N2'
            gas_i = GAS_DATA[gk]
            aE_i = self._get_accommodation_for(gk)

            if is_plate:
                # ── Plate boundaries ──
                for dim in (0, 1):
                    if self.mol_pos[i, dim] < -self.R_ENC:
                        self.mol_pos[i, dim] = -self.R_ENC + 0.05
                        self.mol_vel[i, dim] = abs(self.mol_vel[i, dim])
                    elif self.mol_pos[i, dim] > self.R_ENC:
                        self.mol_pos[i, dim] = self.R_ENC - 0.05
                        self.mol_vel[i, dim] = -abs(self.mol_vel[i, dim])

                # Bottom plate (cold) — elastic bounce
                if self.mol_pos[i, 2] < 0:
                    self.mol_pos[i, 2] = 0.05
                    self.mol_vel[i, 2] = abs(self.mol_vel[i, 2])

                # Top plate (hot filament) — energy transfer
                if self.mol_pos[i, 2] > self.L_WIRE:
                    self.mol_pos[i, 2] = self.L_WIRE - 0.05
                    self.mol_vel[i, 2] = -abs(self.mol_vel[i, 2])
                    seg_x = (self.mol_pos[i, 0] + self.R_ENC) / (2 * self.R_ENC)
                    si = min(max(int(seg_x * self.N_SEG), 0), self.N_SEG - 1)
                    self._collide(si, aE_i, gas_i, gk)
            else:
                # ── Cylindrical boundaries ──
                r_xy = math.sqrt(x * x + y * y)

                # Enclosure wall
                if r_xy >= self.R_ENC:
                    nx, ny = x / r_xy, y / r_xy
                    vn = vx * nx + vy * ny
                    if vn > 0:
                        self.mol_vel[i, 0] -= 2 * vn * nx
                        self.mol_vel[i, 1] -= 2 * vn * ny
                    self.mol_pos[i, 0] = (self.R_ENC - 0.06) * nx
                    self.mol_pos[i, 1] = (self.R_ENC - 0.06) * ny

                # Wire surface
                if r_xy <= self.R_WIRE + 0.08 and 0 <= z <= self.L_WIRE:
                    if r_xy > 1e-6:
                        nx, ny = x / r_xy, y / r_xy
                    else:
                        a = np.random.uniform(0, 2 * np.pi)
                        nx, ny = math.cos(a), math.sin(a)
                    vn = vx * nx + vy * ny
                    if vn < 0:
                        self.mol_vel[i, 0] -= 2 * vn * nx
                        self.mol_vel[i, 1] -= 2 * vn * ny
                    self.mol_pos[i, 0] = (self.R_WIRE + 0.15) * nx
                    self.mol_pos[i, 1] = (self.R_WIRE + 0.15) * ny
                    si = min(int(z / seg_dz), self.N_SEG - 1)
                    self._collide(si, aE_i, gas_i, gk)

                # End caps
                if self.mol_pos[i, 2] < 0:
                    self.mol_pos[i, 2] = 0.05
                    self.mol_vel[i, 2] = abs(self.mol_vel[i, 2])
                elif self.mol_pos[i, 2] > self.L_WIRE:
                    self.mol_pos[i, 2] = self.L_WIRE - 0.05
                    self.mol_vel[i, 2] = -abs(self.mol_vel[i, 2])

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

        self.seg_temps = np.clip(self.seg_temps, self.T_COLD, self.T_HOT + 5)
        self.frame_count += 1

    def _collide(self, seg_idx, aE, gas, gas_key='N2'):
        """Apply energy transfer from a molecular collision on segment *seg_idx*."""
        T_seg = self.seg_temps[seg_idx]
        frac = (T_seg - self.T_COLD) / (self.T_HOT - self.T_COLD + 1e-9)
        cool = aE * (gas['f'] + 1) / 6.0 * self.COOL_BASE * frac
        self.seg_temps[seg_idx] = max(T_seg - cool, self.T_COLD)
        self.collision_count += 1
        self.total_energy_transferred += cool
        self.collision_per_gas[gas_key] = self.collision_per_gas.get(gas_key, 0) + 1

    # ── Drawing ──────────────────────────────────────────────────────────────

    def _draw_scene(self):
        """Render the complete 3D scene (filament + enclosure + molecules)."""
        # Preserve camera angle across redraws
        if hasattr(self, 'ax3d') and self.ax3d is not None:
            try:
                self._elev = self.ax3d.elev
                self._azim = self.ax3d.azim
            except Exception:
                pass

        self.fig.clear()
        with plt.rc_context(MPL_STYLE):
            self.ax3d = self.fig.add_subplot(111, projection='3d')
            ax = self.ax3d

            cmap = cm.coolwarm
            tnorm = mcolors.Normalize(vmin=self.T_COLD - 5, vmax=self.T_HOT + 5)
            is_plate = GAUGE_CONFIGS[self.cfg_var.get()]['geometry'] == 'plates'

            if is_plate:
                self._draw_plate(ax, cmap, tnorm)
            else:
                self._draw_wire(ax, cmap, tnorm)

            # Molecules — grouped by gas species for colour & size
            if len(self.mol_pos) > 0:
                gas_groups = {}
                for j, gk in enumerate(self.mol_gas_keys):
                    gas_groups.setdefault(gk, []).append(j)

                for gk, indices in gas_groups.items():
                    gas = GAS_DATA[gk]
                    idx = np.array(indices)
                    sz = self.MOL_SIZE_BASE * (self.MOL_RADII[gk] / self.MOL_RADII['N2']) ** 2
                    ax.scatter(self.mol_pos[idx, 0], self.mol_pos[idx, 1],
                               self.mol_pos[idx, 2],
                               s=sz, c=gas['color'], alpha=0.85,
                               edgecolors='white', linewidths=0.4,
                               depthshade=True, zorder=5, label=gas['symbol'])

                if self.show_vectors.get():
                    for j in range(len(self.mol_pos)):
                        gk = self.mol_gas_keys[j] if j < len(self.mol_gas_keys) else 'N2'
                        p = self.mol_pos[j]
                        v = self.mol_vel[j] * 0.25
                        ax.plot([p[0], p[0]+v[0]], [p[1], p[1]+v[1]],
                                [p[2], p[2]+v[2]], color=GAS_DATA[gk]['color'],
                                alpha=0.45, linewidth=0.7)

            # Axes setup
            lim = self.R_ENC * 1.1
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
            try:
                ax.xaxis.pane.fill = False
                ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False
            except Exception:
                pass

            ax.view_init(elev=self._elev, azim=self._azim)

        self.fig.tight_layout()
        self.canvas.draw_idle()
        self._update_stats()

    def _draw_wire(self, ax, cmap, tnorm):
        """Render the cylindrical filament with per-segment temperature colour."""
        theta = np.linspace(0, 2 * np.pi, self.N_THETA)
        z_edges = np.linspace(0, self.L_WIRE, self.N_SEG + 1)
        Theta, Z = np.meshgrid(theta, z_edges)
        X = self.R_WIRE * np.cos(Theta)
        Y = self.R_WIRE * np.sin(Theta)

        fc = np.zeros((self.N_SEG, self.N_THETA - 1, 4))
        for i in range(self.N_SEG):
            fc[i, :] = cmap(tnorm(self.seg_temps[i]))
        ax.plot_surface(X, Y, Z, facecolors=fc, shade=False,
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
        ax.plot_surface(Xc, Yc, Zb, facecolors=fc_bot, shade=False, antialiased=False)
        # top cap
        Zt = np.full_like(Xc, self.L_WIRE)
        fc_top = np.full((*Zt.shape, 4), cmap(tnorm(self.seg_temps[-1])))
        ax.plot_surface(Xc, Yc, Zt, facecolors=fc_top, shade=False, antialiased=False)

        # Enclosure wireframe
        if self.show_enclosure.get():
            te = np.linspace(0, 2 * np.pi, 40)
            for zp in (0, self.L_WIRE):
                ax.plot(self.R_ENC * np.cos(te), self.R_ENC * np.sin(te),
                        zp, color=COLORS['accent'], alpha=0.3, linewidth=0.8)
            for a in np.linspace(0, 2 * np.pi, 10, endpoint=False):
                ax.plot([self.R_ENC * math.cos(a)] * 2,
                        [self.R_ENC * math.sin(a)] * 2,
                        [0, self.L_WIRE],
                        color=COLORS['accent'], alpha=0.15, linewidth=0.5)

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
        p = 10 ** self.sl_pressure.get()
        unit = get_pressure_unit()

        # Update live pressure readout
        self._update_live_pressure()

        # Collision rate (per frame)
        col_rate = self.collision_count / max(self.frame_count, 1)

        # Mixture summary
        mix_parts = []
        for gk, fr in sorted(fracs.items(), key=lambda x: -x[1]):
            gas = GAS_DATA[gk]
            hits = self.collision_per_gas.get(gk, 0)
            mix_parts.append(f"  {gas['symbol']:>4s} {fr*100:5.1f}%  αE={self._get_accommodation_for(gk):.2f}  hits={hits}")

        self.stats_label.config(text=(
            f"p = {format_pressure(p, unit)}\n"
            f"Molecules: {len(self.mol_pos)}\n"
            f"───────────────────\n"
            + '\n'.join(mix_parts) + '\n'
            f"───────────────────\n"
            f"Wire: {T_min:.1f}–{T_max:.1f} K  (eq {self.T_HOT:.0f})\n"
            f"Mean T: {T_avg:.1f} K\n"
            f"Collisions: {self.collision_count}\n"
            f"Rate: {col_rate:.1f} /frame\n"
            f"ΔT total: {self.total_energy_transferred:.1f} K"
        ))

    # ── Animation loop ───────────────────────────────────────────────────────

    def _animate(self):
        if not self.running:
            return
        self._step()
        self._draw_scene()
        self.anim_id = self.after(self.INTERVAL, self._animate)

    def _toggle_play(self):
        self.running = not self.running
        if self.running:
            self.btn_play.config(text='⏸  Pause')
            self._animate()
        else:
            self.btn_play.config(text='▶  Resume')
            if self.anim_id:
                self.after_cancel(self.anim_id)
                self.anim_id = None

    def _reset(self):
        if self.running:
            self.running = False
            if self.anim_id:
                self.after_cancel(self.anim_id)
                self.anim_id = None
        self.seg_temps = np.full(self.N_SEG, self.T_HOT)
        self.collision_count = 0
        self.total_energy_transferred = 0.0
        self.collision_per_gas = {}
        self.frame_count = 0
        self._init_molecules()
        self._draw_scene()
        self.btn_play.config(text='▶  Start Simulation')

    # ── Control callbacks ────────────────────────────────────────────────────

    def _on_config_change(self):
        self.seg_temps = np.full(self.N_SEG, self.T_HOT)
        self._init_molecules()
        if not self.running:
            self._draw_scene()

    def _on_pressure_change(self):
        self._update_live_pressure()
        self._init_molecules()
        if not self.running:
            self._draw_scene()

    def _on_pressure_unit_change(self):
        """Called when the user toggles the pressure unit."""
        self._update_live_pressure()
        if not self.running:
            self._update_stats()

    def _update_live_pressure(self):
        """Refresh the prominent real-time pressure readout."""
        p_pa = 10 ** self.sl_pressure.get()
        unit = get_pressure_unit()
        self.live_pressure_label.config(text=format_pressure(p_pa, unit, fmt='{:.4g}'))
        equivs = [format_pressure(p_pa, u, fmt='{:.3g}')
                  for u in PRESSURE_UNITS if u != unit]
        self.live_pressure_detail.config(text='  \u2261  '.join(equivs))


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

class PiraniSimulatorApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Pirani Vacuum Gauge Simulator — Jousten (2008)")
        self.root.geometry("1400x860")
        self.root.minsize(1060, 620)

        apply_dark_theme(self.root)

        # ── Global shared state ──
        APP_STATE['pressure_unit'] = tk.StringVar(value='Pa')

        # ── Header ──
        header = ttk.Frame(self.root)
        header.pack(fill='x', padx=16, pady=(8, 0))

        ttk.Label(header, text="Pirani Vacuum Gauge Simulator",
                  style='Header.TLabel').pack(side='left')

        # Global pressure unit selector (right side of header)
        unit_frame = ttk.Frame(header)
        unit_frame.pack(side='right', pady=4)
        ttk.Label(unit_frame, text='Pressure Unit:', style='Dim.TLabel').pack(side='left', padx=(0, 6))
        unit_combo = ttk.Combobox(unit_frame, textvariable=APP_STATE['pressure_unit'],
                                  values=list(PRESSURE_UNITS.keys()),
                                  state='readonly', width=7)
        unit_combo.pack(side='left')

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

        for label, TabClass in tabs:
            tab = TabClass(self.notebook)
            self.notebook.add(tab, text=f" {label} ")

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
