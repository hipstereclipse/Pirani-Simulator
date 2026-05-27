#!/usr/bin/env python3
"""
Pirani Vacuum Gauge Simulator
Based on: Jousten (2008) "On the gas species dependence of Pirani vacuum gauges"
J. Vac. Sci. Technol. A 26, 352-359

A comprehensive educational tool for understanding Pirani gauge physics,
gas-dependent heat transfer, correction factors, and accommodation coefficients.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import csv
import io
import json
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
import os
import re
import sys
import math
import time
import traceback
import html as html_lib
import webbrowser
from collections import deque
from html.parser import HTMLParser
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

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
DEFAULT_CONVECTION_GAIN = 0.22

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

DEFAULT_GAS_KEYS = tuple(GAS_DATA.keys())
CUSTOM_GAS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'custom_gases.json')
NIST_WEBBOOK_URL = 'https://webbook.nist.gov/cgi/cbook.cgi'
PUBCHEM_PUG_URL = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name'
PUBCHEM_COMPOUND_URL = 'https://pubchem.ncbi.nlm.nih.gov/compound'
IAEA_LIVECHART_URL = 'https://nds.iaea.org/relnsd/v1/data'

CUSTOM_GAS_STATE = {
    'palette': list(DEFAULT_GAS_KEYS),
    'gases': {},
}
GAS_PALETTE_LISTENERS = []

KINETIC_DIAMETER_PM = {
    'H2': 289.0, 'He': 260.0, 'Ne': 275.0, 'CO': 376.0, 'N2': 364.0,
    'O2': 346.0, 'Ar': 340.0, 'CO2': 330.0, 'Kr': 360.0, 'Xe': 396.0,
    'H2O': 265.0, 'CH4': 380.0, 'NH3': 260.0, 'SF6': 550.0,
    'D2': 289.0, 'T2': 289.0,
}

DOF_OVERRIDES = {
    'H2': 5, 'N2': 5, 'O2': 5, 'CO': 5,
    'CO2': 6, 'H2O': 6, 'CH4': 6, 'NH3': 6, 'SF6': 6,
    'D2': 5, 'T2': 5,
}

QUICK_NIST_GAS_NAMES = [
    'Hydrogen', 'Helium', 'Nitrogen', 'Oxygen', 'Argon', 'Carbon dioxide',
    'Carbon monoxide', 'Neon', 'Krypton', 'Xenon', 'Methane', 'Water',
    'Ammonia', 'Sulfur hexafluoride', 'Chlorine', 'Nitrous oxide',
    'Tritium', 'Molecular tritium', 'Deuterium',
]

IAEA_ISOTOPE_GAS_LOOKUP = {
    'tritium': {'nuclide': '3H', 'formula': 'T2', 'name': 'Tritium gas'},
    'molecular tritium': {'nuclide': '3H', 'formula': 'T2', 'name': 'Tritium gas'},
    'deuterium': {'nuclide': '2H', 'formula': 'D2', 'name': 'Deuterium gas'},
    'molecular deuterium': {'nuclide': '2H', 'formula': 'D2', 'name': 'Deuterium gas'},
}


def _gas_sources_text(gas):
    sources = gas.get('sources') or {}
    if isinstance(sources, dict):
        return '; '.join(f'{k}: {v}' for k, v in sources.items())
    return str(sources or gas.get('source', ''))


def _gas_source_label(gas):
    source = str(gas.get('provider') or gas.get('source') or '')
    if 'PubChem' in source:
        return 'PubChem'
    if 'IAEA' in source:
        return 'IAEA'
    if 'NIST' in source or gas.get('nist_id'):
        return 'NIST'
    return 'Default'


for _gas_key, _gas in GAS_DATA.items():
    _gas.setdefault('nist_id', None)
    _gas.setdefault('source', 'Jousten 2008 and simulator transport defaults')
    _gas.setdefault('sources', {
        'm': 'Jousten 2008 Table I / standard molecular weights',
        'f': 'Jousten 2008 Table I',
        'gamma': 'Jousten 2008 Table I / ideal-gas estimate',
        'cbar': 'Jousten 2008 Table I / kinetic-theory reference',
        'plbar': 'Jousten 2008 Table I',
        'transport': 'Room-temperature engineering values used for convection scaling',
    })


def _formula_atom_count(formula):
    counts = re.findall(r'([A-Z][a-z]?)(\d*)', formula or '')
    total = 0
    for _, count in counts:
        total += int(count) if count else 1
    return total


def _estimate_degrees_of_freedom(formula):
    clean = re.sub(r'[^A-Za-z0-9]', '', formula or '')
    if clean in DOF_OVERRIDES:
        return DOF_OVERRIDES[clean]
    atom_count = _formula_atom_count(clean)
    if atom_count <= 1:
        return 3
    if atom_count == 2:
        return 5
    return 6


def _estimate_molecular_radius_pm(gas_key):
    if gas_key in KINETIC_DIAMETER_PM:
        return KINETIC_DIAMETER_PM[gas_key]
    gas = GAS_DATA.get(gas_key, {})
    formula = re.sub(r'[^A-Za-z0-9]', '', str(gas.get('formula', gas_key)))
    if formula in KINETIC_DIAMETER_PM:
        return KINETIC_DIAMETER_PM[formula]
    mass = max(float(gas.get('m', GAS_DATA['N2']['m'])), 1.0)
    return float(np.clip(255.0 + 22.0 * (mass ** (1.0 / 3.0)), 240.0, 620.0))


def _estimate_plbar(gas_key):
    diameter = max(_estimate_molecular_radius_pm(gas_key), 1.0)
    return GAS_DATA['N2']['plbar'] * (KINETIC_DIAMETER_PM['N2'] / diameter) ** 2


def _estimate_transport_for_gas(gas_key, gas):
    mass = max(float(gas.get('m', GAS_DATA['N2']['m'])), 1.0)
    diameter = max(_estimate_molecular_radius_pm(gas_key), 1.0)
    n2_mass = GAS_DATA['N2']['m']
    n2_diameter = KINETIC_DIAMETER_PM['N2']
    mu = GAS_TRANSPORT['N2']['mu0'] * math.sqrt(mass / n2_mass) * (n2_diameter / diameter) ** 2
    f = max(float(gas.get('f', 5)), 3.0)
    k = GAS_TRANSPORT['N2']['k0'] * math.sqrt(n2_mass / mass) * (f / 5.0) * (n2_diameter / diameter) ** 2
    return {'mu0': float(np.clip(mu, 4e-6, 70e-6)), 'k0': float(np.clip(k, 0.003, 0.25))}


def _estimate_transport(gas_key):
    return _estimate_transport_for_gas(gas_key, GAS_DATA.get(gas_key, GAS_DATA['N2']))


def _format_formula_symbol(formula):
    sub = str.maketrans('0123456789', '0123456789')
    if not formula:
        return ''
    # Keep ASCII symbols for imported gases so persisted JSON remains simple.
    return str(formula).translate(sub)


def _mean_speed_from_mass(mass_amu, temperature_k=296.0):
    mass_kg = max(float(mass_amu), 1e-12) * AMU_TO_KG
    return math.sqrt((8.0 * kB * float(temperature_k)) / (math.pi * mass_kg))


def _deterministic_gas_color(key):
    palette = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c',
               '#e67e22', '#34495e', '#d35400', '#16a085', '#c0392b', '#2980b9']
    idx = sum(ord(ch) for ch in str(key)) % len(palette)
    return palette[idx]


def _canonical_gas_key(name, formula, nist_id=None):
    base = re.sub(r'[^A-Za-z0-9]', '', formula or '')
    if not base:
        base = re.sub(r'[^A-Za-z0-9]+', '_', name or '').strip('_')[:18]
    if not base:
        base = f'NIST_{nist_id or len(GAS_DATA) + 1}'
    if base not in GAS_DATA:
        return base
    current = GAS_DATA.get(base, {})
    if nist_id and current.get('nist_id') == nist_id:
        return base
    slug = re.sub(r'[^A-Za-z0-9]+', '_', name or base).strip('_')[:18]
    candidate = slug or base
    if candidate not in GAS_DATA:
        return candidate
    idx = 2
    while f'{candidate}_{idx}' in GAS_DATA:
        idx += 1
    return f'{candidate}_{idx}'


def _normalize_imported_formula(name, formula, mass_amu=None):
    clean = re.sub(r'[^A-Za-z0-9]', '', formula or '')
    low = f'{name or ""} {clean}'.lower()
    mass = float(mass_amu) if mass_amu is not None else 0.0
    if 'tritium' in low:
        return 'T2' if mass >= 5.0 or clean in ('H2', 'T2') else 'T'
    if 'deuterium' in low:
        return 'D2' if mass >= 3.5 or clean in ('H2', 'D2') else 'D'
    return clean


def _register_simulation_gas(key, gas, transport=None, custom=True):
    if not key or not isinstance(gas, dict):
        return None
    clean_key = str(key)
    GAS_DATA[clean_key] = gas
    GAS_DATA[clean_key].setdefault('color', _deterministic_gas_color(clean_key))
    GAS_DATA[clean_key].setdefault('symbol', clean_key)
    GAS_DATA[clean_key].setdefault('name', clean_key)
    GAS_DATA[clean_key].setdefault('f', _estimate_degrees_of_freedom(GAS_DATA[clean_key].get('formula', clean_key)))
    GAS_DATA[clean_key].setdefault('gamma', round((GAS_DATA[clean_key]['f'] + 2.0) / GAS_DATA[clean_key]['f'], 2))
    GAS_DATA[clean_key].setdefault('cbar', int(round(_mean_speed_from_mass(GAS_DATA[clean_key].get('m', GAS_DATA['N2']['m'])))))
    GAS_DATA[clean_key].setdefault('plbar', _estimate_plbar(clean_key))
    GAS_TRANSPORT[clean_key] = transport or GAS_TRANSPORT.get(clean_key) or _estimate_transport(clean_key)
    ACCOM_RATIOS_W.setdefault(clean_key, 1.0)
    ACCOM_RATIOS_Si.setdefault(clean_key, 1.0)
    if custom:
        CUSTOM_GAS_STATE.setdefault('gases', {})[clean_key] = {
            'gas': GAS_DATA[clean_key],
            'transport': GAS_TRANSPORT[clean_key],
        }
    return clean_key


def _load_custom_gases():
    if not os.path.exists(CUSTOM_GAS_FILE):
        return
    try:
        with open(CUSTOM_GAS_FILE, 'r', encoding='utf-8') as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return

    palette = payload.get('palette', [])
    gases = payload.get('gases', {})
    CUSTOM_GAS_STATE['palette'] = [k for k in palette if isinstance(k, str)] or list(DEFAULT_GAS_KEYS)
    CUSTOM_GAS_STATE['gases'] = {}
    for key, entry in gases.items():
        if not isinstance(entry, dict):
            continue
        gas = entry.get('gas', entry)
        transport = entry.get('transport')
        if isinstance(gas, dict):
            _register_simulation_gas(key, gas, transport=transport, custom=True)

    CUSTOM_GAS_STATE['palette'] = [k for k in CUSTOM_GAS_STATE['palette'] if k in GAS_DATA]
    if not CUSTOM_GAS_STATE['palette']:
        CUSTOM_GAS_STATE['palette'] = list(DEFAULT_GAS_KEYS)


def save_custom_gases():
    payload = {
        'palette': [k for k in CUSTOM_GAS_STATE.get('palette', []) if k in GAS_DATA],
        'gases': CUSTOM_GAS_STATE.get('gases', {}),
    }
    try:
        with open(CUSTOM_GAS_FILE, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
    except OSError as exc:
        messagebox.showwarning('Gas Palette', f'Could not save custom gas palette:\n{exc}')


def get_gas_palette_keys(mode='default'):
    if mode == 'custom':
        keys = [k for k in CUSTOM_GAS_STATE.get('palette', []) if k in GAS_DATA]
        return keys or list(DEFAULT_GAS_KEYS)
    return [k for k in DEFAULT_GAS_KEYS if k in GAS_DATA]


def get_all_simulation_gas_keys():
    return list(GAS_DATA.keys())


def _gas_combo_label(key):
    gas = GAS_DATA[key]
    return f"{gas.get('symbol', key)} ({key})"


def _gas_key_from_combo(value, fallback='N2'):
    val = str(value or '')
    m = re.search(r'\(([^)]+)\)\s*$', val)
    if m and m.group(1) in GAS_DATA:
        return m.group(1)
    if val in GAS_DATA:
        return val
    for key in GAS_DATA:
        if val.startswith(f"{GAS_DATA[key].get('symbol', key)} "):
            return key
    return fallback if fallback in GAS_DATA else next(iter(GAS_DATA))


def register_gas_palette_listener(callback):
    if callable(callback) and callback not in GAS_PALETTE_LISTENERS:
        GAS_PALETTE_LISTENERS.append(callback)


def notify_gas_palette_changed():
    for callback in list(GAS_PALETTE_LISTENERS):
        try:
            callback()
        except Exception:
            traceback.print_exc()


class _NISTSearchParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.results = []
        self._href = None
        self._text = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() != 'a':
            return
        attrs = dict(attrs)
        href = attrs.get('href', '')
        if 'cbook.cgi' in href and ('ID=' in href or 'Name=' in href):
            self._href = href
            self._text = []

    def handle_data(self, data):
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag):
        if tag.lower() != 'a' or self._href is None:
            return
        name = html_lib.unescape(''.join(self._text)).strip()
        href = html_lib.unescape(self._href)
        self._href = None
        self._text = []
        if not name or name.lower() in ('nist chemistry webbook', 'main site page'):
            return
        parsed = urlparse.urlparse(href)
        qs = urlparse.parse_qs(parsed.query)
        nist_id = (qs.get('ID') or [''])[0]
        if not nist_id:
            return
        if any(r.get('nist_id') == nist_id for r in self.results):
            return
        url = href if href.startswith('http') else urlparse.urljoin(NIST_WEBBOOK_URL, href)
        self.results.append({
            'name': name,
            'provider': 'NIST',
            'provider_id': nist_id,
            'nist_id': nist_id,
            'url': url,
        })


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        if data and data.strip():
            self.parts.append(data.strip())

    def text(self):
        return '\n'.join(self.parts)


def _fetch_url(url, timeout=12):
    req = urlrequest.Request(url, headers={'User-Agent': 'Mozilla/5.0 PiraniSimulator/1.0'})
    with urlrequest.urlopen(req, timeout=timeout) as resp:
        charset = resp.headers.get_content_charset() or 'utf-8'
        return resp.read().decode(charset, errors='replace')


def _nist_search_url(query):
    return NIST_WEBBOOK_URL + '?' + urlparse.urlencode({'Name': query, 'Units': 'SI'})


def search_nist_gases(query, limit=30):
    html = _fetch_url(_nist_search_url(query))
    parser = _NISTSearchParser()
    parser.feed(html)
    results = parser.results[:limit]
    if results:
        return results
    detail = parse_nist_species_page(html, _nist_search_url(query))
    if detail:
        return [{
            'name': detail['name'],
            'provider': 'NIST',
            'provider_id': detail.get('nist_id', ''),
            'nist_id': detail.get('nist_id', ''),
            'url': detail['source_url'],
            'detail': detail,
        }]
    return []


def parse_nist_species_page(html, source_url):
    text_parser = _TextExtractor()
    text_parser.feed(html)
    text = text_parser.text()
    h1_match = re.search(r'<h1[^>]*>(.*?)</h1>', html, flags=re.I | re.S)
    name = html_lib.unescape(re.sub(r'<.*?>', '', h1_match.group(1))).strip() if h1_match else ''
    if not name:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        name = lines[0] if lines else 'NIST gas'

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    def value_after_label(label):
        for idx, line in enumerate(lines):
            if line.lower().startswith(label.lower()):
                value = line.split(':', 1)[1].strip() if ':' in line else ''
                if value:
                    return value
                if idx + 1 < len(lines):
                    return lines[idx + 1].strip()
        return ''

    formula = re.sub(r'[^A-Za-z0-9]', '', value_after_label('Formula'))
    mw_match = re.search(r'([0-9]+(?:\.[0-9]+)?)', value_after_label('Molecular weight'))
    if mw_match is None:
        return None
    mass = float(mw_match.group(1))
    cas_match = re.search(r'([0-9\-]+)', value_after_label('CAS Registry Number'))
    parsed = urlparse.urlparse(source_url)
    qs = urlparse.parse_qs(parsed.query)
    nist_id = (qs.get('ID') or [''])[0]
    return {
        'name': name,
        'formula': _normalize_imported_formula(name, formula, mass),
        'm': mass,
        'cas': cas_match.group(1) if cas_match else '',
        'nist_id': nist_id,
        'provider': 'NIST',
        'provider_id': nist_id,
        'source_url': source_url,
        'source': 'NIST Chemistry WebBook',
    }


def fetch_nist_gas_detail(result):
    if result.get('detail'):
        return result['detail']
    url = result.get('url')
    if not url and result.get('nist_id'):
        url = NIST_WEBBOOK_URL + '?' + urlparse.urlencode({'ID': result['nist_id'], 'Units': 'SI'})
    if not url:
        return None
    html = _fetch_url(url)
    return parse_nist_species_page(html, url)


def _pubchem_property_url(query):
    props = 'MolecularFormula,MolecularWeight,IUPACName,CanonicalSMILES'
    return f"{PUBCHEM_PUG_URL}/{urlparse.quote(query)}/property/{props}/JSON"


def fetch_pubchem_gas_detail_by_name(query):
    url = _pubchem_property_url(query)
    payload = json.loads(_fetch_url(url))
    props = payload.get('PropertyTable', {}).get('Properties', [])
    if not props:
        return None
    prop = props[0]
    cid = prop.get('CID')
    mass = float(prop.get('MolecularWeight'))
    formula = _normalize_imported_formula(query, prop.get('MolecularFormula', ''), mass)
    name = prop.get('IUPACName') or str(query).strip().title()
    if formula == 'T2' and 'tritium' in str(query).lower():
        name = 'Tritium gas'
    elif formula == 'D2' and 'deuterium' in str(query).lower():
        name = 'Deuterium gas'
    source_url = f'{PUBCHEM_COMPOUND_URL}/{cid}' if cid else 'https://pubchem.ncbi.nlm.nih.gov/'
    return {
        'name': name,
        'formula': formula,
        'm': mass,
        'cas': '',
        'pubchem_cid': cid,
        'provider': 'PubChem',
        'provider_id': str(cid or ''),
        'source_url': source_url,
        'source_data_url': url,
        'source': 'PubChem PUG REST (NIH/NLM)',
    }


def search_pubchem_gases(query, limit=10):
    detail = fetch_pubchem_gas_detail_by_name(query)
    if not detail:
        return []
    return [{
        'name': detail['name'],
        'provider': 'PubChem',
        'provider_id': detail.get('provider_id', ''),
        'url': detail.get('source_url', ''),
        'detail': detail,
    }]


def _iaea_match_for_query(query):
    low = str(query or '').lower()
    for token, info in IAEA_ISOTOPE_GAS_LOOKUP.items():
        if token in low:
            return info
    return None


def _iaea_nuclide_url(nuclide):
    return IAEA_LIVECHART_URL + '?' + urlparse.urlencode({'fields': 'ground_states', 'nuclides': nuclide})


def fetch_iaea_isotope_gas_detail(query):
    info = _iaea_match_for_query(query)
    if not info:
        return None
    url = _iaea_nuclide_url(info['nuclide'])
    text = _fetch_url(url)
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        return None
    row = rows[0]
    atomic_mass_micro_u = float(row.get('atomic_mass') or 0.0)
    if atomic_mass_micro_u <= 0.0:
        return None
    atom_mass_u = atomic_mass_micro_u / 1e6
    atom_count = max(_formula_atom_count(info['formula']), 1)
    mass = atom_mass_u * atom_count
    return {
        'name': info['name'],
        'formula': info['formula'],
        'm': mass,
        'cas': '',
        'provider': 'IAEA',
        'provider_id': info['nuclide'],
        'source_url': url,
        'source': 'IAEA LiveChart of Nuclides',
        'half_life': row.get('half_life', ''),
        'half_life_unit': row.get('unit_hl', ''),
    }


def search_iaea_isotope_gases(query, limit=10):
    detail = fetch_iaea_isotope_gas_detail(query)
    if not detail:
        return []
    return [{
        'name': detail['name'],
        'provider': 'IAEA',
        'provider_id': detail.get('provider_id', ''),
        'url': detail.get('source_url', ''),
        'detail': detail,
    }]


def search_government_gases(query, limit=30):
    results = []
    errors = []
    for search_fn in (search_nist_gases, search_pubchem_gases, search_iaea_isotope_gases):
        try:
            results.extend(search_fn(query, limit=limit))
        except (urlerror.URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append(exc)
    seen = set()
    unique = []
    for result in results:
        ident = (result.get('provider'), result.get('provider_id'), result.get('name'))
        if ident in seen:
            continue
        seen.add(ident)
        unique.append(result)
    if not unique and errors:
        raise errors[-1]
    return unique[:limit]


def fetch_government_gas_detail(result, fallback_query=None):
    provider = result.get('provider')
    detail = None
    if provider == 'PubChem':
        detail = result.get('detail') or fetch_pubchem_gas_detail_by_name(result.get('name', ''))
    elif provider == 'IAEA':
        detail = result.get('detail') or fetch_iaea_isotope_gas_detail(result.get('name', ''))
    else:
        detail = fetch_nist_gas_detail(result)
    if detail:
        return detail

    terms = []
    for term in (result.get('name'), fallback_query):
        if term and term not in terms:
            terms.append(term)
    for term in terms:
        for detail_fn in (fetch_pubchem_gas_detail_by_name, fetch_iaea_isotope_gas_detail):
            try:
                detail = detail_fn(term)
            except (urlerror.URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError):
                detail = None
            if detail:
                return detail
    return None


def build_gas_from_nist_detail(detail):
    name = detail.get('name', 'NIST gas')
    formula = _normalize_imported_formula(name, detail.get('formula', ''), detail.get('m'))
    key = _canonical_gas_key(name, formula, detail.get('nist_id'))
    f = _estimate_degrees_of_freedom(formula)
    gamma = round((f + 2.0) / f, 2)
    mass = float(detail['m'])
    cbar = int(round(_mean_speed_from_mass(mass)))
    source = detail.get('source', 'Government molecular data source')
    source_url = detail.get('source_url', '')
    gas = {
        'name': name,
        'symbol': _format_formula_symbol(formula) or key,
        'formula': formula,
        'm': mass,
        'f': f,
        'gamma': gamma,
        'cbar': cbar,
        'plbar': _estimate_plbar(key),
        'color': _deterministic_gas_color(key),
        'nist_id': detail.get('nist_id'),
        'pubchem_cid': detail.get('pubchem_cid'),
        'provider': detail.get('provider'),
        'provider_id': detail.get('provider_id'),
        'cas': detail.get('cas', ''),
        'source': f'{source} + simulator estimates',
        'source_url': source_url,
        'source_data_url': detail.get('source_data_url', source_url),
        'sources': {
            'm': f"{source} ({source_url})",
            'formula': f"{source} ({source_url})",
            'f': 'Estimated from molecular formula for room-temperature gas behavior',
            'gamma': 'Estimated from degrees of freedom using ideal-gas Cp/Cv',
            'cbar': 'Calculated from source molecular weight at 296 K',
            'plbar': 'Estimated from kinetic diameter/fallback mean-free-path scaling',
            'transport': 'Estimated from N2 transport scaling for convection visualization',
        },
    }
    transport = _estimate_transport_for_gas(key, gas)
    return key, gas, transport


_load_custom_gases()

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
        'conv_gas_sensitivity': 0.75,
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

    Convection-enhanced horizontal wires keep gas transport differences visible
    at high pressure. Square cavities amplify non-uniform flow paths.
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
        return 0.60
    return 0.55


def _effective_gravity(config):
    """Effective gravitational acceleration used by buoyancy/convection terms."""
    try:
        return max(float(config.get('gravity_m_s2', G_STD)), 0.0)
    except (TypeError, ValueError):
        return G_STD


def _convection_characteristic_length(config):
    """Characteristic length for natural-convection scaling from gauge geometry."""
    geom = config.get('geometry')
    if geom == 'plates':
        return max(float(config.get('gap', 1e-3)), 1e-6)
    r2 = float(config.get('enc_r', 8e-3))
    r1 = float(config.get('wire_r', 5e-6))
    return max(r2 - r1, 5e-5)


def _convection_geometry_factor(config):
    """Geometry/orientation coupling for buoyant flow reaching the hot element."""
    geom = config.get('geometry')
    orient = config.get('orientation', 'vertical')
    if geom == 'square_cavity':
        return 0.95
    if geom == 'plates':
        return 0.45
    if orient == 'horizontal':
        return 1.15
    return 0.70


def _convection_pressure_activation(config, p_pa):
    """Smooth pressure gate so buoyancy appears in the gauge's high-pressure range."""
    p_mbar = convert_pressure(np.maximum(np.asarray(p_pa, dtype=np.float64), 0.0), 'mbar')
    p_on = max(float(config.get('conv_p_on_mbar', 80.0)), 1e-9)
    n = max(float(config.get('conv_transition_n', 1.25)), 0.4)
    scaled = np.power(p_mbar / p_on, n)
    activation = scaled / (1.0 + scaled)
    if np.ndim(activation) == 0:
        return float(activation)
    return activation


def _natural_convection_nusselt(ra, pr, config):
    """Return an educational Churchill-Chu style natural-convection Nusselt number."""
    ra_safe = np.maximum(ra, 0.0)
    pr_safe = np.maximum(pr, 1e-12)
    geom = config.get('geometry')
    orient = config.get('orientation', 'vertical')
    if geom == 'cylindrical' and orient == 'horizontal':
        return 0.36 + (0.518 * np.power(ra_safe, 0.25)) / np.power(
            1.0 + np.power(0.559 / pr_safe, 9.0 / 16.0), 4.0 / 9.0)
    return 0.68 + (0.67 * np.power(ra_safe, 0.25)) / np.power(
        1.0 + np.power(0.492 / pr_safe, 9.0 / 16.0), 4.0 / 9.0)


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

    l_char = _convection_characteristic_length(config)
    g_eff = _effective_gravity(config)
    if g_eff <= 0.0:
        return 0.0

    beta = 1.0 / t_film
    ra = g_eff * beta * d_t * (l_char ** 3) / max(nu * alpha, 1e-18)
    ra = float(np.clip(ra, 0.0, 1e12))

    nu_nat = _natural_convection_nusselt(ra, pr, config)

    conv_strength = max(float(nu_nat) - 1.0, 0.0)
    ra_on = max(float(config.get('conv_ra_on', 70.0)), 1e-9)
    ra_n = max(float(config.get('conv_transition_n', 1.25)), 0.4)
    ra_scale = (ra / ra_on) ** ra_n
    activation = _convection_pressure_activation(config, p_pa) * (ra_scale / (1.0 + ra_scale))

    gain = float(config.get('conv_gain', DEFAULT_CONVECTION_GAIN))
    max_mult = float(config.get('conv_max_mult', 1.4))
    mult = gain * _convection_geometry_factor(config) * conv_strength * activation
    return float(np.clip(mult, 0.0, max_mult))


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

    l_char = _convection_characteristic_length(config)
    g_eff = _effective_gravity(config)
    if g_eff <= 0.0:
        return np.zeros_like(pressures_pa)

    beta = 1.0 / t_film
    ra = g_eff * beta * d_t * (l_char ** 3) / np.maximum(nu * alpha, 1e-18)
    ra = np.clip(ra, 0.0, 1e12)

    nu_nat = _natural_convection_nusselt(ra, pr, config)

    conv_strength = np.maximum(nu_nat - 1.0, 0.0)
    ra_on = max(float(config.get('conv_ra_on', 70.0)), 1e-9)
    ra_n = max(float(config.get('conv_transition_n', 1.25)), 0.4)
    ra_scale = np.power(np.maximum(ra, 0.0) / ra_on, ra_n)
    activation = _convection_pressure_activation(config, pressures_pa) * (ra_scale / (1.0 + ra_scale))

    gain = float(config.get('conv_gain', DEFAULT_CONVECTION_GAIN))
    max_mult = float(config.get('conv_max_mult', 1.4))
    mult = gain * _convection_geometry_factor(config) * conv_strength * activation
    return np.clip(mult, 0.0, max_mult)

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


_N2_HEAT_FLOW_LOOKUP_CACHE = {}


def _freeze_for_cache(value):
    if isinstance(value, dict):
        return tuple(sorted((str(k), _freeze_for_cache(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_for_cache(v) for v in value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, (str, int, bool, type(None))):
        return value
    return repr(value)


def _cache_put_bounded(cache, key, value, limit=48):
    if key not in cache and len(cache) >= limit:
        cache.pop(next(iter(cache)))
    cache[key] = value


def _n2_heat_flow_lookup(config, aN2=0.6, t_hot_override=None, t_cold_override=None):
    key = (
        _freeze_for_cache(config),
        round(float(aN2), 12),
        None if t_hot_override is None else round(float(t_hot_override), 12),
        None if t_cold_override is None else round(float(t_cold_override), 12),
    )
    cached = _N2_HEAT_FLOW_LOOKUP_CACHE.get(key)
    if cached is not None:
        return cached

    p_ref = np.logspace(-10, 6.3, 480)
    q_ref, _, _ = calc_heat_flow_vec(
        'N2',
        config,
        p_ref,
        aN2=aN2,
        t_hot_override=t_hot_override,
        t_cold_override=t_cold_override,
    )

    valid = np.isfinite(q_ref) & (q_ref > 0.0)
    if np.count_nonzero(valid) < 2:
        table = (np.array([], dtype=np.float64), np.array([], dtype=np.float64))
        _cache_put_bounded(_N2_HEAT_FLOW_LOOKUP_CACHE, key, table)
        return table

    log_q_ref = np.log(q_ref[valid])
    log_p_ref = np.log(p_ref[valid])
    order = np.argsort(log_q_ref)
    log_q_ref = log_q_ref[order]
    log_p_ref = log_p_ref[order]
    unique = np.concatenate(([True], np.diff(log_q_ref) > 1e-12))
    table = (log_q_ref[unique], log_p_ref[unique])
    _cache_put_bounded(_N2_HEAT_FLOW_LOOKUP_CACHE, key, table)
    return table


def _invert_n2_heat_flow_array(q_values, config, aN2=0.6,
                               t_hot_override=None, t_cold_override=None,
                               fallback=None):
    q_arr = np.asarray(q_values, dtype=np.float64)
    scalar = q_arr.ndim == 0
    log_q_ref, log_p_ref = _n2_heat_flow_lookup(
        config,
        aN2=aN2,
        t_hot_override=t_hot_override,
        t_cold_override=t_cold_override,
    )
    if len(log_q_ref) < 2:
        result = np.asarray(fallback, dtype=np.float64) if fallback is not None else np.maximum(q_arr, 1e-20)
        return float(result) if scalar else result

    q_min = math.exp(float(log_q_ref[0]))
    log_q = np.log(np.maximum(q_arr, q_min))
    log_p_ind = np.interp(
        log_q,
        log_q_ref,
        log_p_ref,
        left=float(log_p_ref[0]),
        right=float(log_p_ref[-1]),
    )
    result = np.exp(log_p_ind)
    return float(result) if scalar else result


def _normalize_mixture_fractions(fracs):
    cleaned = {}
    for gas_key, fraction in (fracs or {}).items():
        if gas_key not in GAS_DATA:
            continue
        try:
            value = max(float(fraction), 0.0)
        except (TypeError, ValueError):
            continue
        if value > 0.0:
            cleaned[gas_key] = value

    total = sum(cleaned.values())
    if total <= 0.0:
        return {'N2': 1.0}
    return {gas_key: value / total for gas_key, value in cleaned.items()}


def calc_mixture_heat_flow(fracs, config, p, aN2=0.6, t_hot_override=None, t_cold_override=None):
    """Calculate mixture heat flow by mole-fraction weighting gas heat flows."""
    q_combined_total = 0.0
    q_mol_total = 0.0
    q_visc_total = 0.0
    for gas_key, fraction in _normalize_mixture_fractions(fracs).items():
        q_combined, q_mol, q_visc = calc_heat_flow(
            gas_key,
            config,
            p,
            aN2=aN2,
            t_hot_override=t_hot_override,
            t_cold_override=t_cold_override,
        )
        q_combined_total += fraction * q_combined
        q_mol_total += fraction * q_mol
        q_visc_total += fraction * q_visc
    return q_combined_total, q_mol_total, q_visc_total


def calc_mixture_heat_flow_vec(fracs, config, pressures, aN2=0.6,
                               t_hot_override=None, t_cold_override=None):
    """Vectorised mixture heat flow by mole-fraction weighting gas heat flows."""
    pressures = np.asarray(pressures, dtype=np.float64)
    q_combined_total = np.zeros_like(pressures, dtype=np.float64)
    q_mol_total = np.zeros_like(pressures, dtype=np.float64)
    q_visc_total = np.zeros_like(pressures, dtype=np.float64)
    for gas_key, fraction in _normalize_mixture_fractions(fracs).items():
        q_combined, q_mol, q_visc = calc_heat_flow_vec(
            gas_key,
            config,
            pressures,
            aN2=aN2,
            t_hot_override=t_hot_override,
            t_cold_override=t_cold_override,
        )
        q_combined_total += fraction * q_combined
        q_mol_total += fraction * q_mol
        q_visc_total += fraction * q_visc
    return q_combined_total, q_mol_total, q_visc_total


def _invert_n2_heat_flow_for_config(q_target, config, aN2=0.6,
                                    t_hot_override=None, t_cold_override=None,
                                    fallback=None):
    q_target = max(float(q_target), 0.0)
    return _invert_n2_heat_flow_array(
        q_target,
        config,
        aN2=aN2,
        t_hot_override=t_hot_override,
        t_cold_override=t_cold_override,
        fallback=fallback,
    )


def calc_mixture_correction_factor_physics(fracs, config, p_true, aN2=0.6,
                                           t_hot_override=None, t_cold_override=None):
    """Return p_true / p_indicated from the simulated N2-calibrated heat-flow curve."""
    p_real = max(float(p_true), 1e-20)
    q_mix, _, _ = calc_mixture_heat_flow(
        fracs,
        config,
        p_real,
        aN2=aN2,
        t_hot_override=t_hot_override,
        t_cold_override=t_cold_override,
    )
    p_indicated = _invert_n2_heat_flow_for_config(
        q_mix,
        config,
        aN2=aN2,
        t_hot_override=t_hot_override,
        t_cold_override=t_cold_override,
        fallback=p_real,
    )
    return p_real / p_indicated if p_indicated > 0.0 else float('inf')


def calc_mixture_indicated_pressure_curve_physics(fracs, config, pressures, aN2=0.6,
                                                 t_hot_override=None, t_cold_override=None):
    """Return N2-calibrated indicated pressure for a mixture pressure curve."""
    p_true = np.maximum(np.asarray(pressures, dtype=np.float64), 1e-20)
    q_mix, _, _ = calc_mixture_heat_flow_vec(
        fracs,
        config,
        p_true,
        aN2=aN2,
        t_hot_override=t_hot_override,
        t_cold_override=t_cold_override,
    )
    return _invert_n2_heat_flow_array(
        q_mix,
        config,
        aN2=aN2,
        t_hot_override=t_hot_override,
        t_cold_override=t_cold_override,
        fallback=p_true,
    )


def calc_correction_factor_curve_physics(gas_key, config, pressures, aN2=0.6,
                                         t_hot_override=None, t_cold_override=None):
    """Vector pressure-dependent CF from simulated gas heat flow and N2 inversion."""
    p_true = np.maximum(np.asarray(pressures, dtype=np.float64), 1e-20)
    q_gas, _, _ = calc_heat_flow_vec(
        gas_key,
        config,
        p_true,
        aN2=aN2,
        t_hot_override=t_hot_override,
        t_cold_override=t_cold_override,
    )
    p_indicated = _invert_n2_heat_flow_array(
        q_gas,
        config,
        aN2=aN2,
        t_hot_override=t_hot_override,
        t_cold_override=t_cold_override,
        fallback=p_true,
    )
    return np.where(p_indicated > 0.0, p_true / p_indicated, np.inf)


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


class GasPaletteDialog:
    """Popup for importing government-sourced gases and editing the custom palette."""

    def __init__(self, parent, on_close=None):
        self.parent = parent
        self.on_close = on_close
        self.nist_results = []
        self.selected_result = None
        self.current_source_url = ''
        self._filter_text = tk.StringVar(value='')

        self.dlg = tk.Toplevel(parent)
        self.dlg.title('Custom Gas Palette')
        self.dlg.geometry('1120x700')
        self.dlg.minsize(980, 600)
        self.dlg.transient(parent.winfo_toplevel())
        self.dlg.protocol('WM_DELETE_WINDOW', self._close)

        self._build_ui()
        self._refresh_known_tree()
        self._show_gas_details('N2')

    def _build_ui(self):
        footer = ttk.Frame(self.dlg)
        footer.pack(side='bottom', fill='x', padx=10, pady=(0, 10))
        self.footer_status = ttk.Label(footer, text='', style='Dim.TLabel')
        self.footer_status.pack(side='left')
        ttk.Button(footer, text='Save Palette', command=self._save).pack(side='right', padx=(6, 0))
        ttk.Button(footer, text='Close', command=self._close).pack(side='right')

        outer = ttk.Frame(self.dlg)
        outer.pack(side='top', fill='both', expand=True, padx=10, pady=10)

        left = ttk.Frame(outer)
        left.pack(side='left', fill='both', expand=True, padx=(0, 8))
        right = ttk.Frame(outer)
        right.pack(side='right', fill='both', expand=True, padx=(8, 0))

        nist_frame = ttk.LabelFrame(left, text=' Government Gas Data Search ')
        nist_frame.pack(fill='both', expand=True, pady=(0, 8))

        search_row = ttk.Frame(nist_frame, style='Card.TFrame')
        search_row.pack(fill='x', padx=8, pady=(8, 4))
        self.nist_query = tk.StringVar(value='argon')
        ttk.Entry(search_row, textvariable=self.nist_query).pack(side='left', fill='x', expand=True, padx=(0, 6))
        ttk.Button(search_row, text='Search Sources', command=self._search_nist).pack(side='left', padx=(0, 4))

        quick_row = ttk.Frame(nist_frame, style='Card.TFrame')
        quick_row.pack(fill='x', padx=8, pady=(0, 4))
        ttk.Label(quick_row, text='Quick:', style='Dim.TLabel').pack(side='left', padx=(0, 4))
        self.quick_var = tk.StringVar(value=QUICK_NIST_GAS_NAMES[0])
        quick_combo = ttk.Combobox(quick_row, textvariable=self.quick_var,
                                   values=QUICK_NIST_GAS_NAMES, state='readonly', width=22)
        quick_combo.pack(side='left')
        quick_combo.bind('<<ComboboxSelected>>', self._quick_search)

        cols = ('source', 'name', 'provider_id')
        self.nist_tree = ttk.Treeview(nist_frame, columns=cols, show='headings', height=8)
        self.nist_tree.heading('source', text='Source')
        self.nist_tree.heading('name', text='Name')
        self.nist_tree.heading('provider_id', text='ID')
        self.nist_tree.column('source', width=80)
        self.nist_tree.column('name', width=250)
        self.nist_tree.column('provider_id', width=90)
        self.nist_tree.pack(fill='both', expand=True, padx=8, pady=4)
        self.nist_tree.bind('<<TreeviewSelect>>', self._on_nist_select)
        self.nist_tree.bind('<Double-1>', lambda e: self._import_selected_nist())

        nist_btns = ttk.Frame(nist_frame, style='Card.TFrame')
        nist_btns.pack(fill='x', padx=8, pady=(0, 8))
        ttk.Button(nist_btns, text='Fetch Details', command=self._fetch_selected_details).pack(side='left')
        ttk.Button(nist_btns, text='Import + Use', command=self._import_selected_nist).pack(side='left', padx=(6, 0))
        self.nist_status = ttk.Label(nist_btns, text='', style='Dim.TLabel')
        self.nist_status.pack(side='right')

        detail_frame = ttk.LabelFrame(left, text=' Selected Gas Properties and Sources ')
        detail_frame.pack(fill='both', expand=True)
        detail_body = ttk.Frame(detail_frame, style='Card.TFrame')
        detail_body.pack(fill='both', expand=True, padx=8, pady=(8, 4))
        self.detail_text = tk.Text(detail_body, wrap='word', height=10, font=(FONT_MONO, 9),
                                   bg=COLORS['bg_input'], fg=COLORS['text'], insertbackground=COLORS['text'])
        detail_scroll = ttk.Scrollbar(detail_body, orient='vertical', command=self.detail_text.yview)
        self.detail_text.configure(yscrollcommand=detail_scroll.set)
        self.detail_text.pack(side='left', fill='both', expand=True)
        detail_scroll.pack(side='right', fill='y')
        self.detail_text.configure(state='disabled')

        detail_actions = ttk.Frame(detail_frame, style='Card.TFrame')
        detail_actions.pack(fill='x', padx=8, pady=(0, 8))
        self.source_hint = ttk.Label(detail_actions, text='No source selected', style='Dim.TLabel')
        self.source_hint.pack(side='left')
        self.source_button = ttk.Button(detail_actions, text='Open Source', command=self._open_current_source)
        self.source_button.pack(side='right')
        self.source_button.configure(state='disabled')

        palette_frame = ttk.LabelFrame(right, text=' Custom Palette ')
        palette_frame.pack(fill='both', expand=True)

        filter_row = ttk.Frame(palette_frame, style='Card.TFrame')
        filter_row.pack(fill='x', padx=8, pady=(8, 4))
        ttk.Label(filter_row, text='Filter:', style='Dim.TLabel').pack(side='left')
        filter_entry = ttk.Entry(filter_row, textvariable=self._filter_text)
        filter_entry.pack(side='left', fill='x', expand=True, padx=(6, 0))
        filter_entry.bind('<KeyRelease>', lambda e: self._refresh_known_tree())

        cols = ('order', 'key', 'name', 'mass', 'source')
        self.known_tree = ttk.Treeview(palette_frame, columns=cols, show='headings', height=15, selectmode='browse')
        headings = {'order': '#', 'key': 'Key', 'name': 'Gas', 'mass': 'amu', 'source': 'Source'}
        widths = {'order': 42, 'key': 76, 'name': 220, 'mass': 64, 'source': 100}
        for col in cols:
            self.known_tree.heading(col, text=headings[col])
            self.known_tree.column(col, width=widths[col])
        self.known_tree.pack(fill='both', expand=True, padx=8, pady=4)
        self.known_tree.bind('<<TreeviewSelect>>', self._on_known_select)
        self.known_tree.bind('<Double-1>', lambda e: self._toggle_selected_known())

        pal_btns = ttk.Frame(palette_frame, style='Card.TFrame')
        pal_btns.pack(fill='x', padx=8, pady=(0, 3))
        ttk.Button(pal_btns, text='Add', command=self._add_selected_known).pack(side='left')
        ttk.Button(pal_btns, text='Remove', command=self._remove_selected_known).pack(side='left', padx=(6, 0))
        ttk.Button(pal_btns, text='Move Up', command=lambda: self._move_selected_palette(-1)).pack(side='left', padx=(6, 0))
        ttk.Button(pal_btns, text='Move Down', command=lambda: self._move_selected_palette(1)).pack(side='left', padx=(6, 0))

        pal_btns2 = ttk.Frame(palette_frame, style='Card.TFrame')
        pal_btns2.pack(fill='x', padx=8, pady=(0, 8))
        ttk.Button(pal_btns2, text='Reset to Default Order', command=self._reset_palette).pack(side='left')

    def _quick_search(self, event=None):
        self.nist_query.set(self.quick_var.get())
        self._search_nist()

    def _set_status(self, text):
        self.nist_status.config(text=text)
        self.dlg.update_idletasks()

    def _search_nist(self):
        query = self.nist_query.get().strip()
        if not query:
            return
        self._set_status('Searching...')
        try:
            self.nist_results = search_government_gases(query)
        except (urlerror.URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError) as exc:
            self.nist_results = []
            self._set_status('Sources unavailable')
            messagebox.showwarning('Gas Search', f'Could not query government gas sources:\n{exc}', parent=self.dlg)
            return
        for item in self.nist_tree.get_children():
            self.nist_tree.delete(item)
        for idx, result in enumerate(self.nist_results):
            self.nist_tree.insert('', 'end', iid=str(idx), values=(
                result.get('provider', 'Source'),
                result.get('name', ''),
                result.get('provider_id', result.get('nist_id', '')),
            ))
        self._set_status(f'{len(self.nist_results)} result(s)')

    def _on_nist_select(self, event=None):
        sel = self.nist_tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        self.selected_result = self.nist_results[idx]
        detail = self.selected_result.get('detail')
        if detail:
            self._show_detail_payload(detail)
        else:
            self._write_details(
                f"{self.selected_result.get('name', '')}\n"
                f"Source: {self.selected_result.get('provider', 'Government source')}\n"
                f"ID: {self.selected_result.get('provider_id', self.selected_result.get('nist_id', ''))}\n\n"
                "Fetch details to inspect usable simulation properties.",
                source_url=self.selected_result.get('url', ''),
            )

    def _fetch_selected_details(self):
        if not self.selected_result:
            return None
        self._set_status('Fetching...')
        try:
            detail = fetch_government_gas_detail(self.selected_result, fallback_query=self.nist_query.get().strip())
        except (urlerror.URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError) as exc:
            self._set_status('Fetch failed')
            messagebox.showwarning('Gas Fetch', f'Could not fetch gas details:\n{exc}', parent=self.dlg)
            return None
        if not detail:
            self._set_status('No molecular weight')
            messagebox.showinfo(
                'Gas Fetch',
                'No checked government source returned enough molecular-weight data for simulation.',
                parent=self.dlg,
            )
            return None
        self.selected_result['detail'] = detail
        self._show_detail_payload(detail)
        self._set_status('Details ready')
        return detail

    def _import_selected_nist(self):
        detail = self._fetch_selected_details()
        if not detail:
            return
        key, gas, transport = build_gas_from_nist_detail(detail)
        key = _register_simulation_gas(key, gas, transport=transport, custom=True)
        if key not in CUSTOM_GAS_STATE['palette']:
            CUSTOM_GAS_STATE['palette'].append(key)
        save_custom_gases()
        self._refresh_known_tree(select_key=key)
        self._show_gas_details(key)
        notify_gas_palette_changed()
        self._set_status(f'Imported {key}')

    def _clean_palette(self):
        seen = set()
        cleaned = []
        for key in CUSTOM_GAS_STATE.get('palette', []):
            if key in GAS_DATA and key not in seen:
                cleaned.append(key)
                seen.add(key)
        if not cleaned:
            cleaned = ['N2'] if 'N2' in GAS_DATA else [next(iter(GAS_DATA))]
        CUSTOM_GAS_STATE['palette'] = cleaned
        return cleaned

    def _filtered_gas_keys(self, keys, filt):
        if not filt:
            return list(keys)
        matches = []
        for key in keys:
            gas = GAS_DATA[key]
            hay = f"{key} {gas.get('name', '')} {gas.get('symbol', '')} {gas.get('formula', '')}".lower()
            if filt in hay:
                matches.append(key)
        return matches

    def _refresh_known_tree(self, select_key=None):
        filt = self._filter_text.get().strip().lower()
        palette = self._clean_palette()
        selected = set(palette)
        for item in self.known_tree.get_children():
            self.known_tree.delete(item)

        ordered_keys = self._filtered_gas_keys(palette, filt)
        available_keys = sorted(
            self._filtered_gas_keys([k for k in GAS_DATA if k not in selected], filt),
            key=lambda k: (k not in DEFAULT_GAS_KEYS, GAS_DATA[k].get('name', k)),
        )

        for key in ordered_keys + available_keys:
            gas = GAS_DATA[key]
            src = _gas_source_label(gas)
            order = str(palette.index(key) + 1) if key in selected else ''
            self.known_tree.insert('', 'end', iid=key, values=(
                order,
                key,
                gas.get('name', key),
                f"{float(gas.get('m', 0.0)):.4g}",
                src,
            ))
        if select_key and self.known_tree.exists(select_key):
            self.known_tree.selection_set(select_key)
            self.known_tree.see(select_key)

    def _on_known_select(self, event=None):
        sel = self.known_tree.selection()
        if sel:
            self._show_gas_details(sel[0])

    def _selected_known_key(self):
        sel = self.known_tree.selection()
        return sel[0] if sel else None

    def _mark_palette_dirty(self, message='Palette changed'):
        self.footer_status.config(text=message)

    def _add_selected_known(self):
        key = self._selected_known_key()
        if not key:
            return
        palette = self._clean_palette()
        if key not in palette:
            palette.append(key)
            CUSTOM_GAS_STATE['palette'] = palette
            self._mark_palette_dirty(f'Added {key}')
        self._refresh_known_tree(select_key=key)
        self._show_gas_details(key)

    def _remove_selected_known(self):
        key = self._selected_known_key()
        if not key:
            return
        palette = self._clean_palette()
        if key not in palette:
            return
        if len(palette) <= 1:
            messagebox.showinfo('Custom Palette', 'Keep at least one gas in the custom palette.', parent=self.dlg)
            return
        palette.remove(key)
        CUSTOM_GAS_STATE['palette'] = palette
        self._mark_palette_dirty(f'Removed {key}')
        self._refresh_known_tree(select_key=key)
        self._show_gas_details(key)

    def _move_selected_palette(self, direction):
        key = self._selected_known_key()
        if not key:
            return
        palette = self._clean_palette()
        if key not in palette:
            messagebox.showinfo('Custom Palette', 'Add this gas before moving it in the custom palette.', parent=self.dlg)
            return
        idx = palette.index(key)
        new_idx = idx + int(direction)
        if new_idx < 0 or new_idx >= len(palette):
            return
        palette[idx], palette[new_idx] = palette[new_idx], palette[idx]
        CUSTOM_GAS_STATE['palette'] = palette
        self._mark_palette_dirty(f'Moved {key}')
        self._refresh_known_tree(select_key=key)
        self._show_gas_details(key)

    def _toggle_selected_known(self):
        key = self._selected_known_key()
        if not key:
            return
        palette = self._clean_palette()
        if key in palette:
            self._remove_selected_known()
        else:
            self._add_selected_known()

    def _reset_palette(self):
        CUSTOM_GAS_STATE['palette'] = list(DEFAULT_GAS_KEYS)
        self._mark_palette_dirty('Restored default palette order')
        self._refresh_known_tree(select_key='N2')
        self._show_gas_details('N2')

    def _set_source_url(self, source_url):
        self.current_source_url = source_url or ''
        if self.current_source_url:
            self.source_button.configure(state='normal')
            self.source_hint.config(text='Source link ready')
        else:
            self.source_button.configure(state='disabled')
            self.source_hint.config(text='No source selected')

    def _open_current_source(self):
        if self.current_source_url:
            webbrowser.open(self.current_source_url)

    def _write_details(self, text, source_url=''):
        self.detail_text.configure(state='normal')
        self.detail_text.delete('1.0', 'end')
        self.detail_text.insert('1.0', text)
        self.detail_text.configure(state='disabled')
        self._set_source_url(source_url)

    def _show_detail_payload(self, detail):
        formula = detail.get('formula') or '(not listed)'
        source = detail.get('source', 'Government source')
        provider_id = detail.get('provider_id') or detail.get('nist_id') or detail.get('pubchem_cid') or 'not listed'
        source_url = detail.get('source_url', '')
        text = (
            f"Name: {detail.get('name', '')}\n"
            f"Formula: {formula}\n"
            f"Molecular weight: {detail.get('m', 0):.6g} amu\n"
            f"CAS: {detail.get('cas', '') or 'not listed'}\n"
            f"Source ID: {provider_id}\n"
            f"Source: {source}\n"
            f"URL: {source_url}\n\n"
            "On import, the simulator uses source formula/molecular weight directly and estimates the Pirani-only fields that the source does not publish."
        )
        self._write_details(text, source_url=source_url)

    def _show_gas_details(self, key):
        gas = GAS_DATA.get(key)
        if not gas:
            return
        transport = GAS_TRANSPORT.get(key, _estimate_transport(key))
        text = (
            f"Key: {key}\n"
            f"Name: {gas.get('name', key)}\n"
            f"Symbol/formula: {gas.get('symbol', key)} / {gas.get('formula', gas.get('symbol', key))}\n"
            f"Molecular mass: {float(gas.get('m', 0.0)):.6g} amu\n"
            f"Degrees of freedom: {gas.get('f')}\n"
            f"gamma Cp/Cv: {gas.get('gamma')}\n"
            f"Mean thermal speed: {gas.get('cbar')} m/s\n"
            f"p*lambda product: {float(gas.get('plbar', 0.0)):.4g} m*mbar\n"
            f"Transport mu0/k0: {transport.get('mu0', 0.0):.4g} Pa*s / {transport.get('k0', 0.0):.4g} W/m/K\n"
            f"In custom palette: {'Yes' if key in CUSTOM_GAS_STATE.get('palette', []) else 'No'}\n\n"
            f"Sources:\n{_gas_sources_text(gas)}"
        )
        self._write_details(text, source_url=gas.get('source_url', ''))

    def _save(self):
        self._clean_palette()
        save_custom_gases()
        notify_gas_palette_changed()
        self._set_status('Saved')
        self.footer_status.config(text='Palette saved')

    def _close(self):
        self._save()
        if callable(self.on_close):
            self.on_close()
        self.dlg.destroy()


def open_gas_palette_dialog(parent, on_close=None):
    return GasPaletteDialog(parent, on_close=on_close)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB: HEAT TRANSFER 2D SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════

class Simulator2DTab(ttk.Frame):
    PRESSURE_POINTS = 240

    def __init__(self, parent):
        super().__init__(parent)
        self.gas_vars = {}
        self.gas_checks = {}
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
        self.gas_frame = ttk.LabelFrame(ctrl_frame, text=' Gas Species ')
        self.gas_frame.pack(fill='x', pady=(0, 8))

        palette_row = ttk.Frame(self.gas_frame, style='Card.TFrame')
        palette_row.pack(fill='x', padx=8, pady=(6, 2))
        ttk.Label(palette_row, text='Palette', style='Card.TLabel').pack(side='left')
        self.palette_mode_var = tk.StringVar(value='default')
        palette_combo = ttk.Combobox(palette_row, textvariable=self.palette_mode_var,
                                     values=['default', 'custom'], state='readonly', width=9)
        palette_combo.pack(side='left', padx=(6, 4))
        palette_combo.bind('<<ComboboxSelected>>', lambda e: self._rebuild_gas_checkbuttons())
        ttk.Button(palette_row, text='Edit Custom',
                   command=lambda: open_gas_palette_dialog(self, self.on_gas_palette_changed)).pack(side='right')

        self.gas_list_frame = ttk.Frame(self.gas_frame, style='Card.TFrame')
        self.gas_list_frame.pack(fill='x', padx=4, pady=(2, 6))
        self._rebuild_gas_checkbuttons(preserve=False)

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

    def _rebuild_gas_checkbuttons(self, preserve=True):
        previous = {k: v.get() for k, v in self.gas_vars.items()} if preserve else {}
        for child in self.gas_list_frame.winfo_children():
            child.destroy()
        self.gas_vars = {}
        self.gas_checks = {}
        keys = get_gas_palette_keys(self.palette_mode_var.get())
        default_selected = {'N2', 'He', 'Ar', 'Xe'}
        for key in keys:
            gas = GAS_DATA[key]
            selected = previous.get(key, key in default_selected)
            var = tk.BooleanVar(value=selected)
            cb = ttk.Checkbutton(self.gas_list_frame,
                                 text=f"{gas.get('symbol', key)} ({gas.get('name', key)})",
                                 variable=var, command=self._schedule_update,
                                 style='TCheckbutton')
            cb.pack(anchor='w', padx=8, pady=1)
            self.gas_vars[key] = var
            self.gas_checks[key] = cb
        if keys and not any(v.get() for v in self.gas_vars.values()):
            self.gas_vars['N2' if 'N2' in self.gas_vars else keys[0]].set(True)
        if hasattr(self, 'ax'):
            self._schedule_update()

    def on_gas_palette_changed(self):
        self._rebuild_gas_checkbuttons(preserve=True)

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
            pressures = np.logspace(-2, 5, self.PRESSURE_POINTS)

            unit = get_pressure_unit()
            uf = PRESSURE_UNITS[unit]['factor']
            p_display = pressures * uf

            selected = [k for k, v in self.gas_vars.items() if v.get() and k in GAS_DATA]
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
                regime_txt = '—— combined  - - molecular  ···· viscous(+gravity/conv)'
                self.ax.text(0.98, 0.02, regime_txt,
                             transform=self.ax.transAxes, ha='right', fontsize=8,
                             color=COLORS['text_dim'])

            self.fig.tight_layout()
            self.canvas.draw_idle()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB: 3D SURFACE SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════

class Simulator3DTab(ttk.Frame):
    SURFACE_GRID = 30
    GAS_COMPARE_POINTS = 56
    MOL_VISC_POINTS = 64

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
        self.gas_frame = ttk.LabelFrame(ctrl_frame, text=' Primary Gas ')
        self.gas_frame.pack(fill='x', pady=(0, 8))

        palette_row = ttk.Frame(self.gas_frame, style='Card.TFrame')
        palette_row.pack(fill='x', padx=8, pady=(6, 2))
        ttk.Label(palette_row, text='Palette', style='Card.TLabel').pack(side='left')
        self.palette_mode_var = tk.StringVar(value='default')
        palette_combo = ttk.Combobox(palette_row, textvariable=self.palette_mode_var,
                                     values=['default', 'custom'], state='readonly', width=9)
        palette_combo.pack(side='left', padx=(6, 4))
        palette_combo.bind('<<ComboboxSelected>>', lambda e: self._rebuild_gas_radios())
        ttk.Button(palette_row, text='Edit Custom',
                   command=lambda: open_gas_palette_dialog(self, self.on_gas_palette_changed)).pack(side='right')

        self.gas_radio_frame = ttk.Frame(self.gas_frame, style='Card.TFrame')
        self.gas_radio_frame.pack(fill='x', padx=4, pady=(2, 6))
        self.gas_3d_var = tk.StringVar(value='Ar')
        self._rebuild_gas_radios(preserve=False)

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

    def _rebuild_gas_radios(self, preserve=True):
        current = self.gas_3d_var.get() if preserve else 'Ar'
        for child in self.gas_radio_frame.winfo_children():
            child.destroy()
        keys = get_gas_palette_keys(self.palette_mode_var.get())
        if current not in keys:
            current = 'Ar' if 'Ar' in keys else ('N2' if 'N2' in keys else keys[0])
        self.gas_3d_var.set(current)
        for key in keys:
            gas = GAS_DATA[key]
            ttk.Radiobutton(self.gas_radio_frame, text=f"{gas.get('symbol', key)}",
                            variable=self.gas_3d_var, value=key,
                            command=self._schedule_update,
                            style='TCheckbutton').pack(anchor='w', padx=8, pady=1)
        if hasattr(self, 'ax'):
            self._schedule_update()

    def on_gas_palette_changed(self):
        self._rebuild_gas_radios(preserve=True)

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
            if gas_key not in GAS_DATA:
                gas_key = 'N2'
                self.gas_3d_var.set(gas_key)
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
        P = np.logspace(-1, 4, self.SURFACE_GRID)
        A = np.linspace(0.1, 1.0, self.SURFACE_GRID)
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
        P = np.logspace(-1, 4, self.SURFACE_GRID)
        T = np.linspace(313, 573, self.SURFACE_GRID)
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
        P = np.logspace(-1, 4, self.SURFACE_GRID)
        if cfg['geometry'] in ('cylindrical', 'square_cavity'):
            G = np.linspace(2, 30, self.SURFACE_GRID)  # enclosure radius in mm
        else:
            G = np.linspace(0.002, 5, self.SURFACE_GRID)  # gap in mm
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
        gases = get_gas_palette_keys(self.palette_mode_var.get())
        P = np.logspace(-1, 5.2, self.GAS_COMPARE_POINTS)
        Z = np.zeros((len(gases), len(P)))

        for i, gk in enumerate(gases):
            Z[i, :] = calc_correction_factor_curve_physics(gk, cfg, P, 0.6)

        for i, gk in enumerate(gases):
            self.ax.plot(np.log10(P), [i]*len(P), Z[i, :],
                         color=GAS_DATA[gk]['color'], linewidth=2.5, label=GAS_DATA[gk]['symbol'])

        self.ax.set_xlabel('log₁₀(P / Pa)', fontsize=9, labelpad=10)
        self.ax.set_ylabel('Gas Species', fontsize=9, labelpad=10)
        self.ax.set_zlabel('CF_X/N₂', fontsize=9, labelpad=10)
        self.ax.set_yticks(range(len(gases)))
        self.ax.set_yticklabels([GAS_DATA[g]['symbol'] for g in gases], fontsize=7)
        self.ax.set_title(f'Pressure-Dependent Correction Factors — {cfg.get("name", "Gauge")}',
                          fontsize=11, color=COLORS['text_bright'], pad=15)
        self.ax.legend(fontsize=7, loc='upper left')

    def _plot_mol_vs_visc(self, cfg, gas_key, cmap_name, wireframe):
        P = np.logspace(-1, 5, self.MOL_VISC_POINTS)
        gas = GAS_DATA[gas_key]
        Q_arr, Qm_arr, Qv_arr = calc_heat_flow_vec(gas_key, cfg, P, 0.6)
        Qs = Q_arr * 1000
        Qms = Qm_arr * 1000
        Qvs = Qv_arr * 1000

        logP = np.log10(P)
        zeros = np.zeros_like(logP)

        self.ax.plot(logP, zeros, Qms, color='#4fc3f7', linewidth=2.5, label='Molecular (Q_mol)')
        self.ax.plot(logP, np.ones_like(logP), Qvs, color='#e67e22', linewidth=2.5, label='Viscous (Q_visc)')
        self.ax.plot(logP, np.ones_like(logP)*2, Qs, color='#6bcb77', linewidth=3, label='Combined')

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
        table_frame = ttk.LabelFrame(ctrl_frame, text=' Gas Properties ')
        table_frame.pack(fill='both', expand=True, pady=(0, 8))

        cols = ('gas', 'm', 'f', 'gamma', 'cbar', 'plbar', 'source')
        self.tree = ttk.Treeview(table_frame, columns=cols, show='headings', height=9)
        headers = {'gas': 'Gas', 'm': 'Mass', 'f': 'DOF', 'gamma': 'γ', 'cbar': 'c̄', 'plbar': 'p·λ̄', 'source': 'Source'}
        for c in cols:
            self.tree.heading(c, text=headers[c])
            self.tree.column(c, width=90 if c == 'source' else 55)
        self.tree.pack(padx=4, pady=4, fill='both', expand=True)

        self._refresh_table()

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

    def on_gas_palette_changed(self):
        self._refresh_table()
        self._update_plot()

    def _refresh_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for key, g in GAS_DATA.items():
            src = _gas_source_label(g)
            self.tree.insert('', 'end', values=(
                g.get('symbol', key), g.get('m'), g.get('f'), g.get('gamma'),
                g.get('cbar'), f"{float(g.get('plbar', 0.0))*1000:.1f}", src))

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
        self.gas_combo = ttk.Combobox(row1, textvariable=self.calc_gas,
                          values=[_gas_combo_label(k) for k in get_all_simulation_gas_keys()],
                          state='readonly', width=20)
        self.gas_combo.pack(side='right')
        self.gas_combo.bind('<<ComboboxSelected>>', self._calc_correction)

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
                self.ht_gas_combo = ttk.Combobox(row, textvariable=var,
                                                 values=get_all_simulation_gas_keys(), state='readonly', width=10)
                self.ht_gas_combo.pack(side='right')
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
        self._refresh_gas_options()
        self._sync_unit_labels()

    def on_gas_palette_changed(self):
        self._refresh_gas_options()

    def _refresh_gas_options(self):
        keys = get_all_simulation_gas_keys()
        self.gas_combo.configure(values=[_gas_combo_label(k) for k in keys])
        if _gas_key_from_combo(self.calc_gas.get(), fallback='Ar') not in GAS_DATA:
            self.calc_gas.set(_gas_combo_label('Ar' if 'Ar' in GAS_DATA else keys[0]))
        elif self.calc_gas.get() in GAS_DATA:
            self.calc_gas.set(_gas_combo_label(self.calc_gas.get()))
        self.ht_gas_combo.configure(values=keys)
        if self.ht_vars['ht_gas'].get() not in GAS_DATA:
            self.ht_vars['ht_gas'].set('N2' if 'N2' in GAS_DATA else keys[0])

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
        return _gas_key_from_combo(self.calc_gas.get(), fallback='N2')

    def _calc_correction(self, *args):
        self._sync_unit_labels()
        try:
            key = self._get_gas_key()
            p_ind_user = float(self.calc_pind.get())
            unit = get_pressure_unit()
            ulbl = PRESSURE_UNITS[unit]['label']
            cf_data = EXPERIMENTAL_CF.get(key, {'mean': calc_correction_factor_theory(key), 'spread': 0.0})
            cf = cf_data['mean']
            spread = cf_data['spread']
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
             "• Q̇_conv — Buoyancy/natural convection from gravity, gas transport, and geometry at higher pressure\n\n"
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
             "• Mean velocity (c̄) — lighter = faster (H₂: 1764 m/s vs Xe: 219 m/s)\n"
             "• High-pressure convection behavior — viscosity, thermal conductivity, density, gravity, and gauge orientation\n\n"
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
            self.canvas.draw_idle()

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
    DRAW_EVERY = 5         # draw every N physics steps

    # ── Molecule visuals ─────────────────────────────────────────────────────
    MOL_SPEED_BASE = 4.0   # display speed for N₂ (units / s)
    MOL_SIZE_BASE = 46     # scatter marker size for N₂
    MOLECULES_PER_PARTICLE = 3.0e16  # real molecules represented by one visual particle
    TRAIL_LENGTH = 8       # stored positions per particle for trajectory trails
    MAX_VISUAL_MOLECULES = 260
    MAX_TRAIL_PARTICLES = 90
    MAX_VECTOR_PARTICLES = 90
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
        self.mol_gas_key_array = np.array([], dtype=object)
        self.mol_group_indices = {}
        # Precomputed per-molecule physics arrays (set in _init_molecules)
        self.mol_aE = np.zeros(0)
        self.mol_f_plus1 = np.zeros(0)
        self.mol_thermal_speed = np.zeros(0)  # Rayleigh scale at T_COLD
        self.mol_conv_response = np.zeros(0)
        self.molecules_per_particle = self.MOLECULES_PER_PARTICLE
        self.mol_trail_pos = np.zeros((0, self.TRAIL_LENGTH, 3))
        self.mol_trail_temp = np.zeros((0, self.TRAIL_LENGTH))
        self.pirani_readings_pa = deque(maxlen=self.PIRANI_AVG_WINDOW)
        self._bridge_calibration_cache = {}

        self._elev = 20
        self._azim = -60
        self._camera_preset_pending = False  # flag to skip reading axes on next draw
        self._pending_reinit_job = None
        self._pending_reinit_draw = False
        self._syncing_mix_fields = False
        self._last_edited_gas_key = None
        self.auto_normalize_var = tk.BooleanVar(value=True)

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

        palette_row = ttk.Frame(mix_frame, style='Card.TFrame')
        palette_row.pack(fill='x', padx=8, pady=(6, 2))
        ttk.Label(palette_row, text='Palette', style='Card.TLabel').pack(side='left')
        self.palette_mode_var = tk.StringVar(value='default')
        palette_combo = ttk.Combobox(palette_row, textvariable=self.palette_mode_var,
                         values=['default', 'custom'], state='readonly', width=9)
        palette_combo.pack(side='left', padx=(6, 4))
        palette_combo.bind('<<ComboboxSelected>>', self._on_palette_mode_change)
        ttk.Button(palette_row, text='Edit Custom',
               command=lambda: open_gas_palette_dialog(self, self.on_gas_palette_changed)).pack(side='right')

        # Preset combobox
        preset_row = ttk.Frame(mix_frame, style='Card.TFrame')
        preset_row.pack(fill='x', padx=8, pady=(2, 2))
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
        self.gas_grid = ttk.Frame(mix_frame, style='Card.TFrame')
        self.gas_grid.pack(fill='x', padx=6, pady=(0, 4))

        # Total & normalize
        total_row = ttk.Frame(mix_frame, style='Card.TFrame')
        total_row.pack(fill='x', padx=8, pady=(2, 6))
        self.total_label = ttk.Label(total_row, text='Total: 0 %',
                                     style='Accent.TLabel')
        self.total_label.pack(side='left')
        self.normalize_btn = ttk.Button(total_row, text='Normalize: On',
                        command=self._toggle_auto_normalize)
        self.normalize_btn.pack(side='right')
        self._update_normalize_button()

        # Composition bar (tiny matplotlib)
        self.comp_fig = Figure(figsize=(2.6, 0.3), dpi=100)
        self.comp_fig.patch.set_facecolor(COLORS['bg_card'])
        self.comp_canvas = FigureCanvasTkAgg(self.comp_fig, master=mix_frame)
        self.comp_canvas.get_tk_widget().pack(fill='x', padx=6, pady=(0, 6))
        self._rebuild_mixture_gas_rows(preserve=False)

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
        self.show_center_of_mass = tk.BooleanVar(value=False)
        ttk.Checkbutton(vis_frame, text='Show gas centers of mass',
            variable=self.show_center_of_mass,
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

        self.sl_color_min = LabeledSlider(color_frame, 'Color Min Temp (K)', 200, 500, 390,
                                          fmt='{:.0f}', unit='K',
                                          command=lambda v: self._on_color_range_change())
        self.sl_color_min.pack(fill='x', padx=6, pady=2)

        self.sl_color_max = LabeledSlider(color_frame, 'Color Max Temp (K)', 300, 700, 396,
                                          fmt='{:.0f}', unit='K',
                                          command=lambda v: self._on_color_range_change())
        self.sl_color_max.pack(fill='x', padx=6, pady=2)

        self.auto_color_scale_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            color_frame,
            text='Auto-scale to filament ±3 °C',
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

        data_frame = ttk.LabelFrame(info, text=' Data ')
        data_frame.pack(fill='x', pady=(0, 6))
        ttk.Button(data_frame, text='Export CSV',
               command=self._export_pressure_csv).pack(fill='x', padx=8, pady=6)

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
            color_min_k = self.T_HOT - 3.0
            color_max_k = self.T_HOT + 3.0
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

    def _get_molecule_radius_pm(self, gas_key):
        return self.MOL_RADII.get(gas_key, _estimate_molecular_radius_pm(gas_key))

    def _mol_count(self):
        """Number of visual molecules, proportional to pressure."""
        p_pa = 10 ** self.sl_pressure.get()
        p_mbar = convert_pressure(p_pa, 'mbar')
        lo_log = -6.0
        hi_log = np.log10(2000.0)
        frac = (np.log10(max(p_mbar, 1e-12)) - lo_log) / (hi_log - lo_log)
        frac = float(np.clip(frac, 0.0, 1.0))
        n_min, n_max = 6, self.MAX_VISUAL_MOLECULES
        shaped = frac ** 1.15
        return int(round(n_min + shaped * (n_max - n_min)))

    def _get_accommodation_for(self, gas_key):
        """Effective accommodation coefficient αE for a specific gas."""
        surface = GAUGE_CONFIGS[self.cfg_var.get()]['surface']
        table = ACCOM_RATIOS_W if surface == 'W' else ACCOM_RATIOS_Si
        return min(0.6 * table.get(gas_key, 1.0), 1.0)

    def _get_visual_convection_response(self, gas_key):
        """Gas-specific response used by the visual buoyancy/convection animation."""
        t_film = max(0.5 * (self.T_HOT + self.T_COLD), 180.0)
        mu_g, k_g, cp_g, _ = _get_gas_transport(gas_key, t_film)
        mu_ref, k_ref, cp_ref, _ = _get_gas_transport('N2', t_film)
        m_ref = GAS_DATA['N2']['m'] / 1000.0
        m_g = GAS_DATA[gas_key]['m'] / 1000.0
        response = (
            (k_g / max(k_ref, 1e-18)) ** 0.35 *
            (mu_ref / max(mu_g, 1e-18)) ** 0.20 *
            (cp_g / max(cp_ref, 1e-18)) ** 0.10 *
            (m_ref / max(m_g, 1e-18)) ** 0.08
        )
        return float(np.clip(response, 0.45, 1.80))

    # ── Mixture management ───────────────────────────────────────────────────

    def _active_mixture_keys(self):
        return [k for k in get_gas_palette_keys(self.palette_mode_var.get()) if k in GAS_DATA]

    def _rebuild_mixture_gas_rows(self, preserve=True):
        previous = {}
        if preserve:
            for key, var in self.gas_pct_vars.items():
                try:
                    previous[key] = max(float(var.get()), 0.0)
                except (ValueError, tk.TclError):
                    previous[key] = 0.0

        for child in self.gas_grid.winfo_children():
            child.destroy()
        self.gas_pct_vars = {}
        self.gas_pct_entries = {}
        self._gas_dot_canvases = []

        for key in self._active_mixture_keys():
            gas = GAS_DATA[key]
            row = ttk.Frame(self.gas_grid, style='Card.TFrame')
            row.pack(fill='x', pady=1)

            dot = tk.Canvas(row, width=10, height=10, bg=COLORS['bg_card'], highlightthickness=0)
            dot.create_oval(1, 1, 9, 9, fill=gas.get('color', COLORS['accent']), outline='')
            dot.pack(side='left', padx=(4, 4))
            self._gas_dot_canvases.append(dot)

            ttk.Label(row, text=f"{gas.get('symbol', key)}", style='Card.TLabel', width=6).pack(side='left')

            var = tk.StringVar(value=f"{previous.get(key, 0.0):.1f}" if previous.get(key, 0.0) else '0')
            entry = ttk.Entry(row, textvariable=var, width=6, justify='right')
            entry.pack(side='right', padx=(0, 4))
            entry.bind('<KeyRelease>', lambda e, k=key: self._on_ratio_change(e, changed_key=k))
            entry.bind('<FocusOut>', lambda e, k=key: self._on_ratio_change(e, changed_key=k))
            ttk.Label(row, text='%', style='Dim.TLabel').pack(side='right')

            self.gas_pct_vars[key] = var
            self.gas_pct_entries[key] = entry

        if self.gas_pct_vars:
            total = sum(max(float(v.get()), 0.0) for v in self.gas_pct_vars.values() if self._is_float(v.get()))
            if total <= 0.0:
                key = 'N2' if 'N2' in self.gas_pct_vars else next(iter(self.gas_pct_vars))
                self.gas_pct_vars[key].set('100.0')
            self._force_mixture_total()
        self._update_composition_display()

    def _is_float(self, value):
        try:
            float(value)
            return True
        except (ValueError, TypeError, tk.TclError):
            return False

    def _set_pct_var(self, key, value):
        if key in self.gas_pct_vars:
            self.gas_pct_vars[key].set(f'{max(value, 0.0):.1f}' if value > 0 else '0')

    def _auto_normalize_enabled(self):
        var = getattr(self, 'auto_normalize_var', None)
        return True if var is None else bool(var.get())

    def _update_normalize_button(self):
        if not hasattr(self, 'normalize_btn'):
            return
        if self._auto_normalize_enabled():
            self.normalize_btn.config(text='Normalize: On', style='Active.TButton')
        else:
            self.normalize_btn.config(text='Normalize: Off', style='TButton')

    def _toggle_auto_normalize(self):
        self.auto_normalize_var.set(not self.auto_normalize_var.get())
        if self._auto_normalize_enabled():
            self._normalize_ratios()
        else:
            self._update_composition_display()
        self._update_normalize_button()
        self._schedule_molecule_reinit(draw_if_idle=True, delay_ms=60)

    def _force_mixture_total(self, changed_key=None, force=False):
        if self._syncing_mix_fields or not self.gas_pct_vars:
            return
        if not force and not self._auto_normalize_enabled():
            return
        self._syncing_mix_fields = True
        try:
            keys = list(self.gas_pct_vars.keys())
            values = {}
            for key in keys:
                try:
                    values[key] = max(float(self.gas_pct_vars[key].get()), 0.0)
                except (ValueError, tk.TclError):
                    values[key] = 0.0

            if changed_key not in values:
                changed_key = self._last_edited_gas_key if self._last_edited_gas_key in values else None

            if changed_key is not None:
                changed_val = min(values[changed_key], 100.0)
                values[changed_key] = changed_val
                others = [k for k in keys if k != changed_key]
                remainder = max(100.0 - changed_val, 0.0)
                other_sum = sum(values[k] for k in others)
                if others:
                    if other_sum > 1e-9:
                        for key in others:
                            values[key] = values[key] / other_sum * remainder
                    else:
                        share = remainder / len(others)
                        for key in others:
                            values[key] = share
                else:
                    values[changed_key] = 100.0
            else:
                total = sum(values.values())
                if total <= 1e-9:
                    first = 'N2' if 'N2' in values else keys[0]
                    values = {k: (100.0 if k == first else 0.0) for k in keys}
                else:
                    values = {k: v / total * 100.0 for k, v in values.items()}

            rounded = {k: round(v, 1) for k, v in values.items()}
            diff = round(100.0 - sum(rounded.values()), 1)
            if abs(diff) >= 0.1:
                target = changed_key if changed_key in rounded else max(rounded, key=rounded.get)
                rounded[target] = max(0.0, rounded[target] + diff)
            for key, value in rounded.items():
                self._set_pct_var(key, value)
        finally:
            self._syncing_mix_fields = False

    def _on_palette_mode_change(self, event=None):
        self._rebuild_mixture_gas_rows(preserve=True)
        self.preset_var.set('custom')
        self.preset_desc.config(text=GAS_MIXTURE_PRESETS['custom']['desc'])
        self._schedule_molecule_reinit(draw_if_idle=True, delay_ms=60)

    def on_gas_palette_changed(self):
        self._rebuild_mixture_gas_rows(preserve=True)
        self._schedule_molecule_reinit(draw_if_idle=True, delay_ms=60)

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
        for key in self.gas_pct_vars:
            self.gas_pct_vars[key].set(str(mix.get(key, 0)))
        self._force_mixture_total()
        self._update_composition_display()

    def _on_preset_change(self, event=None):
        key = self.preset_var.get()
        self._apply_preset(key)
        self._schedule_molecule_reinit(draw_if_idle=True, delay_ms=30)

    def _on_ratio_change(self, event=None, changed_key=None):
        if self._syncing_mix_fields:
            return
        self._last_edited_gas_key = changed_key
        self._force_mixture_total(changed_key=changed_key)
        self._update_composition_display()
        # Switch preset label to Custom if user edits
        self.preset_var.set('custom')
        self.preset_desc.config(text=GAS_MIXTURE_PRESETS['custom']['desc'])
        self._schedule_molecule_reinit(draw_if_idle=True, delay_ms=150)

    def _normalize_ratios(self):
        """Scale entries so they sum to 100 %."""
        fracs = self._get_mixture_fractions()
        for key in self.gas_pct_vars:
            pct = fracs.get(key, 0.0) * 100
            self.gas_pct_vars[key].set(f'{pct:.1f}' if pct > 0 else '0')
        self._force_mixture_total(force=True)
        self._update_composition_display()

    def _update_composition_display(self):
        """Refresh the total label and the stacked composition bar."""
        total = 0.0
        for var in self.gas_pct_vars.values():
            try:
                total += float(var.get())
            except (ValueError, tk.TclError):
                pass
        c = COLORS['accent2'] if abs(total - 100) < 0.05 else COLORS['warn']
        self.total_label.config(text=f'Total: {total:.1f} %', foreground=c)

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

    def _get_raw_mixture_percentages(self):
        raw = {}
        for key, var in self.gas_pct_vars.items():
            try:
                value = max(float(var.get()), 0.0)
            except (ValueError, tk.TclError):
                value = 0.0
            if value > 0.0:
                raw[key] = value
        return raw

    def _format_mixture_for_export(self, values):
        if not values:
            return 'N2 100%'
        parts = []
        for key, value in sorted(values.items(), key=lambda item: -item[1]):
            gas = GAS_DATA.get(key, {})
            label = gas.get('symbol', key)
            parts.append(f'{label} {value:.4g}%')
        return '; '.join(parts)

    def _export_pressure_csv(self):
        cfg = self._get_active_sim_config()
        cfg_name = cfg.get('name', self.cfg_var.get())
        initial_name = re.sub(r'[^A-Za-z0-9_.-]+', '_', f'pirani_{self.cfg_var.get()}_export').strip('_')
        path = filedialog.asksaveasfilename(
            parent=self.winfo_toplevel(),
            title='Export Simulation CSV',
            defaultextension='.csv',
            initialfile=f'{initial_name}.csv',
            filetypes=[('CSV files', '*.csv'), ('All files', '*.*')],
        )
        if not path:
            return

        try:
            rows = self._build_pressure_export_rows(cfg)
            with open(path, 'w', newline='', encoding='utf-8') as fh:
                writer = csv.writer(fh)
                writer.writerows(rows)
            messagebox.showinfo('Export CSV', f'Exported {cfg_name} simulation data to:\n{path}',
                                parent=self.winfo_toplevel())
        except Exception as exc:
            messagebox.showwarning('Export CSV', f'Could not export simulation data:\n{exc}',
                                   parent=self.winfo_toplevel())

    def _build_pressure_export_rows(self, cfg):
        unit = get_pressure_unit()
        unit_label = PRESSURE_UNITS[unit]['label']
        unit_factor = PRESSURE_UNITS[unit]['factor']
        t_unit = get_temperature_unit()
        raw_mix = self._get_raw_mixture_percentages()
        raw_total = sum(raw_mix.values())
        fracs = self._get_mixture_fractions()
        mix_pct_used = {key: value * 100.0 for key, value in fracs.items()}
        aN2 = self._get_nominal_aN2(cfg)

        range_lo_mbar, range_hi_mbar = cfg.get('range_mbar', (1e-4, 1000.0))
        p_min_pa = max(float(range_lo_mbar) * 100.0, 1e-12)
        p_max_pa = max(float(range_hi_mbar) * 100.0, p_min_pa * 10.0)
        sample_count = 160
        real_pressures_pa = np.logspace(np.log10(p_min_pa), np.log10(p_max_pa), sample_count)
        measured_pressures_pa = calc_mixture_indicated_pressure_curve_physics(
            fracs,
            cfg,
            real_pressures_pa,
            aN2=aN2,
        )

        rows = [
            ['Pirani Vacuum Gauge Simulator Export'],
            ['Exported at', time.strftime('%Y-%m-%d %H:%M:%S')],
            ['Gauge', cfg.get('name', self.cfg_var.get())],
            ['Geometry', cfg.get('geometry', '')],
            ['Surface', cfg.get('surface', '')],
            ['Pressure unit', unit_label],
            ['Temperature unit', TEMPERATURE_UNITS[t_unit]['label']],
            ['Auto normalize mixture', 'On' if self._auto_normalize_enabled() else 'Off'],
            ['Raw mixture total percent', f'{raw_total:.6g}'],
            ['Raw mixture entries', self._format_mixture_for_export(raw_mix)],
            ['Simulation mixture used', self._format_mixture_for_export(mix_pct_used)],
            ['Filament temperature K', f'{self.T_HOT:.12g}'],
            ['Environment temperature K', f'{self.T_COLD:.12g}'],
            ['Current pressure setpoint Pa', f'{10 ** self.sl_pressure.get():.12g}'],
            ['Gauge range mbar', f'{range_lo_mbar:.12g} to {range_hi_mbar:.12g}'],
            ['Rows sampled', str(sample_count)],
            ['Measured pressure model', 'N2-calibrated indicated pressure from current heat-flow model'],
            [],
            [f'real_pressure_{unit_label}', f'measured_pressure_{unit_label}',
             f'delta_pressure_{unit_label}', 'delta_percent'],
        ]

        for real_pa, measured_pa in zip(real_pressures_pa, measured_pressures_pa):
            delta_pa = measured_pa - real_pa
            delta_pct = (delta_pa / max(real_pa, 1e-20)) * 100.0
            rows.append([
                f'{real_pa * unit_factor:.12g}',
                f'{measured_pa * unit_factor:.12g}',
                f'{delta_pa * unit_factor:.12g}',
                f'{delta_pct:.12g}',
            ])
        return rows

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
        self.mol_gas_key_array = np.asarray(all_keys, dtype=object)

        # Precompute per-molecule accommodation & (f+1) for vectorised _step
        n = len(self.mol_gas_keys)
        self.mol_group_indices = {}
        if n > 0:
            for gas_key in dict.fromkeys(all_keys):
                self.mol_group_indices[gas_key] = np.flatnonzero(self.mol_gas_key_array == gas_key)
        self.mol_aE = np.empty(n, dtype=np.float64)
        self.mol_f_plus1 = np.empty(n, dtype=np.float64)
        for j, gk in enumerate(self.mol_gas_keys):
            self.mol_aE[j] = self._get_accommodation_for(gk)
            self.mol_f_plus1[j] = GAS_DATA[gk]['f'] + 1

        # Thermal speed scale at T_COLD for wall re-thermalization
        self.mol_thermal_speed = np.empty(n, dtype=np.float64)
        self.mol_conv_response = np.empty(n, dtype=np.float64)
        for j, gk in enumerate(self.mol_gas_keys):
            gas = GAS_DATA[gk]
            speed = self.MOL_SPEED_BASE * (gas['cbar'] / GAS_DATA['N2']['cbar'])
            self.mol_thermal_speed[j] = speed * 0.65 * math.sqrt(self.T_COLD / 296.0)
            self.mol_conv_response[j] = self._get_visual_convection_response(gk)

        # Gas starts in thermal equilibrium with the enclosure walls.
        self.gas_ambient_temp_k = self.T_COLD
        self.wall_coupling_ema = 0.0
        self._sync_molecule_scale_to_pressure_setpoint()

        if n > 0:
            temp_now = self._estimate_particle_temperatures_k()
            self.mol_trail_pos = np.repeat(self.mol_pos[:, np.newaxis, :], self.TRAIL_LENGTH, axis=1)
            self.mol_trail_temp = np.repeat(temp_now[:, np.newaxis], self.TRAIL_LENGTH, axis=1)
        else:
            self.mol_trail_pos = np.zeros((0, self.TRAIL_LENGTH, 3))
            self.mol_trail_temp = np.zeros((0, self.TRAIL_LENGTH))
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
        cfg = GAUGE_CONFIGS[self.cfg_var.get()]
        geometry = cfg['geometry']
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
        p_pa = 10 ** self.sl_pressure.get()
        conv_activation = float(_convection_pressure_activation(cfg, p_pa))
        gravity_scale = _effective_gravity(cfg) / max(G_STD, 1e-12)
        if conv_activation > 1e-6 and gravity_scale > 0.0:
            g_axis = 0 if self._is_horizontal_wire_mode() else 2
            t_mol = self._estimate_particle_temperatures_k()
            temp_drive = np.clip((t_mol - self.T_COLD) / max(self.T_HOT - self.T_COLD, 1.0), 0.0, 2.0)
            response = self.mol_conv_response[:n] if len(self.mol_conv_response) >= n else 1.0
            buoy = (conv_activation * self.BUOYANCY_ACCEL * gravity_scale *
                    _convection_geometry_factor(cfg) * temp_drive * response)
            vel[:, g_axis] += (self.GRAVITY_ACCEL * gravity_scale + buoy) * dt
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

    def _get_pressure_setpoint_pa(self):
        """Current pressure setpoint from the log-pressure slider."""
        return max(float(10 ** self.sl_pressure.get()), 1e-20)

    def _sync_molecule_scale_to_pressure_setpoint(self):
        """Keep represented molecule count consistent with the pressure setpoint."""
        n_visual = len(self.mol_pos)
        if n_visual <= 0:
            self.molecules_per_particle = self.MOLECULES_PER_PARTICLE
            return

        volume_m3 = self._get_defined_volume_m3()
        temperature_k = max(self._get_gas_ambient_temperature_k(), 1.0)
        target_pressure_pa = self._get_pressure_setpoint_pa()
        self.molecules_per_particle = max(
            (target_pressure_pa * volume_m3) / (kB * temperature_k * n_visual),
            1.0,
        )

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
        trail_indices = np.arange(n)
        if n > self.MAX_TRAIL_PARTICLES:
            stride = int(math.ceil(n / self.MAX_TRAIL_PARTICLES))
            trail_indices = trail_indices[::stride]

        for j in trail_indices:
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

        if len(self.mol_gas_key_array) >= len(self.mol_gas_keys):
            hit_keys, hit_counts = np.unique(self.mol_gas_key_array[mi], return_counts=True)
            for gk, count in zip(hit_keys, hit_counts):
                self.collision_per_gas[gk] = self.collision_per_gas.get(gk, 0) + int(count)

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

                for gk, idx in self.mol_group_indices.items():
                    gas = GAS_DATA[gk]
                    sz = self.MOL_SIZE_BASE * (self._get_molecule_radius_pm(gk) / self.MOL_RADII['N2']) ** 2
                    mx, my, mz = self._render_coords(self.mol_pos[idx, 0],
                                                     self.mol_pos[idx, 1],
                                                     self.mol_pos[idx, 2])
                    ax.scatter(mx, my, mz,
                               s=sz, c=gas['color'], alpha=0.78,
                               edgecolors='none', linewidths=0,
                               depthshade=False, zorder=5, label=gas['symbol'])

                if self.show_center_of_mass.get():
                    self._draw_center_of_mass_regions(ax)

                if self.show_vectors.get():
                    vector_indices = np.arange(len(self.mol_pos))
                    if len(vector_indices) > self.MAX_VECTOR_PARTICLES:
                        stride = int(math.ceil(len(vector_indices) / self.MAX_VECTOR_PARTICLES))
                        vector_indices = vector_indices[::stride]
                    segments = []
                    colors = []
                    for j in vector_indices:
                        gk = self.mol_gas_keys[j] if j < len(self.mol_gas_keys) else 'N2'
                        p = self.mol_pos[j]
                        v = self.mol_vel[j] * 0.25
                        p0x, p0y, p0z = self._render_coords(p[0], p[1], p[2])
                        p1x, p1y, p1z = self._render_coords(p[0] + v[0], p[1] + v[1], p[2] + v[2])
                        segments.append([[float(p0x), float(p0y), float(p0z)],
                                         [float(p1x), float(p1y), float(p1z)]])
                        colors.append(mcolors.to_rgba(GAS_DATA[gk]['color'], 0.45))
                    if segments:
                        ax.add_collection3d(art3d.Line3DCollection(
                            segments,
                            colors=colors,
                            linewidths=0.7,
                            zorder=4,
                        ))

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

        self.fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.94)
        self.canvas.draw_idle()
        self._update_stats()

    def _draw_center_of_mass_regions(self, ax):
        """Draw realtime center-of-mass halos for each active gas species."""
        if len(self.mol_pos) == 0:
            return

        azimuth = np.linspace(0.0, 2.0 * np.pi, 18)
        polar = np.linspace(0.0, np.pi, 9)
        sin_polar = np.sin(polar)[:, np.newaxis]
        cos_polar = np.cos(polar)[:, np.newaxis]
        cos_azimuth = np.cos(azimuth)[np.newaxis, :]
        sin_azimuth = np.sin(azimuth)[np.newaxis, :]

        for gas_key, indices in self.mol_group_indices.items():
            valid_indices = np.asarray(indices, dtype=np.intp)
            valid_indices = valid_indices[valid_indices < len(self.mol_pos)]
            if valid_indices.size == 0:
                continue

            gas = GAS_DATA.get(gas_key, GAS_DATA['N2'])
            positions = self.mol_pos[valid_indices]
            mass_weight = max(float(gas.get('m', GAS_DATA['N2']['m'])), 1e-12)
            weights = np.full(valid_indices.size, mass_weight, dtype=np.float64)
            center = np.average(positions, axis=0, weights=weights)
            centered = positions - center
            spread = np.sqrt(np.average(np.einsum('ij,ij->i', centered, centered), weights=weights))
            radius = float(np.clip(0.32 + 0.18 * spread, 0.42, 1.05))

            center_x, center_y, center_z = self._render_coords(center[0], center[1], center[2])
            center_x = float(center_x)
            center_y = float(center_y)
            center_z = float(center_z)

            sphere_x = center_x + radius * sin_polar * cos_azimuth
            sphere_y = center_y + radius * sin_polar * sin_azimuth
            sphere_z = center_z + radius * cos_polar * np.ones_like(cos_azimuth)
            color = gas.get('color', COLORS['accent'])

            ax.plot_surface(
                sphere_x,
                sphere_y,
                sphere_z,
                color=color,
                alpha=0.13,
                linewidth=0,
                shade=False,
                zorder=3,
            )
            ax.plot_wireframe(
                sphere_x,
                sphere_y,
                sphere_z,
                color=color,
                alpha=0.26,
                linewidth=0.45,
                rstride=2,
                cstride=3,
                zorder=4,
            )

            cross_radius = radius * 0.78
            ax.plot([center_x - cross_radius, center_x + cross_radius],
                    [center_y, center_y], [center_z, center_z],
                    color=color, alpha=0.58, linewidth=1.0, zorder=6)
            ax.plot([center_x, center_x], [center_y - cross_radius, center_y + cross_radius],
                    [center_z, center_z], color=color, alpha=0.58, linewidth=1.0, zorder=6)
            ax.plot([center_x, center_x], [center_y, center_y],
                    [center_z - cross_radius, center_z + cross_radius],
                    color=color, alpha=0.58, linewidth=1.0, zorder=6)

            ax.scatter([center_x], [center_y], [center_z],
                       s=82, c=[color], marker='P',
                       edgecolors=COLORS['text_bright'], linewidths=0.9,
                       depthshade=False, zorder=7)
            ax.text(center_x, center_y, center_z + radius * 1.16,
                    f"{gas.get('symbol', gas_key)} COM",
                    color=COLORS['text_bright'], fontsize=7.5,
                    ha='center', va='center', zorder=8,
                    bbox=dict(boxstyle='round,pad=0.22',
                              facecolor=color, edgecolor=COLORS['bg_card'],
                              alpha=0.78))

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

        cfg = self._get_active_sim_config()
        cf_mix = calc_mixture_correction_factor_physics(
            fracs,
            cfg,
            p_real,
            aN2=self._get_nominal_aN2(cfg),
        )
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
        self._bridge_calibration_cache.clear()
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
        self._bridge_calibration_cache.clear()
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
        self._bridge_calibration_cache.clear()
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
        """Return the auto color window centered on filament temperature ±3 °C."""
        cmin = self.T_HOT - 3.0
        cmax = self.T_HOT + 3.0
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
        self._sync_molecule_scale_to_pressure_setpoint()
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
        """Natural-convection loss from pressure, gas transport, gravity, and geometry."""
        p = max(float(p_pa), 0.0)
        dt = max(float(t_hot_k) - float(t_cold_k), 0.0)
        gravity = _effective_gravity(cfg)
        activation = float(_convection_pressure_activation(cfg, p))
        if p <= 0.0 or dt <= 0.0 or gravity <= 0.0 or activation <= 1e-9:
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

        l_char = _convection_characteristic_length(cfg)

        beta = 1.0 / t_film
        ra = gravity * beta * dt * (l_char ** 3) / max(nu * alpha, 1e-18)
        ra = float(np.clip(ra, 0.0, 1e12))
        nu_nat = _natural_convection_nusselt(ra, pr, cfg)
        conv_strength = max(float(nu_nat) - 1.0, 0.0)
        h = max((conv_strength * k_mix) / l_char, 0.0)
        h *= activation * _convection_geometry_factor(cfg) * max(float(cfg.get('conv_gain', DEFAULT_CONVECTION_GAIN)), 0.0)
        A = self._get_sensor_hot_area_m2(cfg)
        return h * A * dt

    def _solve_electro_thermal_state(self, p_pa, fracs, use_n2_only=False):
        """Solve Qel(T)=Qgas(T,P)+Qsupport(T)+Qrad(T)+Qconv(T,P) for filament T."""
        cfg = dict(self._get_active_sim_config())
        gas_cfg = dict(cfg)
        gas_cfg['conv_gain'] = 0.0  # convection is accounted separately in Q_conv
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
                    gas_cfg,
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
        for _ in range(36):
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

    def _bridge_calibration_key(self):
        cfg = self._get_active_sim_config()
        sensor_state = (
            self.sensor_t_ref_k,
            self.sensor_r0_ohm,
            self.sensor_tcr_per_k,
            self.bridge_v_bias,
            self.bridge_r1_ohm,
            self.bridge_r2_ohm,
            self.bridge_v_sensor,
            self.sensor_emissivity,
            self.sensor_support_lambda_wmk,
            self.sensor_support_w_m,
            self.sensor_support_t_m,
            self.sensor_support_l_m,
            self.sensor_extra_support_g_wpk,
        )
        return (_freeze_for_cache(cfg), _freeze_for_cache(sensor_state))

    def _get_n2_bridge_calibration_table(self):
        """Return cached monotonic Vout -> log(pressure) table for N2 calibration."""
        key = self._bridge_calibration_key()
        cached = self._bridge_calibration_cache.get(key)
        if cached is not None:
            return cached

        pressures = np.logspace(-10, math.log10(2e6), 72)
        volts = np.empty_like(pressures)
        for idx, pressure in enumerate(pressures):
            volts[idx] = self._bridge_output_for_pressure(
                pressure,
                {'N2': 1.0},
                use_n2_only=True,
            )['v_out_v']

        valid = np.isfinite(volts)
        if np.count_nonzero(valid) < 2:
            table = (np.array([], dtype=np.float64), np.array([], dtype=np.float64))
            _cache_put_bounded(self._bridge_calibration_cache, key, table, limit=12)
            return table

        volts = volts[valid]
        log_pressures = np.log(pressures[valid])
        order = np.argsort(volts)
        volts = volts[order]
        log_pressures = log_pressures[order]
        unique = np.concatenate(([True], np.diff(volts) > 1e-12))
        table = (volts[unique], log_pressures[unique])
        _cache_put_bounded(self._bridge_calibration_cache, key, table, limit=12)
        return table

    def _invert_n2_bridge_to_pressure(self, v_target):
        """Invert N2 bridge calibration curve to indicated pressure."""
        volts, log_pressures = self._get_n2_bridge_calibration_table()
        if len(volts) >= 2:
            log_p = np.interp(
                float(v_target),
                volts,
                log_pressures,
                left=float(log_pressures[0]),
                right=float(log_pressures[-1]),
            )
            return float(np.exp(log_p))

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
        "g            = effective gravitational acceleration used in buoyancy/convection terms\n"
        "Ra / Pr / Nu = Rayleigh, Prandtl, and Nusselt numbers for natural-convection scaling\n"
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

        register_gas_palette_listener(self._on_gas_palette_changed)

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

    def _on_gas_palette_changed(self):
        for tab in self.tabs:
            handler = getattr(tab, 'on_gas_palette_changed', None)
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
