#!/usr/bin/env python3
"""
Pirani Vacuum Gauge Simulator — Launcher
=========================================

This script checks dependencies and launches the simulator.

Usage:
    python run_pirani.py

Requirements:
    - Python 3.8+
    - matplotlib
    - numpy
    - tkinter (included with most Python installations)
"""

import subprocess
import sys
import os


def check_python_version():
    if sys.version_info < (3, 8):
        print(f"ERROR: Python 3.8+ required. You have {sys.version}")
        sys.exit(1)
    print(f"  Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} ✓")


def check_tkinter():
    try:
        import tkinter
        print(f"  tkinter ✓")
        return True
    except ImportError:
        print("  tkinter ✗  — REQUIRED")
        print()
        print("  tkinter is not installed. Install it with:")
        print("    Ubuntu/Debian:  sudo apt-get install python3-tk")
        print("    Fedora:         sudo dnf install python3-tkinter")
        print("    macOS:          brew install python-tk@3.x  (or reinstall Python from python.org)")
        print("    Windows:        Reinstall Python and check 'tcl/tk' in the installer")
        return False


def check_matplotlib():
    try:
        import matplotlib
        print(f"  matplotlib {matplotlib.__version__} ✓")
        return True
    except ImportError:
        print("  matplotlib ✗  — installing...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'matplotlib'])
        print("  matplotlib ✓  (installed)")
        return True


def check_numpy():
    try:
        import numpy
        print(f"  numpy {numpy.__version__} ✓")
        return True
    except ImportError:
        print("  numpy ✗  — installing...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy'])
        print("  numpy ✓  (installed)")
        return True


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     Pirani Vacuum Gauge Simulator                      ║")
    print("║     Based on Jousten (2008)                            ║")
    print("║     J. Vac. Sci. Technol. A 26, 352-359               ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    print("Checking dependencies...")

    check_python_version()
    tk_ok = check_tkinter()
    np_ok = check_numpy()
    mpl_ok = check_matplotlib()

    if not all([tk_ok, np_ok, mpl_ok]):
        print()
        print("Some dependencies are missing. Please install them and try again.")
        sys.exit(1)

    print()
    print("All dependencies OK. Launching simulator...")
    print()

    # Launch the main application
    script_dir = os.path.dirname(os.path.abspath(__file__))
    simulator_path = os.path.join(script_dir, 'pirani_simulator.py')

    if not os.path.exists(simulator_path):
        print(f"ERROR: Cannot find pirani_simulator.py at {simulator_path}")
        sys.exit(1)

    sys.exit(subprocess.call([sys.executable, simulator_path]))


if __name__ == '__main__':
    main()
