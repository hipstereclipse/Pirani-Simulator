# Sources and Provenance

This document records where the simulator's data and equations come from, and what is implementation-specific.

## Primary Scientific Source

1. K. Jousten, "On the gas species dependence of Pirani vacuum gauges," Journal of Vacuum Science & Technology A, 26(3), 352-359 (2008).
   DOI: https://doi.org/10.1116/1.2897314

Used for:
- gas-species dependence framework,
- correction-factor interpretation,
- tabulated comparisons used by the app (for example, correction factors and accommodation-ratio relationships).

## Physical Constants

Fundamental constants used in code (for example Boltzmann and Stefan-Boltzmann constants) follow standard SI/NIST values.

## Simulator Presets and Engineering Assumptions

Several values in the implementation are practical modeling assumptions intended to keep the simulation numerically stable and educationally useful across a large dynamic range. These include:

- gauge geometry presets,
- smooth bridging coefficients for molecular-to-viscous transition,
- convection scaling terms,
- UI-oriented defaults and range limits.

These assumptions are not direct measurements from a single calibrated instrument and should be interpreted as model parameters.

## Commercial Gauge Context

Some preset names and descriptive ranges in the simulator are aligned with commonly published product-family behavior for conventional wire and MEMS/convection Pirani gauges. They are included for intuition and comparison, not as a substitute for the latest manufacturer calibration or datasheet specifications.

## Recommended Citation for This Software

If citing this repository, cite both:

1. This software repository (version/tag used).
2. Jousten (2008) as the primary scientific reference.

## Scope and Limits

This simulator is designed for:
- education,
- comparative trend analysis,
- rapid what-if exploration.

It is not intended for:
- traceable metrology,
- safety-critical control validation,
- replacing instrument-specific calibration procedures.
