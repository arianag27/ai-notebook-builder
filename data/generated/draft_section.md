# Draft Section

**Request:** Create a beginner widget section for a calculus notebook on Riemann sums
**Section type:** widget section
**Discipline:** Mathematics
**Topic:** Riemann sums
**Coding level:** beginner

---

## Interactive Exploration: Riemann Sums

Use the widgets below to explore the lesson concept without rewriting code each time.

### Setup

Run this cell once to load libraries:

```python
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact, IntSlider, FloatSlider, Dropdown
%matplotlib inline
```

### Instructions

Adjust the controls and observe how the plot changes:

- **Function:** Choose which function to visualize
- **Number of rectangles:** Change the partition count for the Riemann sum
- **Interval bounds:** Set the left and right endpoints
- **Tangent point:** Move the x-value where the tangent line is drawn

### Interactive cell

```python
# Starter scaffold — customize for your lesson
@interact
def explore_riemann(n_rectangles=10, a=0.0, b=1.0):
    # TODO: plot function and Riemann sum rectangles
    pass
```

### What to notice

- What pattern do you see when you change the first parameter?
- Does the result match what you expected from the definition?
- What would you try next to test your understanding?


---

## Notes on Influencing Examples

This draft was shaped by the following corpus patterns (not copied verbatim):

- **sums.ipynb** (Mathematics / Calculus) — referenced the *WIDGET PATTERNS* section
- **AI_Ethics_in_Data_Science.ipynb** (Ethics / Data Science) — referenced the *WIDGET PATTERNS* section
- **Orbitals.ipynb** (Chemistry / Physics) — referenced the *WIDGET PATTERNS* section
- **parental-risk.ipynb** (Public Health / Epidemiology) — referenced the *WIDGET PATTERNS* section
- Style pattern adopted: References interactive tools: interact(), sliders, dropdowns, ipywidgets

---

## Source Chunks Used

Found 5 result(s)

------------------------------------------------------------
Result 1  (similarity: 1.023)

Source:
sums.ipynb

Section:
WIDGET PATTERNS

Discipline:
Mathematics / Calculus

Text:
- ipywidgets library
- @interact decorator / interact()
- widgets API
- dropdown widget
- slider widget

------------------------------------------------------------
Result 2  (similarity: 0.761)

Source:
AI_Ethics_in_Data_Science.ipynb

Section:
WIDGET PATTERNS

Discipline:
Ethics / Data Science

Text:
- external widget file (%run)
- custom ethics widget

------------------------------------------------------------
Result 3  (similarity: 0.688)

Source:
Orbitals.ipynb

Section:
WIDGET PATTERNS

Discipline:
Chemistry / Physics

Text:
- ipywidgets library
- @interact decorator / interact()
- widgets API
- dropdown widget

------------------------------------------------------------
Result 4  (similarity: 0.635)

Source:
parental-risk.ipynb

Section:
WIDGET PATTERNS

Discipline:
Public Health / Epidemiology

Text:
- ipywidgets library
- @interact decorator / interact()
- widgets API
- dropdown widget
- button widget

------------------------------------------------------------
Result 5  (similarity: 0.626)

Source:
sums.ipynb

Section:
HEADINGS

Discipline:
Mathematics / Calculus

Text:
3.1. Visualizing Riemann Sums for a Fixed Function
