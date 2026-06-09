# LLM Notebook Draft

**Generated:** 2026-06-08T17:45:10.144330+00:00
**Discipline:** Calculus
**Topic:** derivatives and integrals
**Coding level:** intermediate
**Retrieved examples:** 12

---

# Understanding Derivatives and Integrals: A Dynamic Exploration  

## Introduction  
Derivatives and integrals are foundational concepts in calculus that describe how quantities change and accumulate. Derivatives measure instantaneous rates of change (like velocity from position), while integrals calculate total accumulation (like distance from velocity). These tools are essential for analyzing motion, optimizing functions, and modeling real-world phenomena. In this notebook, you’ll explore their relationship, practice computing them, and use interactive visualizations to deepen your understanding.  

## Learning Objectives  
By the end of this notebook, you will:  
- Explain the relationship between derivatives and integrals (Fundamental Theorem of Calculus).  
- Compute derivatives and integrals of basic functions (sin, cos, tan).  
- Visualize derivatives (slopes) and integrals (area under curves) using interactive widgets.  
- Apply calculus rules (power rule, chain rule) to symbolic and numerical problems.  

## Suggested Libraries  
```python
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import ipywidgets as widgets
from IPython.display import display
```  

## 1. Derivatives: The Rate of Change  
### 1.1. What is a Derivative?  
A derivative measures how a function changes at a specific point. For example, the derivative of $ f(x) = \sin(x) $ is $ f'(x) = \cos(x) $.  

### 1.2. Computing Derivatives with SymPy  
Use `sympy` to symbolically compute derivatives.  
```python
x = sp.symbols('x')
f = sp.sin(x)  # Example function
derivative = sp.diff(f, x)
print("Derivative of sin(x):", derivative)
```

### 1.3. Visualizing Derivatives with Widgets  
Adjust the function and see its derivative in real time.  
```python
def plot_derivative(func, x_range=(-2*np.pi, 2*np.pi)):
    x_vals = np.linspace(x_range[0], x_range[1], 400)
    y_vals = func(x_vals)
    plt.plot(x_vals, y_vals, label=f"f(x) = {func.__name__}(x)")
    plt.title("Function and Its Derivative")
    plt.legend()
    plt.show()

# Example: Derivative of cos(x)
plot_derivative(np.cos)
```  

## 2. Integrals: Accumulating Area  
### 2.1. What is an Integral?  
An integral calculates the area under a curve. For example, the integral of $ f(x) = \cos(x) $ from $ -\pi $ to $ \pi $ is 0.  

### 2.2. Computing Integrals with SymPy  
Use `sympy` to symbolically integrate functions.  
```python
x = sp.symbols('x')
g = sp.cos(x)  # Example function
integral = sp.integrate(g, x)
print("Integral of cos(x):", integral)
```

### 2.3. Visualizing Integrals with Widgets  
Adjust the limits and see the area under the curve.  
```python
@widgets.interact(a=widgets.FloatSlider(min=-2*np.pi, max=2*np.pi, step=0.5, value=-np.pi),
                  b=widgets.FloatSlider(min=-2*np.pi, max=2*np.pi, step=0.5, value=np.pi))
def plot_integral(a, b):
    x_vals = np.linspace(a, b, 400)
    y_vals = np.cos(x_vals)
    plt.fill_between(x_vals, y_vals, color='skyblue', alpha=0.5)
    plt.title(f"Integral of cos(x) from {a:.2f} to {b:.2f}")
    plt.show()
```  

## 3. Guided Practice: Calculating Derivatives and Integrals  
### 3.1. Practice Problem 1  
**Task:** Compute the derivative of $ f(x) = \tan(x) $ using `sympy`.  
```python
x = sp.symbols('x')
f = sp.tan(x)
derivative = sp.diff(f, x)
print("Derivative of tan(x):", derivative)
# TODO: Verify the result using the chain rule manually
```

### 3.2. Practice Problem 2  
**Task:** Compute the integral of $ g(x) = \sin(x) $ from $ 0 $ to $ \pi $.  
```python
x = sp.symbols('x')
g = sp.sin(x)
integral = sp.integrate(g, (x, 0, np.pi))
print("Integral of sin(x) from 0 to π:", integral)
# TODO: Calculate the area manually and compare
```  

## 4. Visualizing Derivatives and Integrals with Widgets  
### 4.1. Interactive Derivative Plot  
Explore how the derivative of $ f(x) = \sin(x) $ changes with amplitude.  
```python
@widgets.interact(amplitude=widgets.FloatSlider(min=0.1, max=2.0, step=0.1, value=1.0))
def plot_derivative_amplitude(amplitude):
    x_vals = np.linspace(-2*np.pi, 2*np.pi, 400)
    y_vals = amplitude * np.sin(x_vals)
    plt.plot(x_vals, y_vals, label=f"Amplitude = {amplitude:.1f}")
    plt.plot(x_vals, amplitude * np.cos(x_vals), label="Derivative")
    plt.title("Function and Derivative with Adjustable Amplitude")
    plt.legend()
    plt.show()
```  

## Reflection Questions  
1. How do derivatives and integrals relate to each other? Can you think of a real-world scenario where both are used?  
2. What happens to the derivative of a function when its slope becomes vertical (e.g., $ f(x) = \sqrt{x} $)?  
3. How might the Fundamental Theorem of Calculus simplify solving problems involving both derivatives and integrals?  

## Extension Activities  
1. **Advanced Challenge:** Use `sympy` to compute the derivative of $ f(x) = \ln(\sin(x)) $ and verify it using the chain rule.  
2. **Real-World Application:** Model the motion of a particle with position $ s(t) = \sin(t) $ and calculate its velocity and total distance traveled over $ [0, 2\pi] $.  
3. **Creative Exploration:** Design a custom function (e.g., $ f(x) = x^3 - 3x $) and use widgets to visualize its derivative and integral.  

---  
This notebook blends conceptual understanding with hands-on practice, ensuring you grasp both the theory and application of derivatives and integrals.

## Influenced By

- ECC CalEnviroScreen.ipynb
- ecc-calculus__riemann_sum.ipynb
- ecc-calculus__sums.ipynb
- ecc-cs9__ECC CalEnviroScreen.ipynb
- sums.ipynb
