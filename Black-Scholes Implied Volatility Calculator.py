# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 18:18:10 2025

@author: Adrian.ph689
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


# -------------------------------------------
# Black-Scholes Call Option Pricing Function
# -------------------------------------------

def C(S,t,r,D,sigma,T,E):
    """
    Parameters
    ----------
    S        : np.array - Price of underlying asset
    t        : np.array - Time in days, 0 <= t <= T
    r        : float - Annualised risk-free interest rate
    D        : float - Annualised dividend yield
    sigma    : float - Annualised volatility of underlying
    T        : float - Time to expiry (days)
    E        : float - Strike price of option

    Returns
    -------
    np.array - Call option values
    """
    dt = np.maximum( (T - t) , 0.00001) # Annualise time (assuming 252 trading days/year) and avoid dividing by 0 later
    d1 = (np.log(S / E) + (r - D + 0.5*sigma**2) * dt) / (sigma * np.sqrt(dt))
    d2 = d1 - sigma * np.sqrt(dt)
    return S * np.exp(-D * dt) * norm.cdf(d1) - E * np.exp(-r * dt) * norm.cdf(d2)

# ------------------------------------
# Black-Scholes Call Option Vega Value
# ------------------------------------

def Vega(S,t,r,D,sigma,T,E):
    """
    Parameters
    ----------
    S        : np.array - Price of underlying asset
    t        : np.array - Time in days, 0 <= t <= T
    r        : float - Annualised risk-free interest rate
    D        : float - Annualised dividend yield
    sigma    : float - Annualised volatility of underlying
    T        : float - Time to expiry (days)
    E        : float - Strike price of option

    Returns
    -------
    np.array - Vega values of call option
    """
    dt = np.maximum( (T - t) , 0.00001) # Avoid dividing by 0 later
    d1 = (np.log(S / E) + (r - D + 0.5*sigma**2) * dt) / (sigma * np.sqrt(dt))
    return S * np.sqrt(dt) * np.exp(-D * dt) * np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi)
    
# ----------------------------------------------
# Newton-Raphson solution for Implied Volatility
# ----------------------------------------------

def Implied_Vol(S,t,r,D,T,E,V_market,error):
    """
        Parameters
    ----------
    S        : np.array - Price of underlying asset
    t        : np.array - Time in days, 0 <= t <= T
    r        : float - Annualised risk-free interest rate
    D        : float - Annualised dividend yield
    T        : float - Time to expiry (days)
    E        : float - Strike price of option
    V_market : float - Market value of option
    error    : float - The acceptable error in the implied volatility

    Returns
    -------
    float - Implied volatility
    """
    for i in range(5):  # Num iterations until convergence
        vol = np.random.uniform(0.01, 1)
        count = 0
        while count < 30:
            V = C(S, t, r, D, vol, T, E)
            vega = Vega(S, t, r, D, vol, T, E)
            if vega == 0:
                break
            dv = (V - V_market) / vega
            vol -= dv
            vol = max(vol, 0.0001)
            if abs(dv) < error:
                return np.clip(vol,0.0001,5) # Prevents edge case as t -> T when S = E
            count += 1
    return np.nan  # All retries failed

Implied_Vol_Vector = np.vectorize(Implied_Vol)

# ----------
# Parameters
# ----------

r = 0.08              # Annualised risk-free interest rate
D = 0                 # Annualised dividend yield
T = 0.25              # Time to expiry (4 months in years)
E = 100               # Strike price of option
V_market = 6.51       # Market value of option
error = 0.001         # Acceptable error in implied volatility

# --------------------------------------------------------
# 2D Plot of Implied Volatility against Time and Underlying Price
# --------------------------------------------------------

#Creating S & t arrays
S = np.linspace(50,150,100) # Resolution per £1
t = np.linspace(0,T,int(T*252))  # Resolution per day

#Plotting 2D graphs
fig, axs = plt.subplots(1,2, figsize=(10,5))

#Plotting Value against t
axs[0].plot(t , Implied_Vol_Vector(101.5,t,r,D,T,E,V_market,error) , color='black')
axs[0].set_xlabel('Time (days)')
axs[0].set_ylabel('Implied Volatility')
axs[0].set_title('Implied Call Option Volatility against Time (S=100)')

#Plotting Value against S
axs[1].plot(S , Implied_Vol_Vector(S,0.00001,r,D,T,E,V_market,error) , color='red') #0.00001 to avoid dividing by zero
axs[1].set_xlabel('Underlying Value (£)')
axs[1].set_ylabel('Implied Volatility')
axs[1].set_title('Implied Call Option Volatility against Underlying (t~0)')

plt.suptitle('Black-Scholes Call Option Implied Volatility', fontsize = 15)
plt.tight_layout()
plt.show() 

# ------------------------------------------------------------
# 3D Plot of Call Value varying with Time and Underlying Price
# ------------------------------------------------------------

#Creating meshgrid for plotting
S_grid, t_grid = np.meshgrid(S,t)
Imp_Vol = Implied_Vol_Vector(S_grid,t_grid,r,D,T,E,V_market,error)

fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection="3d")

ax.plot_surface(t_grid, S_grid ,Imp_Vol, cmap="Spectral")
ax.set_xlabel("Time (years)")
ax.set_ylabel('Underlying Value (£)')
ax.set_zlabel('Implied Volatility')
ax.set_title('Black-Scholes Call Option Implied Volatility Surface Plot')
plt.show()
