# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 09:50:14 2025

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
    S     : np.array - Price of underlying asset
    t     : np.array - Time in days, 0 <= t <= T
    r     : float - Annualised risk-free interest rate
    D     : float - Annualised dividend yield
    sigma : float - Annualised volatility of underlying
    T     : float - Time to expiry (days)
    E     : float - Strike price of option

    Returns
    -------
    np.array - Call option values
    """
    dt = np.maximum( (T - t)/252 , 0.00001) # Annualise time (assuming 252 trading days/year) and avoid dividing by 0 later
    d1 = (np.log(S / E) + (r - D + 0.5*sigma**2) * dt) / (sigma * np.sqrt(dt))
    d2 = d1 - sigma * np.sqrt(dt)
    return S * np.exp(-D * dt) * norm.cdf(d1) - E * np.exp(-r * dt) * norm.cdf(d2)

# ----------
# Parameters
# ----------

r = 0.05            #Annualised risk-free interest rate
D = 0.02            #Annualised dividend yield
sigma = 0.1         #Annualised volatility of underlying
T = 5*5             #Time to expiry (5 weeks ~ 25 trading days)
E = 100             #Strike price of option

# --------------------------------------------------------
# 2D Plots of Call Value against Time and Underlying Price
# --------------------------------------------------------

#Creating S & t arrays
N = T                       #Resolution of arrays = 1 day
S = np.linspace(0,150,N+1)
t = np.linspace(0,T,N+1)

#Plotting 2D graphs
fig, axs = plt.subplots(1,2, figsize=(10,5))

#Plotting Value against t
axs[0].plot(t , C(100,t,r,D,sigma,T,E) , color='black')
axs[0].set_xlabel('Time (days)')
axs[0].set_ylabel('Call Option Value (£)')
axs[0].set_title('Call Option Value against Time (S=100)')

#Plotting Value against S
axs[1].plot(S , C(S,0.00001,r,D,sigma,T,E) , color='red') #0.00001 to avoid dividing by zero
axs[1].set_xlabel('Underlying Value (£)')
axs[1].set_ylabel('Call Option Value (£)')
axs[1].set_title('Call Option Value against Underlying (t~0)')

plt.suptitle('Black-Scholes Call Option Pricing', fontsize = 15)
plt.tight_layout()
plt.show() 

# ------------------------------------------------------------
# 3D Plot of Call Value varying with Time and Underlying Price
# ------------------------------------------------------------

#Creating meshgrid for plotting
S_grid, t_grid = np.meshgrid(S,t)
Call = C(S_grid,t_grid,r,D,sigma,T,E)

fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection="3d")

ax.plot_surface(t_grid, S_grid ,Call, cmap="Spectral")
ax.set_xlabel("Time (days)")
ax.set_ylabel('Underlying Value (£)')
ax.set_zlabel('Call Option Value (£)')
ax.set_title('Black-Scholes Call Option Value Surface Plot')
plt.show()