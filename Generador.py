# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 09:02:27 2025

@author: gabri
"""

import numpy as np
import matplotlib.pyplot as plt

# Parámetros de la onda

vmax = 1 # Amplitud maxima
dc = 0 # Valor medio
f0 = 500 # Hz
ph = 0*np.pi # Fase

fs = 1000 # frecuencia de muestreo
N = 1000 # Cantidad de muestras

ts = 1/fs # tiempo de muestreo

# Grilla de sampleo temporal
tt = np.linspace(0, (N-1)*ts, N)
argg = 2*np.pi*f0*tt+ph

xx = np.sin(argg)*vmax

plt.plot(tt, xx)

#%% Onda de 999 Hz

f0 = 999 # Hz

argg = 2*np.pi*f0*tt+ph
xx = np.sin(argg)*vmax
plt.plot(tt, xx)

#%% Onda de 1001 Hz

f0 = 1001 # Hz
plt.plot(tt, xx)

#%% Onda de 2001 Hz

f0 = 2001 # Hz
plt.plot(tt, xx)

#%% Onda predeterminada

from scipy import signal

xx = signal.square(tt, duty=0.075)
plt.plot(tt, xx)