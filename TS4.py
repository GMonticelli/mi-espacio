# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 19:35:54 2025

@author: gabri
"""

import numpy as np
import matplotlib.pyplot as plt

# Defino mis parámetros
R = 200
a1 = np.sqrt(2) # la amplitud máxima de la senoidal
N = 1000 # la cantidad de muestras
fr = np.random.uniform(-1/2, 1/2, R)

dc = 0 # su valor medio (volts)
ff = 0.9 # la frecuencia fundamental de la onda (Hz)
fs = 1000 # la frecuencia de muestreo del ADC
df = fs/N # Hz
ts = 1/fs # tiempo entre cada muestra

omega0 = fs/4
omega1 = omega0 + fr*df

#tt es el vector de instantes de tiempo en los que se toman las muestras de la señal
tt = np.linspace(0, (N-1)*ts, N).reshape((N,1)) #Vector de N posiciones
tt_m = np.tile(tt, (1, R)) # matriz de tiempo

# Argumento del seno
arg=2*np.pi*omega1*tt_m

# nk es la señal de ruido
snr = 10 #dB
pot_nk = 10**(-snr/10)
sigma = np.sqrt(pot_nk)

nk = np.random.normal(0, sigma, (N,R))

#Señal senoidal
xk = a1 * np.sin(arg) + nk

#plt.plot(xk)

#%% FFT

from scipy import signal

xk_fft = np.fft.fft(xk, axis=0) / N
ffx = np.linspace(0, (N-1)*df, N)
bfrec = ffx <= fs/2

plt.figure()
for i in range(200):
    plt.plot(ffx[bfrec], 10*np.log10(2*np.abs(xk_fft[bfrec, i])**2))

#Ventana Blackman
v_blackman = np.blackman(N).reshape((N,1))
SBlackmanharris = xk_fft*v_blackman

#Ventana Flattop
v_flattop = signal.windows.flattop(N).reshape((N,1))
SFlattop = xk_fft*v_flattop

#Estimador de amplitud
a1_R = np.abs(xk_fft[250,:])
a1_B = np.abs(SBlackmanharris[250,:])
a1_F = np.abs(SFlattop[250,:])

# Histogramas
plt.figure()
bins = 20
plt.hist(a1_R, bins=bins, label='FFT RECTANGULAR', alpha=0.3)
plt.hist(a1_B, bins=bins, label='FFT BLACKMAN', alpha=0.3)
plt.hist(a1_F, bins=bins, label='FFT FLATTOP', alpha=0.3)
plt.legend()

#%% Estimador omega

potencia_omega = (1/N)*np.abs(xk_fft[bfrec,:])**2
potencia_omega_B = (1/N)*np.abs(SBlackmanharris[bfrec,:])**2
potencia_omega_F = (1/N)*np.abs(SFlattop[bfrec,:])**2

maximo_omega = np.max(potencia_omega, axis=0)*df
maximo_omega_B = np.max(potencia_omega_B, axis=0)*df
maximo_omega_F = np.max(potencia_omega_F, axis=0)*df

plt.figure()
bins = 20
plt.hist(maximo_omega, bins=bins, label='FFT RECTANGULAR', alpha=0.3)
plt.hist(maximo_omega_B, bins=bins, label='FFT BLACKMAN', alpha=0.3)
plt.hist(maximo_omega_F, bins=bins, label='FFT FLATTOP', alpha=0.3)
plt.legend()

#Calculo de Sesgo y varianza
#a1

sesgo_a1_R = np.mean(a1_R - a1)
sesgo_a1_B = np.mean(a1_B - a1)
sesgo_a1_F = np.mean(a1_F - a1)

varianza_a1_R = np.var(a1_R - a1)
varianza_a1_B = np.var(a1_B - a1)
varianza_a1_F = np.var(a1_F - a1)

#Omega
sesgo_omega = np.mean(maximo_omega - omega1)
sesgo_omega_B = np.mean(maximo_omega_B - omega1)
sesgo_omega_F = np.mean(maximo_omega_F - omega1)

varianza_omega = np.var(maximo_omega - omega1)
varianza_omega_B = np.var(maximo_omega_B - omega1)
varianza_omega_F = np.var(maximo_omega_F - omega1)
