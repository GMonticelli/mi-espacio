#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 14:58:13 2025

@author: mariano
"""

# módulos y funciones a importar

import numpy as np

# Datos de la simulación

f0 = 1 # Hz
fs = 1000 # frecuencia de muestreo (Hz)
N = 1000 # cantidad de muestras

# Datos del ADC
B = 16 # bits
Vf = 2 # rango simétrico de +/- Vf Volts
q =  Vf/2**(B-1)# paso de cuantización de q Volts

# Datos del ruido (potencia de la señal normalizada, es decir 1 W)
pot_ruido_cuant = q**2/12# Watts 
kn = 10. # escala de la potencia de ruido analógico
pot_ruido_analog = pot_ruido_cuant * kn # 

ts = 1/fs  # tiempo de muestreo
df =fs/N  # resolución espectral

# Grilla de sampleo temporal

tt = np.linspace(0, (N-1)*ts, N)
argg = 2*np.pi*f0*tt

xx = np.sqrt(2)*np.sin(argg)
varianza = np.var(xx)

analog_sig = xx/np.sqrt(varianza) # señal analógica normalizada sin ruido
nn = np.random.normal(0, np.sqrt(pot_ruido_cuant), N) # señal de ruido de analógico
sr = analog_sig + nn# señal analógica de entrada al ADC (con ruido analógico)
srq = np.round(sr/q)*q # señal cuantizada

nq = srq - sr # señal de ruido de cuantización




#%% Visualización de resultados

import matplotlib.pyplot as plt

##################
# Señal temporal
##################

plt.figure(1)


plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q))
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')
plt.plot(tt, srq, lw=1, ls='solid', color='green', marker='o', markersize=1,markerfacecolor='blue', markeredgecolor='blue', fillstyle='none', label='$s_{RQ}$ -Señal cuantizada')
plt.plot(tt, sr, lw=1, ls='dashdot', color='blue', label='$s_{R}$ - Señal analógica de entrada al ADC (con ruido analógico)')
plt.plot(tt, analog_sig, lw=2, color='orange', ls='dotted', label='$s$-Señal analógica con ruido analógico')
axes_hdl = plt.gca()
axes_hdl.legend()
plt.show()

#%% 

plt.figure(2)
ft_SR = 1/N*np.fft.fft( sr, axis = 0 )
ft_Srq = 1/N*np.fft.fft( srq, axis = 0 )
ft_As = 1/N*np.fft.fft( analog_sig, axis = 0)
ft_Nq = 1/N*np.fft.fft( nq, axis = 0 )
ft_Nn = 1/N*np.fft.fft( nn, axis = 0 )

# grilla de sampleo frecuencial
ff = np.linspace(0, (N-1)*df, N)

bfrec = ff <= fs/2

Nnq_mean = np.mean(np.abs(ft_Nq)**2)
nNn_mean = np.mean(np.abs(ft_Nn)**2)

plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_As[bfrec])**2), color='orange', ls='dotted', label='$ s $ (sig.)' )
plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([nNn_mean, nNn_mean]) ), '--r', label= '$ \overline{n} = $' + '{:3.1f} dB'.format(10* np.log10(2* nNn_mean)) )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_SR[bfrec])**2), ':g', label='$ s_R = s + n $' )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Srq[bfrec])**2), lw=2, label='$ s_Q = Q_{B,V_F}\{s_R\}$' )
plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([Nnq_mean, Nnq_mean]) ), '--c', label='$ \overline{n_Q} = $' + '{:3.1f} dB'.format(10* np.log10(2* Nnq_mean)) )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Nn[bfrec])**2), ':r')
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Nq[bfrec])**2), ':c')
plt.plot( np.array([ ff[bfrec][-1], ff[bfrec][-1] ]), plt.ylim(), ':k', label='BW', lw = 0.5  )

plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q) )
plt.ylabel('Densidad de Potencia [dB]')
plt.xlabel('Frecuencia [Hz]')
axes_hdl = plt.gca()
axes_hdl.legend()

#############
# Histograma
#############

plt.figure(3)
bins = 10
plt.hist(nq.flatten()/(q), bins=bins)
plt.plot( np.array([-1/2, -1/2, 1/2, 1/2]), np.array([0, N/bins, N/bins, 0]), '--r' )
plt.title( 'Ruido de cuantización para {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q))

plt.xlabel('Pasos de cuantización (q) [V]')

