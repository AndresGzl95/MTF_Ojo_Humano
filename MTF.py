# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 19:00:19 2025

@author: andre
"""

import numpy as np
import matplotlib.pyplot as plt

# Definir las funciones f(x) y g(x)
def f(x):
    return ( (2*x) / np.exp(x) ) ** 2

def g(x):
    return np.exp(-x)  # Distribución auxiliar (exponencial)

# Constante M tal que f(x) <= M * g(x) para todo x >= 0
M = 4  # Se calcula analíticamente: max(f(x)/g(x)) = 4 en x=1

# Generar muestras por aceptación-rechazo
def aceptacion_rechazo(n_samples):
    samples = []
    while len(samples) < n_samples:
        x = np.random.exponential(scale=1.0)  # Muestra de g(x) ~ Exp(1)
        u = np.random.uniform(0, 1)
        if u <= f(x) / (M * g(x)):
            samples.append(x)
    return np.array(samples)

# Generar 10,000 muestras
n_samples = 10_000
samples = aceptacion_rechazo(n_samples)

plt.figure(figsize=(10, 6))
plt.hist(samples, bins=50, density=True, alpha=0.6, label='Muestras generadas')

# Graficar f(x) teórica
x_vals = np.linspace(0, 10, 1000)
plt.plot(x_vals, f(x_vals), 'r-', lw=2, label='f(x) teórica')

plt.xlabel('x (Frecuencia espacial)')
plt.ylabel('Densidad')
plt.title('Simulación de la MTF del Ojo Humano')
plt.legend()
plt.grid()
plt.show()

# Calcular la MTF empírica (normalizada)
def mtf(x):
    return np.exp(-0.1 * x)  # Modelo simplificado para el ojo humano

x_mtf = np.linspace(0, 10, 100)
y_mtf = mtf(x_mtf)

# Graficar MTF
plt.figure(figsize=(10, 6))
plt.plot(x_mtf, y_mtf, 'b-', lw=3, label='MTF teórica')
plt.xlabel('Frecuencia espacial (ciclos/grado)')
plt.ylabel('Contraste relativo')
plt.title('Función de Transferencia de Modulación (MTF) del Ojo Humano')
plt.legend()
plt.grid()
plt.show()

# Guardar muestras en CSV
#import pandas as pd
#df = pd.DataFrame({'Frecuencia': samples})
#df.to_csv('mtf_samples.csv', index=False)