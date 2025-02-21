import numpy as np
# Stap 1 van het algoritme
# We berekenen hier de afgeleide
def find_derivative(tensor):
    dim = len(tensor) #cubic -> ok :)
    zeros = np.zeros(dim)
    