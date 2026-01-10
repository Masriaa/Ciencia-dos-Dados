import numpy as np

def calcular_peso_cubado(comprimento, altura, largura, fator_cubagem=6000):
    volume_cm3 = np.array(comprimento) * np.array(altura) * np.array(largura)
    peso_cubado_kg = volume_cm3 / fator_cubagem
    return peso_cubado_kg * 1000

def haversine_vectorized(lat1, lon1, lat2, lon2, R=6371.0):
    # Converte graus para radianos
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c