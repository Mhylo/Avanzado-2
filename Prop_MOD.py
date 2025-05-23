import numpy as np

def angular_spectrum_propagation(field, wavelength, dx, dy, z, n):
    """
    Propaga un campo complejo usando el método de espectro angular
    con índice de refracción fijo.
    
    Parámetros:
    -----------
    field : 2D np.ndarray (complex)
        Campo complejo inicial (amplitud + fase)
    wavelength : float
        Longitud de onda en metros
    dx, dy : float
        Pasos espaciales en x e y (m)
    z : float
        Distancia de propagación (m)
    n : float
        Índice de refracción del medio
    
    Retorna:
    --------
    field_propagated : 2D np.ndarray (complex)
        Campo propagado a la distancia z
    """
    k0 = 2 * np.pi / wavelength  # número de onda en vacío
    
    M, N = field.shape
    kx = np.fft.fftfreq(N, dx) * 2 * np.pi  # frecuencias espaciales en x (rad/m)
    ky = np.fft.fftfreq(M, dy) * 2 * np.pi  # frecuencias espaciales en y (rad/m)
    KX, KY = np.meshgrid(kx, ky)
    
    # Cálculo del k_z usando índice de refracción n
    kz_squared = (n * k0)**2 - KX**2 - KY**2
    # Evitar raíces complejas para valores evanescentes
    kz = np.sqrt(np.maximum(0, kz_squared)) + 1j * np.sqrt(np.maximum(0, -kz_squared))
    
    # Transformada de Fourier del campo inicial
    field_ft = np.fft.fft2(field)
    
    # Aplicar propagador angular
    H = np.exp(1j * kz * z)
    
    # Campo propagado en frecuencia espacial
    field_ft_propagated = field_ft * H
    
    # Volver al dominio espacial
    field_propagated = np.fft.ifft2(field_ft_propagated)
    
    return field_propagated
