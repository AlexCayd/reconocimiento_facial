# Archivo: reconocimiento_facial.py

import numpy as np
from PIL import Image
import os

# NOTA: Todas las funciones ahora aceptan un argumento 'log' para registrar mensajes
# en lugar de usar print().

def cargar_imagenes(log):
    log.append("\n" + "="*50)
    log.append("CARGANDO IMÁGENES DE ENTRENAMIENTO")
    log.append("="*50)
    
    tamaño_objetivo = (256, 256)
    todas_las_caracteristicas = []
    dataset_dir = "dataset"
    
    if not os.path.exists(dataset_dir) or not os.listdir(dataset_dir):
        log.append(f"ERROR: La carpeta '{dataset_dir}' no existe o está vacía.")
        return None, log

    archivos = sorted([f for f in os.listdir(dataset_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    log.append(f"Se encontraron {len(archivos)} imágenes en el dataset.")

    for nombre_archivo in archivos:
        ruta_archivo = os.path.join(dataset_dir, nombre_archivo)
        vector, _ = cargar_y_procesar_imagen(ruta_archivo, tamaño_objetivo)
        if vector is not None:
            todas_las_caracteristicas.append(vector)
        else:
            log.append(f"ERROR: No se pudo cargar {nombre_archivo}")
            return None, log
            
    return np.array(todas_las_caracteristicas), log

def cargar_imagen_prueba(log):
    tamaño_objetivo = (256, 256)
    test_dir = 'imagen_prueba'
    
    if not os.path.exists(test_dir) or not os.listdir(test_dir):
        log.append(f"ERROR: La carpeta '{test_dir}' no existe o está vacía.")
        return None, log
        
    # Asumimos que solo hay una imagen en la carpeta de prueba
    nombre_archivo = os.listdir(test_dir)[0]
    ruta_archivo = os.path.join(test_dir, nombre_archivo)
    
    vector, _ = cargar_y_procesar_imagen(ruta_archivo, tamaño_objetivo)
    
    if vector is not None:
        log.append(f"Imagen de prueba '{nombre_archivo}' cargada correctamente.")
        return vector, log
    else:
        log.append(f"ERROR: No se pudo cargar la imagen de prueba '{nombre_archivo}'.")
        return None, log

def aplicar_pca(A, log):
    log.append("\n" + "="*50)
    log.append("APLICANDO PCA")
    log.append("="*50)

    mu = np.mean(A, axis=0)
    Xc = A - mu
    log.append(f"Datos centrados (Xc = A - μ). Dimensiones: {Xc.shape}")

    U, s, Vt = np.linalg.svd(Xc.T, full_matrices=False)
    n = A.shape[0]
    lambdas = (s**2) / (n-1)
    V = U

    log.append("\nValores propios (λ) más importantes:")
    for i in range(min(10, len(lambdas))):
        log.append(f"   λ{i+1}: {lambdas[i]:.6f}")
    
    total_varianza = np.sum(lambdas)
    varianza_acumulada = 0.0
    k = 0
    umbral = 0.95
    log.append(f"\nBuscando 'k' para alcanzar el {umbral:.0%} de varianza...")
    
    for i, lam in enumerate(lambdas):
        varianza_acumulada += lam / total_varianza
        if varianza_acumulada >= umbral:
            k = i + 1
            log.append(f"   Se necesitan {k} componentes para explicar el {varianza_acumulada:.2%} de la varianza.")
            break
            
    return mu, Xc, V[:, :k], k, varianza_acumulada, log

# ... (cargar_y_procesar_imagen y reconstruir_imagen no necesitan cambios si no imprimen nada)
def cargar_y_procesar_imagen(nombre_archivo, tamaño_objetivo=None):
    try:
        imagen = Image.open(nombre_archivo).convert('L')
        if tamaño_objetivo:
            imagen = imagen.resize(tamaño_objetivo)
        img_array = np.array(imagen)
        return img_array.flatten().astype(float) / 255.0, imagen.size
    except Exception as e:
        print(f"ERROR al procesar {nombre_archivo}: {e}") # Este print puede quedar para la consola del server
        return None, None

def reconstruir_imagen(z, Vk, mu):
    Xc_reconstruido = np.dot(Vk, z)
    return Xc_reconstruido + mu

def calcular_error_de_reconstruccion(x, mu, Vk, log, img_label="imagen"):
    log.append("\n" + "="*50)
    log.append(f"CALCULANDO ERROR DE RECONSTRUCCIÓN PARA: {img_label}")
    log.append("="*50)

    log.append("   Paso 1: Centrando la imagen...")
    x_centrado = x - mu
    
    log.append("   Paso 2: Proyectando al espacio PCA...")
    z = np.dot(Vk.T, x_centrado)
    log.append(f"      Proyección obtenida (vector de {len(z)} dims).")

    log.append("   Paso 3: Reconstruyendo la imagen...")
    x_reconstruido = reconstruir_imagen(z, Vk, mu)
    log.append("      Imagen reconstruida.")
    
    log.append("   Paso 4: Calculando el error...")
    error = np.linalg.norm(x - x_reconstruido)
    log.append(f"      Error de reconstrucción: {error:.4f}")
    
    return error, x_reconstruido, log

# Esta es la nueva función principal que llamaremos desde Flask
def ejecutar_analisis_completo():
    log = [] # Inicializamos la lista para guardar los mensajes
    results = {} # Inicializamos el diccionario para guardar los resultados

    # --- FASE DE ENTRENAMIENTO ---
    A, log = cargar_imagenes(log)
    if A is None:
        return None, log

    mu, Xc, Vk, k, var_expl, log = aplicar_pca(A, log)
    results['k'] = k
    results['variance_explained'] = var_expl

    # --- FASE DE EVALUACIÓN ---
    log.append("\n" + "="*60)
    log.append("FASE DE EVALUACIÓN DEL SISTEMA Y UMBRAL DE DECISIÓN (T)")
    log.append("="*60)

    errores_entrenamiento = []
    for i, x_entrenamiento in enumerate(A):
        error, _, log = calcular_error_de_reconstruccion(x_entrenamiento, mu, Vk, log, img_label=f"Entrenamiento {i+1}.jpg")
        errores_entrenamiento.append(error)

    umbral_T = max(errores_entrenamiento)
    results['threshold_t'] = umbral_T
    log.append(f"\nUMBRAL DE DECISIÓN (T) ESTABLECIDO EN: {umbral_T:.4f}")

    # --- EVALUAR IMAGEN DE PRUEBA ---
    x_prueba, log = cargar_imagen_prueba(log)
    if x_prueba is None:
        return None, log

    error_prueba, x_reconstruido, log = calcular_error_de_reconstruccion(x_prueba, mu, Vk, log, img_label="Prueba")
    results['reconstruction_error'] = error_prueba

    # --- VEREDICTO FINAL ---
    log.append("\n" + "="*50)
    log.append("VEREDICTO FINAL")
    log.append("="*50)
    log.append(f"   Error de la imagen de prueba: {error_prueba:.4f}")
    log.append(f"   Umbral de decisión (T): {umbral_T:.4f}")

    if error_prueba <= umbral_T:
        results['is_recognized'] = True
        log.append("\n   >> RESULTADO: Rostro RECONOCIDO.")
        log.append("   Explicación: El error está por debajo del umbral.")
    else:
        results['is_recognized'] = False
        log.append("\n   >> RESULTADO: Rostro DESCONOCIDO.")
        log.append("   Explicación: El error supera el umbral.")

    # Aquí agregaríamos la lógica para guardar las imágenes y devolver las rutas
    # pero por ahora, solo devolvemos los datos y el log.
    
    return results, log