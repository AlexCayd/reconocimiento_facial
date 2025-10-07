import numpy as np
from PIL import Image
import os
import sys

print("=== RECONOCIMIENTO FACIAL CON PCA ===\n")

def cargar_y_procesar_imagen(nombre_archivo, tamaño_objetivo=None):
    """
    Carga la imagenes y devuelve los vectores correspondientes
    """
    try:
        imagen = Image.open(nombre_archivo)
        
        # Convertir a escala de grises (blanco y negro)
        if imagen.mode != 'L':
            imagen = imagen.convert('L')
        
        # Redimensionar para que todas las imágenes tengan el mismo tamaño
        if tamaño_objetivo:
            imagen = imagen.resize(tamaño_objetivo)
        
        # Resultado: una matriz de 256 filas, 256 columnas, cada celda de la matriz  tiene un numero de entre 0 a 255.
        # Convertir imagen a array de numpy
        img_array = np.array(imagen)  
        vector_caracteristicas = img_array.flatten()  

        # Resultado:  .flatten es como tomar cada fila de la matriz y ponerla una detrás de la otra para formar una única y larguísima fila. La imagen ahora es un vector 1D de 65,536 números (256 * 256).
        # Normalizar (convertir 0-255 a 0.0-1.0)
        vector_caracteristicas = vector_caracteristicas.astype(float) / 255.0
        
        return vector_caracteristicas, imagen.size
        
    except Exception as e:
        print(f"ERROR al procesar {nombre_archivo}: {e}")
        return None, None

def cargar_imagenes():
    """
    Carga las 20 imágenes de ENTRENAMIENTO (1.jpg a 20.jpg)
    Devuelve una matriz donde cada fila es una imagen convertida a números
    """

    print("\n" + "="*50)
    print("CARGANDO IMÁGENES DE ENTRENAMIENTO")
    print("="*50)
    
    
    # Refactor: Usar un tamaño fijo para simplificar en lugar de llamar a la función
    tamaño_objetivo = (256, 256)  # Fijo, no se llama a determinar_tamaño_comun

    if tamaño_objetivo is None:
        return None
    
    todas_las_caracteristicas = []
    
    for i in range(1, 21): 
        nombre_archivo = f"{i}.jpg"
        
        # Convertir imagen a vector de números
        vector, tamaño_original = cargar_y_procesar_imagen(nombre_archivo, tamaño_objetivo)
        
        if vector is not None:
            todas_las_caracteristicas.append(vector)

            persona = "Persona 1" if i <= 10 else "Persona 2"
            
        else:
            print(f"ERROR: No se pudo cargar {nombre_archivo}")
            return None
    
    # Verificar que tenemos exactamente 20 imágenes
    if len(todas_las_caracteristicas) != 20:
        print(f"ERROR: Se esperaban 20 imágenes, pero se cargaron {len(todas_las_caracteristicas)}")
        return None
    
    # Convertir lista a matriz numpy para PCA
    # with np.printoptions(threshold=sys.maxsize):
    #    print(todas_las_caracteristicas)
    return np.array(todas_las_caracteristicas)

def cargar_imagen_prueba():
    """
    Carga la imagen 21.jpg (la que queremos clasificar)
    """
    # Refactor: Usar un tamaño fijo para simplificar en lugar de llamar a la función
    tamaño_objetivo = (256, 256)  # Fijo, debe coincidir con las de entrenamiento
    
    vector, tamaño_original = cargar_y_procesar_imagen("21.jpg", tamaño_objetivo)
    
    if vector is not None:
        print(f"21.jpg cargada correctamente")
        return vector
    else:
        print("ERROR: No se pudo cargar 21.jpg")
        return None
    
def aplicar_pca(A):
    """
    Aplica PCA (Análisis de2 Componentes Principales) a las imágenes
    """
    print(f"\n" + "="*50)
    print("APLICANDO PCA")
    print("="*50)
    

    # PASO 1: Calcular vector de medias (μ) - igual que tu código original
    mu = np.mean(A, axis=0)  # Promedio de cada píxel en todas las imágenes
    
    # PASO 2: Centrar los datos (Xc = A - μ) - igual que tu código original
    Xc = A - mu  # Cada imagen menos la imagen promedio

    # Imprimir Xc
    print(f"\nDatos centrados (Xc = A - μ):")
    print(f"   Dimensiones de Xc: {Xc.shape}")
    print(f"   Primeros 3 valores de la primera imagen centrada: {Xc[0, :3]}")  
    
    # SVD (Singular Value Decomposition): Xc = U * S * V^T
    U, s, Vt = np.linalg.svd(Xc.T, full_matrices=False)
    
    # Convertir valores singulares a eigenvalores 
    n = A.shape[0]
    lambdas = (s**2) / (n-1)  # Estos son los eigenvalores (λ)
    V = U  # Estos son los eigenvectores
    
    # PASO 4: Mostrar resultados 
    print(f"\nValores propios (λ) más importantes:")
    num_mostrar = min(10, len(lambdas))
    for i in range(num_mostrar):
        print(f"   λ{i+1}: {lambdas[i]:.6f}")
    
    # PASO 5: Varianza explicada
    total_varianza = np.sum(lambdas)
    varianza_acumulada = 0.0
    k = 0
    umbral = 0.95  # Objetivo: Explicar el 95% de la varianza

    print(f"\nBuscando el número de componentes 'k' o el rango 'r' para alcanzar el {umbral:.0%} de varianza...")
    
    for i, lam in enumerate(lambdas):
    # Suma la varianza del componente actual
        varianza_acumulada += lam / total_varianza
        
        # Comprueba si hemos alcanzado el umbral
        if varianza_acumulada >= umbral:
            k = i + 1  # Guardamos el número de componentes (índice + 1)
            print(f"   Se necesitan {k} componentes para explicar el {varianza_acumulada:.2%} de la varianza.")
            break # Detenemos el bucle una vez que encontramos k

    # Si por alguna razón no se alcanza el umbral, k será 0. Podemos manejarlo
    if k == 0:
        k = len(lambdas) # Usar todos los componentes si no se alcanza
        print("Advertencia: No se alcanzó el umbral, se usarán todos los componentes.")

    
    return mu, Xc, V[:, :k] # Retorna los primeros 'k' componentes

def proyectar_al_pca(Xc, Vk):
    """
    Proyecta todas las imágenes centradas (Xc) al subespacio PCA de k dimensiones.

    Esta función toma la matriz de imágenes (donde cada imagen es un vector de píxeles)
    y la transforma en una matriz de proyecciones. Cada imagen, que originalmente
    requería un gran número de píxeles para ser descrita, ahora será representada
    por un vector mucho más pequeño de 'k' coordenadas.

    La proyección se realiza de forma vectorizada (sin bucles) mediante una
    única y eficiente multiplicación de matrices.

    La operación matemática es: Z = Xc · Vk

    Args:
        Xc (np.array): La matriz de datos centrados (imágenes - cara promedio).
                    Su forma es (num_imagenes, num_pixeles).
        Vk (np.array): La matriz que contiene los k eigenvectores (eigenfaces)
                    más importantes. Su forma es (num_pixeles, k).

    Returns:
        np.array: La matriz de proyecciones Z, con forma (num_imagenes, k).
                Cada fila es el nuevo vector de características de una imagen en el
                "espacio de caras".
    """
    print(f"\n" + "="*50)
    print(f"PROYECTANDO AL SUBESPACIO DE {Vk.shape[1]} COMPONENTES")
    print("="*50)

    Z = np.dot(Xc, Vk)

    print(f"Se han proyectado {Z.shape[0]} imágenes a un espacio de {Z.shape[1]} dimensiones.")
    
    # Opcional: Imprimir las primeras proyecciones para verificar
    for i in range(min(5, len(Z))):
        persona = "Persona 1" if i < 10 else "Persona 2"
        # np.array2string formatea el vector para que se vea bien
        proyeccion_str = np.array2string(Z[i], precision=4, floatmode='fixed')
        print(f"   Proyección Imagen {i+1:2d} ({persona}): {proyeccion_str}")
        
    return Z

def reconstruir_imagen(z, Vk, mu):
    """
    Reconstruye una imagen a partir de su proyección en el espacio PCA.
    """
    Xc_reconstruido = np.dot(Vk, z)
    X_reconstruido = Xc_reconstruido + mu

    # Imprimir el X_reconstruido
    print(f"\nReconstrucción de imagen desde proyección z:")
    print(f"   Dimensiones de la imagen reconstruida: {X_reconstruido.shape}")
    print(f"   Primeros 3 valores de la imagen reconstruida: {X_reconstruido[:3]}") 

    return X_reconstruido

def calcular_error_de_reconstruccion(x_test, mu, Vk):
    print("\n" + "="*50)
    print("CALCULANDO ERROR DE RECONSTRUCCIÓN ")
    print("="*50)

    # 1. Centrar la imagen de prueba
    print("   Paso 1: Centrando la imagen de prueba...")
    x_test_centrado = x_test - mu

     # 2. Proyectar la imagen al espacio de k dimensiones para obtener 'z_test'
    #    Para proyectar un solo vector, se multiplica por la transpuesta de Vk.
    print("   Paso 2: Proyectando la imagen al espacio PCA...")
    z_test = np.dot(Vk.T, x_test_centrado)
    print(f"      Proyección obtenida (vector de {len(z_test)} dimensiones).")

    # 3. Reconstruir la imagen a partir de su proyección
    print("   Paso 3: Reconstruyendo la imagen desde la proyección...")
    x_reconstruido = reconstruir_imagen(z_test, Vk, mu)
    print("      Imagen reconstruida.")
     # 4. Calcular el error (norma de la diferencia) entre la original y la reconstruida
    print("   Paso 4: Calculando el error...")
    error = np.linalg.norm(x_test - x_reconstruido)
    print(f"      Error de reconstrucción: {error:.4f}")
    
    return error

def evaluar_sistema(A, mu, Vk):
    """
    Realiza la evaluación completa del sistema:
    1. Calcula el umbral T a partir de los errores de las imágenes de entrenamiento.
    2. Evalúa una imagen de prueba contra ese umbral.
    """
    print("\n" + "="*60)
    print("FASE DE EVALUACIÓN DEL SISTEMA Y UMBRAL DE DECISIÓN (T)")
    print("="*60)

    # --- PASO 1: Calcular el umbral T con los datos de entrenamiento ---
    print("\nCalculando errores de reconstrucción para las imágenes de entrenamiento...")
    errores_entrenamiento = []
    # En el arreglo guardamos todos los errores de las imágenes de entrenamiento, para luego definir el umbral T como el error máximo
    for i, x_entrenamiento in enumerate(A):
        # Usamos la misma función de cálculo de error para cada imagen de entrenamiento
        error = calcular_error_de_reconstruccion(x_entrenamiento, mu, Vk)
        errores_entrenamiento.append(error)
        print(f"   Error para la imagen de entrenamiento {i+1}.jpg: {error:.4f}")

    # Definimos el umbral T como el error máximo encontrado en el entrenamiento 
    umbral_T = max(errores_entrenamiento)
    print(f"\nEl error máximo en el set de entrenamiento fue: {umbral_T:.4f}")
    print(f"UMBRAL DE DECISIÓN (T) ESTABLECIDO EN: {umbral_T:.4f}")

    # --- PASO 2: Evaluar la imagen de prueba ---
    print("\nCargando imagen de prueba para evaluación final...")
    x_prueba = cargar_imagen_prueba() # Carga "21.jpg"

    if x_prueba is None:
        print("No se puede continuar sin la imagen de prueba.")
        return

    error_prueba = calcular_error_de_reconstruccion(x_prueba, mu, Vk)

    # --- PASO 3: Dar el veredicto final ---
    print("\n" + "="*50)
    print("VEREDICTO FINAL")
    print("="*50)
    print(f"   Error de reconstrucción de la imagen de prueba: {error_prueba:.4f}")
    print(f"   Umbral de decisión (T): {umbral_T:.4f}")

    if error_prueba <= umbral_T:
        print("\n   >> RESULTADO: Rostro RECONOCIDO.")
        print("   Explicación: El error está por debajo del umbral, lo que significa que la cara se ajusta bien al modelo aprendido.")
    else:
        print("\n   >> RESULTADO: Rostro DESCONOCIDO.")
        print("   Explicación: El error supera el umbral, lo que indica que la cara es significativamente diferente a las caras del entrenamiento.")



def main():
    """
    Función principal - ejecuta todo el proceso paso a paso
    """
    
    # PASO 1: Cargar imágenes de entrenamiento (1.jpg a 20.jpg)
    A = cargar_imagenes()
    if A is None:
        print("\nERROR CRÍTICO: No se pudieron cargar las imágenes de entrenamiento")
        return
    
    print(f"Total de imágenes de entrenamiento: {A.shape[0]}")
    
    # PASO 2: Aplicar PCA
    mu, Xc, Vk = aplicar_pca(A)

    # PASO 3: Proyectar datos al primer componente
    Z = proyectar_al_pca(Xc, Vk) # Z contendrá la matriz de 20x13   proyecciones
    
     # --- FASE DE EVALUACIÓN ---
    evaluar_sistema(A, mu, Vk)

    print(f"\n" + "="*60)
    print("PROCESO COMPLETADO")
    print("="*60)

# EJECUTAR EL PROGRAMA
if __name__ == "__main__":
    main()