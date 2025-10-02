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
        
        # Convertir imagen a array de numpy
        img_array = np.array(imagen)  
        vector_caracteristicas = img_array.flatten()  # Convierte de matriz a vector
        
        # Normalizar (convertir 0-255 a 0-1)
        vector_caracteristicas = vector_caracteristicas.astype(float) / 255.0
        
        return vector_caracteristicas, imagen.size
        
    except Exception as e:
        print(f"ERROR al procesar {nombre_archivo}: {e}")
        return None, None

def determinar_tamaño_comun():
    print("Revisando tamaños de las imágenes existentes...")
    
    tamaños_encontrados = []
    
    for i in range(1, 21):
        nombre = f"{i}.jpg"
        try:
            img = Image.open(nombre)
            tamaños_encontrados.append(img.size)
        except:
            print(f"  {nombre}: No encontrada")
            continue
    
    if not tamaños_encontrados:
        print("No se encontraron imágenes")
        return None
    
    tamaño_estandar = (256, 256)  
    
    return tamaño_estandar

def cargar_imagenes():
    """
    Carga las 20 imágenes de ENTRENAMIENTO (1.jpg a 20.jpg)
    Devuelve una matriz donde cada fila es una imagen convertida a números
    """
    print("\n" + "="*50)
    print("CARGANDO IMÁGENES DE ENTRENAMIENTO")
    print("="*50)
    
    tamaño_objetivo = determinar_tamaño_comun()
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
    
    # Usar el mismo tamaño que las imágenes de entrenamiento
    tamaño_objetivo = determinar_tamaño_comun()
    
    vector, tamaño_original = cargar_y_procesar_imagen("21.jpg", tamaño_objetivo)
    
    if vector is not None:
        print(f"21.jpg cargada correctamente")
        return vector
    else:
        print("ERROR: No se pudo cargar 21.jpg")
        return None

def mostrar_estadisticas_basicas(A):
    """
    Muestra información básica sobre las imágenes cargadas
    """
    
    print(f"Total de imágenes de entrenamiento: {A.shape[0]}")
    
    # Analizar diferencias entre las dos personas
    grupo_persona1 = A[:10]   # Primeras 10 imágenes (filas 0-9)
    grupo_persona2 = A[10:]   # Últimas 10 imágenes (filas 10-19)

def aplicar_pca(A):
    """
    Aplica PCA (Análisis de Componentes Principales) a las imágenes
    """
    print(f"\n" + "="*50)
    print("APLICANDO PCA")
    print("="*50)
    
    # PASO 1: Calcular vector de medias (μ) - igual que tu código original
    mu = np.mean(A, axis=0)  # Promedio de cada píxel en todas las imágenes
    
    # PASO 2: Centrar los datos (Xc = A - μ) - igual que tu código original
    Xc = A - mu  # Cada imagen menos la imagen promedio
    
    # SVD (Singular Value Decomposition): Xc = U * S * V^T
    U, s, Vt = np.linalg.svd(Xc.T, full_matrices=False)
    
    # Convertir valores singulares a eigenvalores 
    n = A.shape[0]
    lambdas = (s**2) / (n-1)  # Estos son los eigenvalores (λ)
    V = U  # Estos son los eigenvectores
    
    # PASO 4: Mostrar resultados (igual que tu código original)
    print(f"\nValores propios (λ) más importantes:")
    num_mostrar = min(10, len(lambdas))
    for i in range(num_mostrar):
        print(f"   λ{i+1}: {lambdas[i]:.6f}")
    
    # PASO 5: Varianza explicada
    total_varianza = np.sum(lambdas)
    varianza_acumulada = 0
    for i in range(min(5, len(lambdas))):
        porcentaje = (lambdas[i]/total_varianza) * 100
        varianza_acumulada += porcentaje
        print(f"   Componente {i+1}: {porcentaje:.2f}%")
    
    print(f"Primeros 5 componentes explican: {varianza_acumulada:.2f}% de la varianza")
    
    return mu, Xc, V[:, 0]  # media, datos centrados, primer eigenvector

def proyectar_al_pca(Xc, v1):
    """
    Proyecta todas las imágenes al primer componente principal
    
    z = v1^T * (imagen - μ)
    """
    print(f"\n" + "="*50)
    print("PROYECTANDO AL PRIMER COMPONENTE PRINCIPAL")
    print("="*50)
    
    print(f"Usando el primer componente principal (v1)")
    print(f"Dimensión del vector v1: {len(v1):,}")
    
    z_values = []  # Lista para guardar todas las proyecciones z
    
    print(f"\nCalculando z = v1^T * (imagen_i - μ) para cada imagen:")
    
    # Para cada imagen de entrenamiento, calcular su proyección z
    for i in range(len(Xc)):
        z = np.dot(v1, Xc[i])  # Producto punto entre v1 y la imagen centrada
        z_values.append(z)
        
        # Determinar a qué persona pertenece
        persona = "Persona 1" if i < 10 else "Persona 2"
        print(f"   Imagen {i+1:2d}.jpg: z = {z:8.4f} ({persona})")
    
    print(f"\n{len(z_values)} proyecciones calculadas")
    
    return z_values

def clasificar_con_vecino_cercano(mu, v1, z_values):
    """
    Clasifica la imagen 21.jpg usando el algoritmo 1-NN (vecino más cercano)
    """
    print(f"\n" + "="*60)
    print("CLASIFICACIÓN CON ALGORITMO 1-NN (VECINO MÁS CERCANO)")
    print("="*60)
    
    # PASO 1: Cargar y procesar la imagen de prueba
    print("Procesando imagen de prueba...")
    x_test = cargar_imagen_prueba()
    if x_test is None:
        print("No se puede continuar sin la imagen 21.jpg")
        return
    
    # PASO 2: Centrar la imagen de prueba (igual que tu código original)
    print("\nCentrando imagen de prueba...")
    x_test_centered = x_test - mu  
    print(f"Imagen 21.jpg centrada")
    
    # PASO 3: Proyectar al PCA 
    print("\nHaciendo las proyecciones...")
    z_test = np.dot(v1, x_test_centered)  # z = v1^T * (imagen_prueba - μ)
    print(f"Proyección z de imagen 21.jpg: {z_test:.4f}")
    
    # PASO 4: Calcular distancias 
    print(f"\nCalculando distancias a todas las imágenes de entrenamiento:")
    print(f"(Usando distancia |z_prueba - z_entrenamiento|)")
    
    distancias = []
    
    for i, z_entrenamiento in enumerate(z_values):
        distancia = abs(z_test - z_entrenamiento)
        distancias.append(distancia)
        
        persona = "Persona 1" if i < 10 else "Persona 2"
        print(f"   Distancia a {i+1:2d}.jpg ({persona}): {distancia:.4f}")
    
    # PASO 5: Encontrar vecino más cercano (igual que tu código original)
    print(f"\nEncontrando vecino más cercano...")
    indice_minimo = np.argmin(distancias)  # Índice de la distancia mínima
    distancia_minima = distancias[indice_minimo]
    
    # Determinar clasificación
    clasificacion = "Persona 1" if indice_minimo < 10 else "Persona 2"
    imagen_mas_cercana = f"{indice_minimo+1}.jpg"
    
    # MOSTRAR RESULTADO FINAL
    print(f"\n" + "="*50)
    print("RESULTADO DE LA CLASIFICACIÓN")
    print("="*51)
    print(f"Vecino más cercano: {imagen_mas_cercana}")
    print(f"Distancia mínima: {distancia_minima:.4f}")
    print(f"CLASIFICACIÓN: La imagen 21.jpg es de {clasificacion}")
    
    # Análisis adicional para dar más confianza
    print(f"\nAnálisis adicional:")
    distancia_promedio_p1 = np.mean([distancias[i] for i in range(10)])
    distancia_promedio_p2 = np.mean([distancias[i] for i in range(10, 20)])
    
    print(f"   Distancia promedio a Persona 1: {distancia_promedio_p1:.4f}")
    print(f"   Distancia promedio a Persona 2: {distancia_promedio_p2:.4f}")
    
    diferencia = abs(distancia_promedio_p1 - distancia_promedio_p2)
    mejor_grupo = "Persona 1" if distancia_promedio_p1 < distancia_promedio_p2 else "Persona 2"
    
    # Nivel de confianza
    if diferencia > 0.02:
        confianza = "ALTA"
    elif diferencia > 0.01:
        confianza = "MEDIA"
    else:
        confianza = "BAJA"
    
    print(f"Confirmación: Más similar a {mejor_grupo}")
    print(f"Nivel de confianza: {confianza}")

def main():
    """
    Función principal - ejecuta todo el proceso paso a paso
    """
    
    # PASO 1: Cargar imágenes de entrenamiento (1.jpg a 20.jpg)
    A = cargar_imagenes()
    if A is None:
        print("\nERROR CRÍTICO: No se pudieron cargar las imágenes de entrenamiento")
        return
    
    # PASO 2: Mostrar estadísticas
    mostrar_estadisticas_basicas(A)
    
    # PASO 3: Aplicar PCA
    mu, Xc, v1 = aplicar_pca(A)
    
    # PASO 4: Proyectar datos al primer componente
    z_values = proyectar_al_pca(Xc, v1)
    
    # PASO 5: Clasificar imagen de prueba
    clasificar_con_vecino_cercano(mu, v1, z_values)
    
    print(f"\n" + "="*60)
    print("PROCESO COMPLETADO EXITOSAMENTE")
    print("="*61)

# EJECUTAR EL PROGRAMA
if __name__ == "__main__":
    main()