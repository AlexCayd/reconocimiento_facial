import os
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for

import reconocimiento_facial as rf

app = Flask(__name__)

# Asegurarse de que las carpetas necesarias existan al iniciar la app
os.makedirs('dataset', exist_ok=True)
os.makedirs('imagen_prueba', exist_ok=True)


@app.route("/")
def index():
    msg = request.args.get('msg', None)
    return render_template("index.html", msg=msg)

@app.route("/analizar", methods=['POST'])
def procesar():
    dataset_files = request.files.getlist("dataset_images")
    test_file = request.files.get("test_image")

    if len(dataset_files) < 5 or not test_file or test_file.filename == '':
        return redirect(url_for('index', msg="Error: Asegúrate de subir al menos 5 imágenes de dataset y 1 de prueba."))

    # --- GUARDADO DE IMÁGENES SUBIDAS ---
    dataset_dir = 'dataset'
    test_dir = 'imagen_prueba'
    uploads_dir = 'static/uploads' # NUEVO: Definir la carpeta de subidas estáticas

    # Limpiar y guardar imágenes del dataset
    for f in os.listdir(dataset_dir): os.remove(os.path.join(dataset_dir, f))
    for file in dataset_files: file.save(os.path.join(dataset_dir, file.filename))

    # Limpiar y guardar imagen de prueba para el script de análisis
    for f in os.listdir(test_dir): os.remove(os.path.join(test_dir, f))
    test_file.save(os.path.join(test_dir, "1.jpg"))
    test_file.stream.seek(0) # Reiniciar el puntero del archivo para volver a guardarlo

    # Limpiar y guardar imagen de prueba para MOSTRARLA en el HTML
    for f in os.listdir(uploads_dir): os.remove(os.path.join(uploads_dir, f))
    test_image_filename = "test_image.jpg"
    test_file.save(os.path.join(uploads_dir, test_image_filename))

    # --- LLAMADA AL MOTOR DE ANÁLISIS ---
    results, logs = rf.ejecutar_analisis_completo()

    if results is None:
        return render_template("index.html", logs=logs, msg="Ocurrió un error durante el análisis.")

    # --- PREPARACIÓN FINAL DE DATOS PARA EL HTML ---
    # NUEVO: Añadir las rutas a las imágenes subidas para que el HTML las encuentre
    results['test_image_path'] = os.path.join(uploads_dir, test_image_filename)

    # Guardar una copia de la primera imagen del dataset para mostrarla en el HTML
    first_dataset_filename = os.listdir(dataset_dir)[0]
    first_dataset_image = Image.open(os.path.join(dataset_dir, first_dataset_filename))
    first_dataset_image.save(os.path.join(uploads_dir, "dataset_1.jpg"))
    results['first_dataset_image_path'] = os.path.join(uploads_dir, "dataset_1.jpg")

    # --- PASAR TODO A LA PLANTILLA ---
    return render_template("index.html", results=results, logs=logs)


@app.route("/restablecer", methods=['POST'])
def restablecer():
    msg = "Datos restablecidos correctamente."
    for folder in ['dataset', 'imagen_prueba', 'static/results', 'static/uploads']:
        if os.path.exists(folder):
            for f in os.listdir(folder):
                ruta = os.path.join(folder, f)
                if os.path.isfile(ruta):
                    os.remove(ruta)
    return redirect(url_for('index', msg=msg))


if __name__ == '__main__':
    app.run(debug=True)