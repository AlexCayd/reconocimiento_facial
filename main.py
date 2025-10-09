from importlib.metadata import files
import os
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for
import reconocimiento_facial.py

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/success", methods=['GET', 'POST'])
def procesar():
    msg = ''
    if request.method == "POST":
        # Obtener archivos, metodo especial de flask
        files = request.files.getlist("files") 

        # Verificacion backend de cantidad de archivos para el dataset
        if len(files) <= 4:
            print("No hay suficientes archivos para ejecutar el programa")
            msg = "No hay suficientes archivos para ejecutar el programa."
        else:
            # Limpiar imagenes 
            
            #ejecutar_reconocimiento_facial(files)
            os.makedirs('dataset', exist_ok=True)
            for idx, file in enumerate(files, start=1):
                filename = file.filename.lower()
                if filename.endswith('.jpg') or filename.endswith('.jpeg'):
                    file.save(os.path.join('dataset', f"{idx}.jpg"))
                else:
                    img = Image.open(file.stream)
                    rgb_img = img.convert('RGB')
                    rgb_img.save(os.path.join('dataset', f"{idx}.jpg"), format='JPEG')

            print("--- Archivos Recibidos ---")
            for file in files:
                print(f"Nombre del archivo: {file.filename}")                
            print("--------------------------")

            msg = f"Formulario Exitoso: Archivos procesados en el backend. Se recibieron {len(files)} archivos."
    
        return render_template("program.html", msg=msg)

    return redirect(url_for('index', msg=msg)) 

    

@app.route("/reset", methods=['GET', 'POST'])
def restablecer():
    msg = "Datos restablecidos correctamente."
    # Vaciar carpeta dataset
    for f in os.listdir('dataset'):
        ruta = os.path.join('dataset', f)
        if os.path.isfile(ruta):
            os.remove(ruta)
    # Vaciar carpeta imagen_prueba
    for f in os.listdir('imagen_prueba'):
        ruta = os.path.join('imagen_prueba', f)
        if os.path.isfile(ruta):
            os.remove(ruta)
    return redirect(url_for('index', msg=msg))

