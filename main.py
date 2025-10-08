from flask import Flask, render_template, request, redirect, url_for
#import reconocimiento-facial.py

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
            print("--- Archivos Recibidos ---")
            for file in files:
                print(f"Nombre del archivo: {file.filename}")
                # Si quieres guardar el archivo: file.save('ruta/destino/' + file.filename)
            print("--------------------------")

            msg = f"Formulario Exitoso: Archivos procesados en el backend. Se recibieron {len(files)} archivos."
    
        return render_template("program.html", msg=msg)

    return redirect(url_for('index', msg=msg)) 


