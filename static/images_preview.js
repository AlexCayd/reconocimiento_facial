
document.getElementById('fileInput').addEventListener('change', function(event) {
    const input = event.target;
    const files = input.files;
    const previewContainer = document.getElementById('previewContainer');
    
    // Limpiar cualquier vista previa anterior
    previewContainer.innerHTML = ''; 

    if ( files && files.length > 0) {
        
        // 2. Iterar sobre la lista de archivos
        for (let i = 0; i < input.files.length; i++) {
            const archivo = input.files[i];

            // Asegurarse de que el archivo sea una imagen
            if (!archivo.type.startsWith('image/')) {
                continue; // Saltar si no es una imagen
            }
            
            // Crear el elemento <img> para la nueva vista previa
            const newPreview = document.createElement('img');
            newPreview.style.maxWidth = '200px'; 
            newPreview.style.maxHeight = '200px';
            newPreview.style.minHeight = '200px';
            newPreview.style.minWidth = '200px';
            newPreview.alt = archivo.name; 

            // Crear un objeto FileReader para leer el archivo
            const lector = new FileReader();

            // 3. Definir la función que se ejecuta cuando el archivo se ha cargado
            lector.onload = (function(imgElement) {
                return function(e) {                
                    imgElement.src = e.target.result;
                    previewContainer.appendChild(imgElement);
                };
            })(newPreview); // Se usa un closure para pasar el elemento img correcto

            lector.readAsDataURL(archivo);
        }

    }
    // Si no hay archivos, el contenedor ya se limpió al principio.
});