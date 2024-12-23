# howIAworks
El proyecto pretende visualizar de forma gráfica el proceso de selección del siguiente token en la IA de texto generativo.


## Pasos de configuración

1. Instalar todas las dependencias y librerías, ejecutar en la terminal: pip install -r requirements.txt

2. Descargar el archivo con los datos de los embendings: glove.42B.300d.txt [Descargar archivo glove]([https://www.luisllamas.es](https://downloads.cs.stanford.edu/nlp/data/glove.42B.300d.zip))

3. Para ejecutar el programa se debe de ir al archivo main: **app.py** y ejecutar el comando *py app.py*

### A tener en cuenta que en el archivo main se específica la cantidad de embendings importados de la librería (por defecto 10k). Este valor se puede modificar tanto como se quiera, cuantos mas valores se importen de la líbreria la respuesta es mas acertada.
