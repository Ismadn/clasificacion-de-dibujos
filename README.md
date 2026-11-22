# Trabajo Práctico: Clasificación de imágenes con modelos de predicción

## Descripción

En este trabajo práctico desarrollamos distintos modelos de clasificación utilizando imágenes de 28 × 28 píxeles. El objetivo principal fue entrenar y evaluar modelos de Machine Learning para predecir las clases correspondientes a cada imagen.

Se trabajó con dos algoritmos de clasificación: K-Nearest Neighbors (KNN) y árboles de decisión.

## Objetivos

- Analizar la variabilidad de los atributos (píxeles) de las imágenes.
- Entrenar modelos KNN utilizando distintas cantidades y selecciones de atributos, y comparar sus resultados.
- Evaluar árboles de decisión utilizando distintos criterios de división y profundidades.
- Comparar el desempeño de los modelos mediante métricas de evaluación.
- Realizar predicciones sobre un conjunto de imágenes de evaluación.

## Tecnologías utilizadas

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Instrucciones para ejecutar el código

### 1. Instalar las dependencias

Para ejecutar el código es necesario tener instaladas las bibliotecas mencionadas anteriormente:

```bash
pip install numpy pandas matplotlib scikit-learn
```

### 2. Preparar los archivos

Descargar los archivos necesarios para ejecutar el proyecto y completar la variable `carpeta` con la ruta del directorio donde se encuentran:

```python
carpeta = "Ruta donde estén tus archivos"
```

Los archivos necesarios son:
- `clases.csv`
- `evaluacion.csv`
- `imagenes.csv`

### 3. Ejecutar el código

Se recomienda ejecutar las celdas en orden, ya que algunas variables y conjuntos de datos se utilizan en las secciones posteriores.

El código está organizado en dos bloques principales:
- **Ejercicio 2:** entrenamiento y evaluación de modelos KNN para clasificar las imágenes correspondientes a las clases 2 y 9.
- **Ejercicio 3:** entrenamiento y evaluación de árboles de decisión, utilizando validación cruzada y distintos criterios de división.

Se recomienda utilizar Spyder para ejecutar el código. Si algún gráfico no se visualiza correctamente en otro entorno, se puede agregar `plt.show()` al final del bloque correspondiente.

## Resultados

Al finalizar la ejecución, se genera el archivo `TP-02-0.7869-ITJ.csv`, que contiene las predicciones realizadas sobre el conjunto de evaluación.

El archivo se exporta utilizando `;` como separador de columnas. Para utilizar otro separador, se puede modificar el parámetro `sep` de la función `to_csv()`.

