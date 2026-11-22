#Grupo ITJ
#Integrantes: Di Niro Ismael, Domínguez Tomás, Ottaviano Jaime.

#En este archivo desarrollamos el Trabajo Práctico 2, donde principalmente entrenamos modelos de predicción.

#Para correr el código deberás completar la variable "carpeta" con la dirección en donde tengas los archivos.
#Se recomienda recorrer las celdas en orden.
#%% Sección de imports 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict

#%% 
#Fijamos la semilla para obtener consistencia en los modelos
semilla = 89 
#%%
#Carga de datos
carpeta = "Ruta donde estén tus archivos"
clases = pd.read_csv(carpeta +"clases.csv")
evaluacion = pd.read_csv(carpeta +"evaluacion.csv")
imagenes = pd.read_csv(carpeta +"imagenes.csv")
#%% 
#Código para visualizar las imágenes
#Plot imagen 
for i in range(80,100) : 
    img = np.array(imagenes.drop(columns = ['label']).iloc[i]).reshape((28,28)) 
    plt.imshow(img, cmap='gray') 
    plt.show() 

############
#%% Ejercicio 1.a) 
#Justificación 
variabilidad = imagenes.drop(columns="label")
variabilidad = variabilidad.var().sort_values()
#Podemos ver cuáles son los píxeles que menos varían.
#Se podrían descartar los primeros veinte.

#%% Ejercicio 2)
#########################################
#%%Ejercicio 2.a) 


#Separamos del df original (imagenes.csv) solo las clases 2 y 9 
df_label_2y9 = imagenes[(imagenes['label'] == 2) | (imagenes['label'] == 9)]


#Contamos cuántas muestras se tienen de cada clase 
df_label_2y9['label'].value_counts()
#%% Ejercicio 2.c)

X = df_label_2y9.drop(columns=['label'])
y = df_label_2y9['label']

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2,
                                                    stratify=y, 
                                                    random_state=semilla)

#%% Ejercicio 2.d) 
# Decidimos elegir 4 atributos para usar en nuestro modelo KNN,
# lo que vamos a ir variando son las posiciones de los píxeles.
# Además fijamos como K = 3 a la hora de elegir la cantidad de vecinos a comparar

# Acá visualizamos las imágenes para poder elegir cuáles píxeles utilizar

X = df_label_2y9.drop(columns=['label'])
y = df_label_2y9['label']

# Plot imagen
for i in range(0, 20):
    img = np.array(X.iloc[i]).reshape((28, 28))

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img, cmap='gray')

    ax.set_xticks(np.arange(28))
    ax.set_yticks(np.arange(28))
    ax.set_xticks(np.arange(-0.5, 28, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 28, 1), minor=True)
    ax.grid(which='minor', color='red', linewidth=0.5)

    plt.show()
############ 
#%%% KNN con k = 3 y 4 atributos

#Elegimos 4 píxeles de arriba y al centro de la imagen
Xtrain =  X_train[[ "pixel"+str(28*1 + 12),
                   "pixel"+str(28*1 + 13), 
                   "pixel"+str(28*2 + 12),
                   "pixel"+str(28*2 + 13)]] 

Xtest = X_test[[ "pixel"+str(28*1 + 12),
                   "pixel"+str(28*1 + 13), 
                   "pixel"+str(28*2 + 12),
                   "pixel"+str(28*2 + 13)]]

#creamos el modelo
modelo = KNeighborsClassifier(n_neighbors = 3)
modelo.fit(Xtrain,y_train)

#hacemos la predicción con los datos guardados para test
prediccion = modelo.predict(Xtest)

#Medimos la exactitud
exactitud = accuracy_score(y_test, prediccion)
print("Accuracy con k = 3 y 4 atributos de arriba en el centro\n-->", exactitud)

#%%% KNN con k = 3 y 4 atributos

#Elegimos 4 píxeles del centro de la imagen
Xtrain = X_train[["pixel"+str(28*13 + 13), 
                  "pixel"+str(28*13 + 14), 
                  "pixel"+str(28*14 + 13), 
                  "pixel"+str(28*14+ 14)]]

Xtest = X_test[["pixel"+str(28*13 + 13), 
                  "pixel"+str(28*13 + 14), 
                  "pixel"+str(28*14 + 13), 
                  "pixel"+str(28*14+ 14)]]

#creamos el modelo
modelo = KNeighborsClassifier(n_neighbors = 3)
modelo.fit(Xtrain,y_train)

#hacemos la predicción con los datos guardados para test
prediccion = modelo.predict(Xtest)

#Medimos la exactitud
exactitud = accuracy_score(y_test, prediccion)
print("Accuracy con k = 3 y 4 atributos en el centro\n-->", exactitud)
#%%% KNN con k = 3 y 2 atributos 

#acá elegimos 2 atributos cercanos a los que elegimos anteriormente para comparar accuracy
Xtrain = X_train[["pixel"+str(28*13 + 13), 
                  "pixel"+str(28*13 + 14)]]

Xtest = X_test[["pixel"+str(28*13 + 13), 
                  "pixel"+str(28*13 + 14)]]

#creamos el modelo
modelo = KNeighborsClassifier(n_neighbors = 3)
modelo.fit(Xtrain,y_train)

#hacemos la predicción con los datos guardados para test
prediccion = modelo.predict(Xtest)

#Medimos la exactitud
exactitud = accuracy_score(y_test, prediccion)
print("Accuracy con k = 3 y 2 atributos en el centro\n-->", exactitud)
#%%% KNN con k = 3 y 2 atributos
#ídem que antes solo que ahora al centro y arriba de la imagen
Xtrain =  X_train[[ "pixel"+str(28*1 + 12),
                   "pixel"+str(28*1 + 13)]] 

Xtest = X_test[[ "pixel"+str(28*1 + 12),
                   "pixel"+str(28*1 + 13)]]

#creamos el modelo
modelo = KNeighborsClassifier(n_neighbors = 3)
modelo.fit(Xtrain,y_train)

#hacemos la predicción con los datos guardados para test
prediccion = modelo.predict(Xtest)

#Medimos la exactitud
exactitud = accuracy_score(y_test, prediccion)
print("Accuracy con k = 3 y 2 atributos de arriba en el centro\n-->", exactitud)

#%%% KNN con k = 3 y 4 atributos
# Acá lo que decidimos fue elegir cuatro píxeles al azar, para medir cómo cambia el accuracy

# Fijamos semilla para las funciones de np.random
np.random.seed(semilla)

atributos = ["pixel" + str(np.random.randint(1, 785))
             for i in range(4)]
Xtrain =  X_train[atributos] 

Xtest = X_test[atributos]

#creamos el modelo
modelo = KNeighborsClassifier(n_neighbors = 3)
modelo.fit(Xtrain,y_train)

#hacemos la predicción con los datos guardados para test
prediccion = modelo.predict(Xtest)

#Medimos la exactitud
exactitud = accuracy_score(y_test, prediccion)
print("Accuracy con k = 3 y 4 atributos elegidos al al azar\n-->", exactitud)

#%% Ejercicio 2.d) 
# Acá lo que hacemos es ir variando los k en el modelo KNN 
# y además ir variando la cantidad de atributos
#%%% KNN con k = 5 con 4 atributos

Xtrain =  X_train[[ "pixel"+str(28*1 + 12),
                   "pixel"+str(28*1 + 13), 
                   "pixel"+str(28*2 + 12),
                   "pixel"+str(28*2 + 13)]] 
    
Xtest = X_test[[ "pixel"+str(28*1 + 12),
                   "pixel"+str(28*1 + 13), 
                   "pixel"+str(28*2 + 12),
                   "pixel"+str(28*2 + 13)]]

modelo = KNeighborsClassifier(n_neighbors = 5)
modelo.fit(Xtrain,y_train)

prediccion = modelo.predict(Xtest)

exactitud = accuracy_score(y_test, prediccion)
print("Accuracy con k = 5 -->", exactitud)

#%%% KNN con k = 4 con 4 atributos

Xtrain =  X_train[[ "pixel"+str(28*1 + 12),
                   "pixel"+str(28*1 + 13), 
                   "pixel"+str(28*2 + 12),
                   "pixel"+str(28*2 + 13)]] 
    
Xtest = X_test[[ "pixel"+str(28*1 + 12),
                   "pixel"+str(28*1 + 13), 
                   "pixel"+str(28*2 + 12),
                   "pixel"+str(28*2 + 13)]]

modelo = KNeighborsClassifier(n_neighbors = 4)
modelo.fit(Xtrain,y_train)

prediccion = modelo.predict(Xtest)

exactitud = accuracy_score(y_test, prediccion)
print("Accuracy con k = 4 -->", exactitud)

#%%% KNN con k = 6
Xtrain =  X_train[[ "pixel"+str(28*1 + 12),
                   "pixel"+str(28*1 + 13), 
                   "pixel"+str(28*2 + 12),
                   "pixel"+str(28*2 + 13)]] 
    
Xtest = X_test[[ "pixel"+str(28*1 + 12),
                   "pixel"+str(28*1 + 13), 
                   "pixel"+str(28*2 + 12),
                   "pixel"+str(28*2 + 13)]]

modelo = KNeighborsClassifier(n_neighbors = 6)
modelo.fit(Xtrain,y_train)

prediccion = modelo.predict(Xtest)

exactitud = accuracy_score(y_test, prediccion)
print("Accuracy con k = 6 -->", exactitud)
 
#%%% KNN con k = 5 y 2 atributos
#ídem que antes solo que ahora al centro y arriba de la imagen
Xtrain =  X_train[[ "pixel"+str(28*1 + 12),
                   "pixel"+str(28*1 + 13)]]

Xtest = X_test[[ "pixel"+str(28*1 + 12),
                   "pixel"+str(28*1 + 13)]]

#creamos el modelo
modelo = KNeighborsClassifier(n_neighbors = 5)
modelo.fit(Xtrain,y_train)

#hacemos la predicción con los datos guardados para test
prediccion = modelo.predict(Xtest)

#Medimos la exactitud
exactitud = accuracy_score(y_test, prediccion)
print("Accuracy con k = 5 y 2 atributos de arriba en el centro\n-->", exactitud) 

#%% Ejercicio 3)
################################################################
#%% Ejercicio 3.a)

# Acá separamos para 
imagenes = pd.read_csv(carpeta +"imagenes.csv")
y = imagenes['label']
dev,held_out = train_test_split(imagenes,                               
                                test_size=0.2,
                                random_state=semilla,
                                stratify=y)

#%%% Ejercicio 3.b) 
#Chequeamos si las clases están balanceadas.
imagenes["label"].value_counts().sort_index().plot(kind="bar")

#%%% Ejercicio 3.c) 
X_dev = dev.drop(columns = 'label')
y_dev = dev['label']

# Fijamos a elección el K = 5 y en este "for" usamos criterio de Gini
for i in range(1,11):
    modelo = DecisionTreeClassifier(criterion='gini',
                                    max_depth= i,
                                    random_state=semilla)


    resultados = cross_val_score(modelo,
                                 X_dev,
                                 y_dev,
                                 cv=5,
                                 scoring='accuracy')
    print(f"porcentaje de acierto con {i} de profundidad y criterio de Gini",resultados.mean()*100)

# Fijamos a eleccion el K = 5 y en este "for" usamos criterio de Entropía
for i in range(1,11):
    modelo = DecisionTreeClassifier(criterion='entropy',
                                    max_depth= i,
                                    random_state=semilla)


    resultados = cross_val_score(modelo,
                                 X_dev,
                                 y_dev,
                                 cv=5,
                                 scoring='accuracy')
    print(f"porcentaje de acierto con {i} de profundidad y criterio de Entropia",resultados.mean()*100)

#%% Ejercicio 3.d) 
X_dev = dev.drop(columns = 'label')
y_dev = dev['label']

# Fijamos a elección el K = 5 y en este "for" usamos criterio de Entropía
modelo_final = DecisionTreeClassifier(criterion='entropy',
                                max_depth= 10,
                                random_state=semilla) 
modelo_final.fit(X_dev,y_dev)



X_held_out = held_out.drop(columns = 'label')
y_held_out = held_out['label']


prediccion = modelo_final.predict(X_held_out)
accuracy = accuracy_score(y_held_out, prediccion)
print( "accuracy:", accuracy)

#%% Ejercicio 3.e) 
# Predecimos las imágenes del conjunto de evaluación de la cátedra

X_evaluacion = evaluacion.drop(columns=['id']) 


prediccion_evaluacion = modelo_final.predict(X_evaluacion)
 
df_evaluacion = pd.DataFrame()
df_evaluacion['id'] = evaluacion['id']
df_evaluacion['label'] = prediccion_evaluacion

df_evaluacion.to_csv('TP-02-0.7869-ITJ.csv',index=False, sep=";")
#En caso de querer como separador "," simplemente eliminar ese parámetro.