# Inteligencia Artificial para la Ciencia de los Datos
# Implementación de clasificadores 
# Dpto. de C. de la Computación e I.A. (Univ. de Sevilla)
# ===================================================================


# --------------------------------------------------------------------------
# Autor(a) del trabajo:
#
# APELLIDOS: Tejero Ruíz
# NOMBRE: David
#
# Segundo componente (si se trata de un grupo):
#
# APELLIDOS: Gil Castilla
# NOMBRE: Miguel
# ----------------------------------------------------------------------------


# *****************************************************************************
# HONESTIDAD ACADÉMICA Y COPIAS: un trabajo práctico es un examen, por lo
# que debe realizarse de manera individual. La discusión y el intercambio de
# información de carácter general con los compañeros se permite, pero NO AL
# NIVEL DE CÓDIGO. Igualmente el remitir código de terceros, OBTENIDO A TRAVÉS
# DE LA RED, DE HERRAMIENTAS DE GENERACIÓN DE CÓDIGO o cualquier otro medio, 
# se considerará plagio. Si tienen dificultades para realizar el ejercicio, 
# consulten con el profesor. En caso de detectarse plagio, supondrá 
# una calificación de cero en la asignatura, para todos los alumnos involucrados. 
# Sin perjuicio de las medidas disciplinarias que se pudieran tomar. 
# *****************************************************************************


# IMPORTANTE: NO CAMBIAR EL NOMBRE NI A ESTE ARCHIVO NI A LAS CLASES, MÉTODOS
# Y ATRIBUTOS QUE SE PIDEN. EN PARTICULAR: NO HACERLO EN UN NOTEBOOK.

# NOTAS: 
# * En este trabajo NO se permite usar Scikit Learn, EXCEPTO algunos métodos 
#   que se indican exprésamente en el enunciado. En particular no se permite
#   usar ningún clasificador de Scikit Learn.  
# * Supondremos que los conjuntos de datos se presentan en forma de arrays numpy, 
#   Se valorará el uso eficinte de numpy. 


# ====================================================
# PARTE I: IMPLEMENTACIÓN DEL CLASIFICADOR NAIVE BAYES
# ====================================================

# Se pide implementar el clasificador Naive Bayes, en su versión categórica
# con suavizado y log probabilidades (descrito en el tema 2, diapositivas 22 a
# 34). En concreto:


# ----------------------------------
# I.1) Implementación de Naive Bayes
# ----------------------------------

# Definir una clase NaiveBayes con la siguiente estructura:

# class NaiveBayes():

#     def __init__(self,k=1):
#                 
#          .....
         
#     def entrena(self,X,y):

#         ......

#     def clasifica_prob(self,ejemplo):

#         ......

#     def clasifica(self,ejemplo):

#         ......

import numpy as np
import math

class NaiveBayes():
    def __init__(self,k=1): 
        self.k = k
        self.clases = None
        self.num_clases = None
        self.prob_clases = None

        self.valores_atributos = None
        self.prob_atributos = None
        self.num_atributos = None
        self.N = None
        self.entrenado = False
        
    def entrena(self,X:np.array,y:np.array):
        # Calculamos el número de clases y el número de atributos
        self.clases = np.unique(y)
        self.num_clases = len(self.clases)
        self.prob_clases = np.zeros(self.num_clases)

        # Obtenemos los valores de cada atributo
        self.valores_atributos = [np.unique(X[:, i]) for i in range(X.shape[1])] # buscamos los valores únicos en cada columna de dataset

        self.num_atributos = X.shape[1] # obteniendo el número de columnas del array numpy X
        # para guardar las probabilidades de cada atributo, al no saber el número de valores que puede tomar cada atributo se guarda en un 
        # diccionario para facilitar el acceso a la hora de clasificar, su acceso más tarde será: self.prob_atributos[clase][atributo][valor]
        self.prob_atributos = {cls: {atr: {val: 0 for val in self.valores_atributos[atr]}  for atr in range(self.num_atributos)} for cls in self.clases}
        # Calculamos el número de ejemplos
        N = len(y)

        # Calculamos la probabilidad de cada clase
        for i in range(self.num_clases):
            # implementamos la fórmula de la diapositiva 26 del tema 2: P(C=c)=n(C=c)/N
            self.prob_clases[i] = np.sum(y == self.clases[i]) / N

        # Calculamos la probabilidad de cada atributo
        for j,clase in enumerate(self.clases):
            for i in range(self.num_atributos):
                for s,val in enumerate(self.valores_atributos[i]):
                    # implementamos la otra fórmula de la diapositiva 26 del tema 2: P(A=v, C=c)=n(A=v, C=c)/n(C=c)
                    self.prob_atributos[clase][i][val] = np.log((np.sum(X[y == self.clases[j]][:,i] == self.valores_atributos[i][s]) + self.k )/ (np.sum(y == self.clases[j])+self.k*len(self.valores_atributos[i])))
        
        self.entrenado = True

    def clasifica_prob(self,ejemplo:np.array):
        if not self.entrenado:
            raise ClasificadorNoEntrenado("El clasificador no ha sido entrenado, llama al método entrena antes de clasificar")
        # Inicializamos el diccionario:
        prob = {clase: np.log(self.prob_clases[i]) for i,clase in enumerate(self.clases)}

        for i in range(self.num_clases): # recorremos las clases
           for j in range(self.num_atributos): # recorremos los atributos
               prob[self.clases[i]] += self.prob_atributos[self.clases[i]][j][ejemplo[j]] # el valor de atributo será el que nos proporciona el ejemplo
        
        # calculamos en primer lugar el total para normalizar el resultado de las probabilidades y que quede entre un valor
        # entre 0 y 1 cada probabilidad sumando 1 entre todas ellas:
        total = np.sum([np.exp(prob[clase]) for clase in prob])

        # pasamos de logprobabilidad a probabilidad:
        for clase in prob:
            prob[clase] = np.exp(prob[clase]) / total
            
        return prob
    
    def clasifica(self,ejemplo:np.array):
        if not self.entrenado:
            raise ClasificadorNoEntrenado("El clasificador no ha sido entrenado, llama al método entrena antes de clasificar")
        
        # Obtenemos las probabilidades de cada clase para el ejemplo
        prob = self.clasifica_prob(ejemplo)

        # Obtenemos la clase con mayor probabilidad
        return max(prob, key=prob.get)
    
# * El constructor recibe como argumento la constante k de suavizado (por
#   defecto 1) 

# * Método entrena, recibe como argumentos dos arrays de numpy, X e y, con los DONE
#   datos y los valores de clasificación respectivamente. Tiene como efecto el
#   entrenamiento del modelo sobre los datos que se proporcionan.  

# * Método clasifica_prob: recibe un ejemplo (en forma de array de numpy) y
#   devuelve una distribución de probabilidades (en forma de diccionario) que
#   a cada clase le asigna la probabilidad que el modelo predice de que el
#   ejemplo pertenezca a esa clase. 

# * Método clasifica: recibe un ejemplo (en forma de array de numpy) y
#   devuelve la clase que el modelo predice para ese ejemplo.   

# Si se llama a los métodos de clasificación antes de entrenar el modelo, se
# debe devolver (con raise) una excepción:

class ClasificadorNoEntrenado(Exception): pass

  
# Ejemplo "jugar al tenis":


# >>> nb_tenis=NaiveBayes(k=0.5)
# >>> nb_tenis.entrena(X_tenis,y_tenis)
# >>> ej_tenis=np.array(['Soleado','Baja','Alta','Fuerte'])
# >>> nb_tenis.clasifica_prob(ej_tenis)
# {'no': 0.7564841498559081, 'si': 0.24351585014409202}
# >>> nb_tenis.clasifica(ej_tenis)
# 'no'

# DESCOMENTAR PARA EJECUTAR #
# from jugar_tenis import *
# nb_tenis=NaiveBayes(k=0.5)
# nb_tenis.entrena(X_tenis,y_tenis)
# print("Entrenando el clasificador con el dataset de jugar al tenis...")
# ej_tenis=np.array(['Soleado','Baja','Alta','Fuerte'])
# print("Clasificando el ejemplo: ",ej_tenis)
# print("Probabilidades de cada clase: ",nb_tenis.clasifica_prob(ej_tenis))
# print("Clase predicha: ",nb_tenis.clasifica(ej_tenis))

# RESULTADOS:
# Probabilidades de cada clase:  {'no': 0.7564841498559081, 'si': 0.24351585014409202}
# Clase predicha:  no
# Como vemos, concuerda con el resultado esperado

# ----------------------------------------------
# I.2) Implementación del cálculo de rendimiento
# ----------------------------------------------

# Definir una función "rendimiento(clasificador,X,y)" que devuelve la
# proporción de ejemplos bien clasificados (accuracy) que obtiene el
# clasificador sobre un conjunto de ejemplos X con clasificación esperada y. 

# Ejemplo:

# >>> rendimiento(nb_tenis,X_tenis,y_tenis)
# 0.9285714285714286

def rendimiento(clasificador,X:np.array,y:np.array):
    # Obtenemos las predicciones para cada ejemplo
    predicciones = [clasificador.clasifica(ejemplo) for ejemplo in X]

    # Calculamos el rendimiento
    return np.sum(predicciones == y) / len(y)

# DESCOMENTAR PARA EJECUTAR # (descomentar también la parte I.1)
# print("Rendimiento del clasificador NB con el conjunto de jugar tenis: ",rendimiento(nb_tenis,X_tenis,y_tenis),"\n \n")

# RESULTADOS:
# Rendimiento del clasificador NB con el conjunto de jugar tenis:  0.9285714285714286
# Vemos que se obtiene el valos esperado

# --------------------------
# I.3) Aplicando Naive Bayes
# --------------------------

# Usando el clasificador implementado, obtener clasificadores con el mejor
# rendimiento posible para los siguientes conjunto de datos:

# - Votos de congresistas US
# - Concesión de prestamos
# - Críticas de películas en IMDB (ver NOTA con instrucciones para obtenerlo)

# En todos los casos, será necesario separar un conjunto de test para dar la
# valoración final de los clasificadores obtenidos. Si fuera necesario, se permite usar
# train_test_split de Scikit Learn, para separar el conjunto de test y/o
# validación. Ajustar también el valor del parámetro de suavizado k. 

# Mostrar el proceso realizado en cada caso, y los rendimientos obtenidos. 


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# NOTA:
# INSTRUCCIONES PARA OBTENER EL CONJUNTO DE DATOS IMDB A USAR EN EL TRABAJO

# Este conjunto de datos ya se ha usado en un ejercicio del tema de modelos 
# probabilísticos. Los textos en bruto y comprimidos están en aclImdb.tar.gz, 
# que se ha de descomprimir previamente (NOTA: debido a la gran cantidad de archivos
# que aparecen al descomprimir, se aconseja pausar la sincronización si se está conectado
# a algún servicio en la nube).

# NO USAR TODO EL CONJUNTO: extraer, usando random.sample, 
# 2000 críticas en el conjunto de entrenamiento y 400 del conjunto de test. 
# Usar por ejemplo la siguiente secuencia de instrucciones, para extraer los textos:


# >>> import random as rd
# >>> from sklearn.datasets import load_files
# >>> reviews_train = load_files("data/aclImdb/train/")
# >>> muestra_entr=random.sample(list(zip(reviews_train.data,
#                                     reviews_train.target)),k=2000)
# >>> text_train=[d[0] for d in muestra_entr]
# >>> text_train = [doc.replace(b"<br />", b" ") for doc in text_train]
# >>> yimdb_train=np.array([d[1] for d in muestra_entr])
# >>> reviews_test = load_files("data/aclImdb/test/")
# >>> muestra_test=random.sample(list(zip(reviews_test.data,
#                                         reviews_test.target)),k=400)
# >>> text_test=[d[0] for d in muestra_test]
# >>> text_test = [doc.replace(b"<br />", b" ") for doc in text_test]
# >>> yimdb_test=np.array([d[1] for d in muestra_test])

# Ahora restaría vectorizar los textos. Puesto que la versión NaiveBayes que
# se ha pedido implementar es la categórica (es decir, no es la multinomial),
# a la hora de vectorizar los textos lo haremos simplemente indicando en cada
# componente del vector si el correspondiente término del vocabulario ocurre
# (1) o no ocurre (0). Para ello, usar CountVectorizer de Scikit Learn, con la
# opción binary=True. Para reducir el número de características (es decir,
# reducir el vocabulario), usar "stop words" y min_df=50. Se puede ver cómo
# hacer esto en el ejercicio del tema de modelos probabilísticos.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#######################
# IMPORTAMOS DATASTES #
#######################

# Importamos esta función de sklearn para dividir los datos en tres subconjuntos, de
# modo que usaremos un conjunto de entramiento para entrenar el modelo, uno de validación
# para ajustar los hiperparámetros y uno de test para evaluar el rendimiento final del modelos
from sklearn.model_selection import train_test_split

### Conjunto de datos Crédito
def importa_datos_credito(train_size=0.5, validation_size=0.2 ,test_size=0.3, random_state=1):
    # Imporatmos los datos y los dividimos sando train_test_split de Scikit Learn con estratificación
    # en una proporción por defecto de 50-20-30 respectivamente
    if (train_size + validation_size + test_size) != 1:
        raise ValueError("La suma de los tamaños de los conjuntos debe ser 1")
    
    import credito
    ## Leemos todos los datos
    X_credito = np.array([d[:-1] for d in credito.datos_con_la_clase])
    y_credito = np.array([d[-1] for d in credito.datos_con_la_clase])


    Xe_credito, Xaux_credito, ye_credito, yaux_credito = train_test_split(X_credito, y_credito, test_size=(1.0-train_size), random_state=random_state, stratify=y_credito)
    if validation_size != 0:
        Xv_credito, Xt_credito, yv_credito, yt_credito = train_test_split(Xaux_credito, yaux_credito, test_size=(test_size/(1.0-train_size)), random_state=random_state, stratify=yaux_credito)
    else:
        Xv_credito = None
        yv_credito = None
        Xt_credito = Xaux_credito
        yt_credito = yaux_credito
        
    return Xe_credito, ye_credito, Xv_credito, yv_credito, Xt_credito, yt_credito

### Conjunto de datos Votos
def importa_datos_votos():
    import votos
    X_votos = votos.datos
    y_votos = votos.clasif

    # el conjunto de datos de votos tiene unas marcas que indican los límites de train, test y validacion,
    # por lo que vamos a usarlas para separar los datos en los conjuntos correspondientes
    X_votos_train = X_votos[:279]
    y_votos_train = y_votos[:279]

    X_votos_valid = X_votos[279:346]
    y_votos_valid = y_votos[279:346]

    X_votos_test = X_votos[346:]
    y_votos_test = y_votos[346:]

    return X_votos_train, y_votos_train, X_votos_valid, y_votos_valid, X_votos_test, y_votos_test


### Conjunto de datos IMDB
def importa_datos_imdb():
    import random as rd
    from sklearn.datasets import load_files
    from sklearn.feature_extraction.text import CountVectorizer

    # Cargamos los textos de entrenamiento (sustituimos los saltos de línea por espacios)
    reviews_train = load_files("aclImdb/train/")
    muestra_entr=rd.sample(list(zip(reviews_train.data,
                                    reviews_train.target)),k=2000)
    text_train=[d[0] for d in muestra_entr]
    text_train = [doc.replace(b"<br />", b" ") for doc in text_train]
    yimdb_train=np.array([d[1] for d in muestra_entr])

    # Cargamos los textos de test (sustituimos los saltos de línea por espacios)
    reviews_test = load_files("aclImdb/test/")
    muestra_test=rd.sample(list(zip(reviews_test.data,
                                        reviews_test.target)),k=400)
    text_test=[d[0] for d in muestra_test]
    text_test = [doc.replace(b"<br />", b" ") for doc in text_test]
    yimdb_test=np.array([d[1] for d in muestra_test])

    # Vectorizamos los textos como se nos indica, usando el conjunto de train
    vectTrain = CountVectorizer(min_df = 50, stop_words="english").fit(text_train)

    text_train = vectTrain.transform(text_train)
    text_test = vectTrain.transform(text_test)

    Ximdb_train = (text_train.toarray()>0).astype(int)
    Ximdb_test = (text_test.toarray()>0).astype(int)

    print("Tamaño del vocabulario de aclimdb train usado: {}".format(len(vectTrain.vocabulary_)))
    return Ximdb_train, yimdb_train, Ximdb_test, yimdb_test

# Ximdb_train, yimdb_train, Ximdb_test, yimdb_test = importa_datos_imdb()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# USO DEL CLAS NB CON LOS DATOS DE CREDITO
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

X_credito_train, y_credito_train, X_credito_valid, y_credito_valid, X_credito_test, y_credito_test = importa_datos_credito(train_size=0.7, validation_size=0.1 ,test_size=0.2, random_state=1)

# DESCOMENTAR PARA EJECUTAR #
# rend_credito = [0]
# k_values = np.arange(0.01,10,0.01)
# for x in k_values:#for x in range(1,20):
#     nb_credito=NaiveBayes(k=x)
#     nb_credito.entrena(X_credito_train,y_credito_train)
#     rend = rendimiento(nb_credito,X_credito_test,y_credito_test)
    
#     #tratamos de maximizar el rendimiento:
#     if all(rend > r for r in rend_credito):
#         rend_credito.append(rend)
#         nb_credito_max = nb_credito
#         x_credito_max = x
# # Con el mejor clasificador, calculamos el rendimiento 
# print("El mejor hiperparámetro k es: ",x_credito_max)
# print("El mejor rendimiento del clasificador NB con el conjunto train de creditos es: ",rendimiento(nb_credito_max,X_credito_train,y_credito_train))
# print("El mejor rendimiento del clasificador NB con el conjunto test de creditos es: ",max(rend_credito))
# print("El rendimiento del clasificador NB con el conjunto validación de creditos es: ",rendimiento(nb_credito_max,X_credito_valid,y_credito_valid),"\n \n")

# RESULTADOS:
# El mejor hiperparámetro k es:  1.28
# El mejor rendimiento del clasificador NB con el conjunto train de creditos es:  0.711453744493392
# El mejor rendimiento del clasificador NB con el conjunto test de creditos es:  0.7022900763358778
# El rendimiento del clasificador NB con el conjunto validación de creditos es:  0.7076923076923077

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# USO DEL CLAS NB CON LOS DATOS DE VOTOS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

X_votos_train, y_votos_train, X_votos_valid, y_votos_valid, X_votos_test, y_votos_test = importa_datos_votos()

# DESCOMENTAR PARA EJECUTAR #
# rend_votos = [0]
# k_values = np.arange(0.01,10,0.01) # np.arange(0.1,1,0.1)
# for x in k_values: # tras ver que 1 era el mejor en el rango de 1 a 20, probamos con valores cercanos
#     nb_votos=NaiveBayes(k=x)
#     nb_votos.entrena(X_votos_train,y_votos_train)
#     rend = rendimiento(nb_votos,X_votos_valid,y_votos_valid)
#     #tratamos de maximizar el rendimiento:
#     if all(rend > r for r in rend_votos):
#         rend_votos.append(rend)
#         nb_votos_max = nb_votos
#         x_votos_max = x
# # Con el mejor clasificador, calculamos el rendimiento
# print("El mejor hiperparámetro k es: ",x_votos_max)
# print("El mejor rendimiento del clasificador NB con el conjunto train de votos es: ",rendimiento(nb_votos_max,X_votos_train,y_votos_train))
# print("El rendimiento del clasificador NB con el conjunto validación de votos es: ",max(rend_votos))
# print("El mejor rendimiento del clasificador NB con el conjunto test de votos es: ",rendimiento(nb_votos_max,X_votos_test,y_votos_test),"\n \n")

# RESULTADOS:
# El mejor hiperparámetro k es:  0.01
# El mejor rendimiento del clasificador NB con el conjunto train de votos es:  0.910394265232975
# El rendimiento del clasificador NB con el conjunto validación de votos es:  0.9701492537313433
# El mejor rendimiento del clasificador NB con el conjunto test de votos es:  0.8426966292134831 

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# USO DEL CLAS NB CON LOS DATOS DE IMDB
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ximdb_train, yimdb_train, Ximdb_test, yimdb_test = importa_datos_imdb()

# De el conjunto de prueba, extraemos el 60% para validación y dejamos el 40% para dar el rendimiento final
Ximdb_validacion, Ximdb_test, yimdb_validacion, yimdb_test = train_test_split(Ximdb_test, yimdb_test, test_size=0.6, random_state=41, stratify=yimdb_test)

# DESCOMENTAR PARA EJECUTAR #
# rend_imdb = [0]
# k_values = np.arange(0.01,0.5,0.01)
# for x in k_values: # tras ver que 1 era el mejor en el rango de 1 a 20, probamos con valores cercanos
#     nb_imdb=NaiveBayes(k=x)
#     nb_imdb.entrena(Ximdb_train,yimdb_train)
#     rend = rendimiento(nb_imdb,Ximdb_validacion,yimdb_validacion)
#     #tratamos de maximizar el rendimiento:
#     if all(rend > r for r in rend_imdb):
#         rend_imdb.append(rend)
#         nb_imdb_max = nb_imdb
#         x_imdb_max = x
# # Con el mejor clasificador, calculamos el rendimiento
# print("El mejor hiperparámetro k es: ",x_imdb_max)
# print("El rendimiento del clasificador NB con el conjunto train de IMDB es: ",rendimiento(nb_imdb_max,Ximdb_train,yimdb_train))
# print("El mejor rendimiento del clasificador NB con el conjunto de validacion de IMDB es: ",max(rend_imdb))
# print("El rendimiento del clasificador NB con el conjunto test de IMDB es: ",rendimiento(nb_imdb_max,Ximdb_test,yimdb_test),"\n \n")

# RESULTADOS:
# El mejor hiperparámetro k es:  0.01
# El rendimiento del clasificador NB con el conjunto train de IMDB es:  0.854
# El mejor rendimiento del clasificador NB con el conjunto de validacion de IMDB es:  0.71875
# El rendimiento del clasificador NB con el conjunto test de IMDB es:  0.8125 


# =====================================================
# PARTE II: MODELOS LINEALES PARA CLASIFICACIÓN BINARIA
# =====================================================

# En esta SEGUNDA parte se pide implementar en Python un clasificador binario
# lineal, basado en regresión logística. 



# ---------------------------------------------
# II.1) Implementación de un clasificador lineal
# ---------------------------------------------

# En esta sección se pide implementar un clasificador BINARIO basado en
# regresión logística, con algoritmo de entrenamiento de descenso por el
# gradiente mini-batch (para minimizar la entropía cruzada).

# En concreto se pide implementar una clase: 

# class RegresionLogisticaMiniBatch():

#     def __init__(self,clases=[0,1],normalizacion=False,
#                  rate=0.1,rate_decay=False,batch_tam=64)
#         .....
        
#     def entrena(self,X,y,n_epochs,reiniciar_pesos=False,pesos_iniciales=None):

#         .....        

#     def clasifica_prob(self,ejemplo):

#         ......
    
#     def clasifica(self,ejemplo):
                        
#          ......

        

# Explicamos a continuación cada uno de estos elementos:


# * El constructor tiene los siguientes argumentos de entrada:

#   + Una lista clases (de longitud 2) con los nombres de las clases del
#     problema de clasificación, tal y como aparecen en el conjunto de datos. 
#     Por ejemplo, en el caso de los datos de las votaciones, esta lista sería
#     ["republicano","democrata"]. La clase que aparezca en segundo lugar de
#     esta lista se toma como la clase positiva.  

#   + El parámetro normalizacion, que puede ser True o False (False por
#     defecto). Indica si los datos se tienen que normalizar, tanto para el
#     entrenamiento como para la clasificación de nuevas instancias. La
#     normalización es la estándar: a cada característica se le resta la media
#     de los valores de esa característica en el conjunto de entrenamiento, y
#     se divide por la desviación típica de dichos valores.

#  + rate: si rate_decay es False, rate es la tasa de aprendizaje fija usada
#    durante todo el aprendizaje. Si rate_decay es True, rate es la
#    tasa de aprendizaje inicial. Su valor por defecto es 0.1.

#  + rate_decay, indica si la tasa de aprendizaje debe disminuir en
#    cada epoch. En concreto, si rate_decay es True, la tasa de
#    aprendizaje que se usa en el n-ésimo epoch se debe de calcular
#    con la siguiente fórmula: 
#       rate_n= (rate_0)*(1/(1+n)) 
#    donde n es el número de epoch, y rate_0 es la cantidad introducida
#    en el parámetro rate anterior. Su valor por defecto es False. 

#  + batch_tam: indica el tamaño de los mini batches (por defecto 64) que se
#    usan para calcular cada actualización de pesos.



# * El método entrena tiene los siguientes parámetros de entrada:

#  + X e y son los datos del conjunto de entrenamiento y su clasificación
#    esperada, respectivamente. El primero es un array con los ejemplos, y el
#    segundo un array con las clasificaciones de esos ejemplos, en el mismo
#    orden.

#  + n_epochs: número de veces que se itera sobre todo el conjunto de
#    entrenamiento.

#  + reiniciar_pesos: si es True, cada vez que se llama a entrena, se
#    reinicia al comienzo del entrenamiento el vector de pesos de
#    manera aleatoria (típicamente, valores aleatorios entre -1 y 1).
#    Si es False, solo se inician los pesos la primera vez que se
#    llama a entrena. En posteriores veces, se parte del vector de
#    pesos calculado en el entrenamiento anterior, excepto que se diera
#    explícitamente el vector de pesos en el parámetro peso_iniciales.  

#  + pesos_iniciales: si no es None y el parámetro anterior reiniciar_pesos 
#    es False, es un array con los pesos iniciales. Este parámetro puede ser
#    útil para empezar con unos pesos que se habían obtenido y almacenado como
#    consecuencia de un entrenamiento anterior.



# * Los métodos clasifica y clasifica_prob se describen como en el caso del
#   clasificador NaiveBayes. Igualmente se debe devolver
#   ClasificadorNoEntrenado si llama a los métodos de clasificación antes de
#   entrenar. 

# Se recomienda definir la función sigmoide usando la función expit de
# scipy.special, para evitar "warnings" por "overflow":

# from scipy.special import expit    
#
# def sigmoide(x):
#    return expit(x)


# ----------------------------------------------------------------


# Ejemplo de uso, con los datos del cáncer de mama, que se puede cargar desde
# Scikit Learn:

# >>> from sklearn.datasets import load_breast_cancer
# >>> cancer=load_breast_cancer()

# >>> X_cancer,y_cancer=cancer.data,cancer.target


# >>> lr_cancer=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True,normalizacion=True)

# >>> lr_cancer.entrena(Xe_cancer,ye_cancer,10000)

# >>> rendimiento(lr_cancer,Xe_cancer,ye_cancer)
# 0.9906103286384976
# >>> rendimiento(lr_cancer,Xt_cancer,yt_cancer)
# 0.972027972027972

# -----------------------------------------------------------------

# Definimos la función sigmoide tal y como se nos indica
from scipy.special import expit    
def sigmoide(x):
   return expit(x)

class RegresionLogisticaMiniBatch():

    def __init__(self,clases=[0,1],normalizacion=False,
                 rate=0.1,rate_decay=False,batch_tam=64):
        
        # Datos del conjunto de entrenamiento
        self.clases = clases
        self.n_dim = None

        # Normalización estándar
        self.normalizacion = normalizacion
        self.media = None
        self.std = None

        # Tasa de aprendizaje
        self.rate = rate
        self.rate_decay = rate_decay

        # Tamaño de los mini batches
        self.batch_tam = batch_tam

        # Variables internas
        self.modelo_entrenado = False
        self.w = None

    def entrena(self,X,y,n_epochs,reiniciar_pesos=False,pesos_iniciales=None):

        # Obtenemos la dimensión del modelo
        self.n_dim = X.shape[1] # Número de atributos (x1, x2, ..., xn)

        ########
        # Normalizamos los datos
        if self.normalizacion:
            self.__ajusta_normalizador(X)
            X_norm = self.__normaliza(X)
        else:
            X_norm = X
        
        # Ampliamos los datos con x0 = 1
        X_ampliada = np.hstack((np.ones((len(X_norm),1)),X_norm))

        # Ajustamos y a valores numéricos de 0.0 y 1.0
        y = np.array([0.0 if clase == self.clases[0] else 1.0 for clase in y])
        
        ########
        # Inicializamos los pesos: np.array(w0, w1, ..., wn)

        if reiniciar_pesos: 
            # Si reiniciar_pesos es True, asignamos pesos aleatorios
            self.w = np.random.uniform(-1,1,self.n_dim+1)

        elif pesos_iniciales is not None:
             # Si no, asignamos los pesos iniciales si estos se han proporcionado
             self.w = pesos_iniciales

        else:
            # Si no, asignamos pesos aleatorios o nos quedamos con los que ya teníamos
            if self.w is None:
                self.w = np.random.uniform(-1,1,self.n_dim+1)

        # Para que los resultados sean reproducibles vamos a fijar una semilla a partir de los cuales
        # se generarán los números aleatorios tomados como random_state en np.random.permutation
        seed = 42
        random_states = np.random.RandomState(seed).randint(0, 100000, n_epochs, dtype=np.int64)

        ########
        # Entrenamos el modelo
        for epoch in range(n_epochs):

            # La tasa de aprendizaje dependerá de cada epoch si rate_decay es True 
            if self.rate_decay:
                learning_rate = self.rate / (1 + epoch)
            else:
                learning_rate = self.rate
            
            # Mezclamos los datos para generar aleatoriedad en los mini batches
            indices = np.random.RandomState(random_states[epoch]).permutation(len(X_ampliada))
            
            X_ampliada_mezclada= X_ampliada[indices]
            y_mezclada = y[indices]

            for i in range(0, len(X_ampliada_mezclada), self.batch_tam):
                # Obtenemos el conjunto de datos del mini batch
                # asegurando que no nos salimos del rango
                fin = min(i + self.batch_tam, len(X_ampliada_mezclada))
                X_batch = X_ampliada_mezclada[i:fin]
                y_batch = y_mezclada[i:fin]

                # Calculamos la predicción
                y_pred = sigmoide(np.dot(X_batch, self.w))
                error = y_pred - y_batch

                # Actualizamos los pesos
                self.w = self.w - learning_rate * np.dot(X_batch.T, error)

                # Esta expresión matricial realiza de forma compacta el cálculo que se corresponde 
                # con la actualización de los pesos por descenso por el gradiente:
                #
                # wi = wi - learning_rate * sum(error(ejk)*xi(ejk) for ejk in error)
                #
                # donde, las matrices que es multiplican son:
                #   
                # X_batch.T = [[x0(ej0), x0(ej1), ..., x0(ejk)],
                #              [x1(ej0), x1(ej1), ..., x1(ejk)], 
                #              ...,
                #              [xn(ej0), xn(ej1), ..., xn(ejk)]]
                #
                # error      = [error(ej0),
                #               error(ej1),
                #               ...,
                #               error(ejn)]
                #     
                # np.dot(X_batch.T, error) = [error(ej0)*x0(ej0) + error(ej1)*x0(ej1) + ... + error(ejn)*x0(ejn),
                #                             error(ej0)*x1(ej0) + error(ej1)*x1(ej1) + ... + error(ejn)*x1(ejn),
                #                             ...,
                #                             error(ej0)*xn(ej0) + error(ej1)*xn(ej1) + ... + error(ejn)*xn(ejn)]
                #
                # Es decir, cada peso se actualiza como:
                # wi = wi - lerning_rate * (error(ej0)*xi(ej0) + error(ej1)*xi(ej1) + ... + error(ejn)*xi(ejn))
                #


        # Indicamos que el modelo ha sido entrenado
        self.modelo_entrenado = True

    def clasifica_prob(self,ejemplo):
        # Lanzamos una excepción si el modelo no ha sido entrenado
        if not self.modelo_entrenado:
            raise ClasificadorNoEntrenado
        
        # Normalizamos el ejemplo si es necesario
        if self.normalizacion:
            ejemplo_norm = self.__normaliza(ejemplo)
        else:
            ejemplo_norm = ejemplo


        # Ampliamos el ejemplo con x0 = 1
        ejemplo_ampliado = np.hstack((np.ones(1),ejemplo_norm))

        # Calculamos la probabilidad de que el ejemplo pertenezca a cada clase
        probabilidad = sigmoide(np.dot(ejemplo_ampliado, self.w))

        # Ajustamos a un diccionario con las clases y sus probabilidades
        # (la clase positiva es la segunda de la lista de clases)
        # (la clase negativa es la primera de la lista de clases)
        return {self.clases[0]: 1 - probabilidad, self.clases[1]: probabilidad} 

    def clasifica(self,ejemplo:np.ndarray):
        # Lanzamos una excepción si el modelo no ha sido entrenado
        if not self.modelo_entrenado:
            raise ClasificadorNoEntrenado
        
        # Normalizamos el ejemplo si es necesario
        if self.normalizacion:
            ejemplo_norm = self.__normaliza(ejemplo)
        else:
            ejemplo_norm = ejemplo

        # Ampliamos el ejemplo con x0 = 1
        ejemplo_ampliado = np.hstack((np.ones(1),ejemplo_norm))

        # Calculamos la probabilidad de que el ejemplo pertenezca a cada clase
        probabilidad = sigmoide(np.dot(ejemplo_ampliado, self.w))

        # Devolvemos la clase con mayor probabilidad
        if probabilidad >= 0.5:
            return self.clases[1]
        else:
            return self.clases[0]

    def __ajusta_normalizador(self, X):
        # Calculamos media y desviación típica del conjunto de datos
        # y las guardamos en los atributos de la clase
        self.media = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
    
    def __normaliza(self, X):
        # Aplicamos normalización estándar a los datos
        return (X.copy() - self.media) / self.std

def rendimiento(clasificador, X, y):
    # Calculamos el número de aciertos
    aciertos = 0
    for i in range(len(X)):
        if clasificador.clasifica(X[i]) == y[i]:
            aciertos += 1
    
    # Devolvemos el porcentaje de aciertos
    return aciertos / len(X)

# Se comprueba su correcto funcionamiento con todo el conjunto de datos del 
# cáncer de mama a modo de prueba básica de funcionamiento

# DESCOMENTAR PARA EJECUTAR #
# from sklearn.datasets import load_breast_cancer
# cancer=load_breast_cancer()
# X_cancer,y_cancer=cancer.data,cancer.target
# lr_cancer=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True,normalizacion=True)
# lr_cancer.entrena(X_cancer,y_cancer,10000)
# print(rendimiento(lr_cancer,X_cancer,y_cancer))
# RESULTADO: 0.9876977152899824

# -----------------------------------
# II.2) Aplicando Regresión Logística 
# -----------------------------------

# Usando el clasificador implementado, obtener clasificadores con el mejor
# rendimiento posible para los siguientes conjunto de datos:

# - Votos de congresistas US
# - Cáncer de mama 
# - Críticas de películas en IMDB

# Como antes, será necesario separar conjuntos de validación y test para dar
# la valoración final de los clasificadores obtenidos Se permite usar
# train_test_split de Scikit Learn para esto. Ajustar los parámetros de tamaño
# de batch, tasa de aprendizaje y rate_decay. En alguno de los conjuntos de
# datos puede ser necesaria normalización.

# Mostrar el proceso realizado en cada caso, y los rendimientos obtenidos. 

# -----------------------------------------------------------------

################################
# Búsqueda de hiperparámetros

# Dado que se debe ajustar el modelo de regresión logística a los conjuntos de datos,
# y se nos pide que el modelo ha de tener el mejor rendimiento posible, vamos a
# realizar una búsqueda de hiperparámetros para cada conjunto de datos.

# Se van a probar distintos valores de tasa de aprendizaje, tamaños de batch y rate_decay
# para obtener el mejor rendimiento posible, al menos en la búsqueda realizada para los primeros
# conjuntos de datos.

# Tasas de aprendizaje:
rates = [1.0, 0.1, 0.01]
# Tamaño de batch
batch_tams = [32, 64, 128]
# Uso de rate_decay
rate_decays = [True, False]
# Uso de normalización
normalizaciones = [True, False]
# Número de epochs
n_epochs = [1000, 5000, 10000]

# Generamos todas las combinaciones de hiperparámetros
opciones = [(rate, batch_tam, rate_decay, normalizacion, n_epoch)
            for rate in rates
            for batch_tam in batch_tams
            for rate_decay in rate_decays
            for normalizacion in normalizaciones
            for n_epoch in n_epochs]

def evaluacion_hiperparametros(opciones, Xe, ye, Xv, yv, Model, clases=[0,1], debug=False):

    print('Evaluando %d combinaciones hiperparámetros...' % len(opciones))

    # Evaluamos el rendimiento de cada combinación de hiperparámetros
    rendimiento_opt = []
    modelos_entrenados = []
    for opt in opciones:

        # Entrenamos un modelo para la combinación de hiperparámetros opt
        if Model == RegresionLogisticaMiniBatch:
            modelo = Model(rate=opt[0], batch_tam=opt[1], rate_decay=opt[2], normalizacion=opt[3], clases=clases)

        elif Model == RL_OvR:
            modelo = Model(rate=opt[0], batch_tam=opt[1], rate_decay=opt[2], clases=clases)
        else:
            raise Exception("Modelo no reconocido")

        modelo.entrena(Xe, ye, opt[4])

        # Guardamos el rendimiento en la validación y el modelo entrenado
        rendimiento_opt.append(rendimiento(modelo, Xv, yv))
        modelos_entrenados.append(modelo)

        if debug:
            print("Iteración: ", len(rendimiento_opt), " de ", len(opciones))
            print("Hiperparámetros: ", opt)
            print("Rendimiento en validacion: ", rendimiento_opt[-1])
            

    # Obtenemos el mejor rendimiento y la mejor combinación de hiperparámetros
    mejor_rendimiento_index = np.argmax(rendimiento_opt)
    mejor_opt = opciones[mejor_rendimiento_index]
    mejor_modelo = modelos_entrenados[mejor_rendimiento_index]

    print("Rendimiento en entrenamiento: ", rendimiento(mejor_modelo, Xe, ye))
    print("Rendimiento en validación: ", rendimiento(mejor_modelo, Xv, yv))
    print("Mejor combinación de hiperparámetros: ", mejor_opt)

    return mejor_modelo

################################
### Votos de congresistas US ###
################################

# Importamos los datos y los dividimos sando train_test_split de Scikit Learn con estratificación
# en una proporción 50-20-30 respectivamente, usndo la función definida anteriormente
X_votos_train, y_votos_train, X_votos_valid, y_votos_valid, X_votos_test, y_votos_test = importa_datos_votos()

# Evaluamos el rendimiento de cada combinación de hiperparámetros

# DESCOMENTAR PARA EJECUTAR #
# print("\nVotos de congresistas US")
# lr_votos = evaluacion_hiperparametros(opciones, X_votos_train, y_votos_train, X_votos_valid, y_votos_valid, RegresionLogisticaMiniBatch)
# print("Rendimiento en test: ", rendimiento(lr_votos, X_votos_test, y_votos_test))

# RESULTADOS
# Rendimiento en entrenamiento:  0.992831541218638
# Rendimiento en validación:  0.9850746268656716
# Mejor combinación de hiperparámetros:  (1.0, 32, True, True, 1000)
# Rendimiento en test:  0.8876404494382022

# Es decir, el modelo con mejor rendimiento para este conjunto de datos,
# es el que tiene los siguientes hiperparámetros:
# - Tasa de aprendizaje inicial de 1.0 con rate_decay
# - Tamaño de batch: 32
# - Normalización: True
# - Número de epochs: 1000

######################
### Cáncer de Mama ###
######################

# Importamos los datos y los dividimos sando train_test_split de Scikit Learn con estratificación
# en una proporción 50-20-30 respectivamente
from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
X_cancer,y_cancer=cancer.data,cancer.target
Xe_cancer, Xaux_cancer, ye_cancer, yaux_cancer = train_test_split(X_cancer, y_cancer, test_size=0.5, random_state=41, stratify=y_cancer)
Xv_cancer, Xt_cancer, yv_cancer, yt_cancer = train_test_split(Xaux_cancer, yaux_cancer, test_size=0.6, random_state=41, stratify=yaux_cancer)

# Evaluamos el rendimiento de cada combinación de hiperparámetros

# DESCOMENTAR PARA EJECUTAR #
# print("\nCáncer de mama")
# lr_cancer = evaluacion_hiperparametros(opciones, Xe_cancer, ye_cancer, Xv_cancer, yv_cancer, RegresionLogisticaMiniBatch)
# print("Rendimiento en test: ", rendimiento(lr_cancer, Xt_cancer, yt_cancer))

# RESULTADOS
# Rendimiento en entrenamiento:  0.9823943661971831
# Rendimiento en validación:  0.9824561403508771
# Mejor combinación de hiperparámetros:  (0.01, 32, True, True, 10000)
# Rendimiento en test:  0.9473684210526315

# Es decir, el modelo con mejor rendimiento para este conjunto de datos,
# es el que tiene los siguientes hiperparámetros:
# - Tasa de aprendizaje inicial de 0.01 con rate_decay
# - Tamaño de batch: 32
# - Normalización: True
# - Número de epochs: 10000

#####################
### Críticas IMDB ###
#####################

Ximdb_train, yimdb_train, Ximdb_test, yimdb_test = importa_datos_imdb()

# De el conjunto de prueba, extraemos el 60% para validación y dejamos el 40% para dar el rendimiento final
Ximdb_validacion, Ximdb_test, yimdb_validacion, yimdb_test = train_test_split(Ximdb_test, yimdb_test, test_size=0.6, random_state=41, stratify=yimdb_test)

# Evaluamos el rendimiento de cada combinación de hiperparámetros, pero reducimos el espacio de búsqueda
# ya que la dimensión de los datos es muy grande y el tiempo de ejecución de ejecución es considerablemente
# mayor que en los otros conjuntos de datos

# Dado que en este caso los atributos son binarios, no hace falta que normalicemos los datos
# Además, siempre hemos obtenido un mejor rendimiento con rate_decay, por lo que lo vamos a fijar a True

# Por tanto, vamos a probar distintos valores de tasa de aprendizaje, tamaños de batch y rate_decay

# Tasas de aprendizaje:
rates = [1.0, 0.1, 0.01]
# Tamaño de batch
batch_tams = [128, 256, 512]
# Uso de rate_decay
rate_decays = [True]
# Uso de normalización
normalizaciones = [False]
# Número de epochs
n_epochs = [1000, 5000]

# Generamos todas las combinaciones de hiperparámetros
opciones = [(rate, batch_tam, rate_decay, normalizacion, n_epoch)
            for rate in rates
            for batch_tam in batch_tams
            for rate_decay in rate_decays
            for normalizacion in normalizaciones
            for n_epoch in n_epochs]

# DESCOMENTAR PARA EJECUTAR #
# print("\nConjunto de datos IMDB")
# lr_imdb = evaluacion_hiperparametros(opciones, Ximdb_train, yimdb_train, Ximdb_validacion, yimdb_validacion, RegresionLogisticaMiniBatch, debug=True)
# print("Rendimiento en test: ", rendimiento(lr_imdb, Ximdb_test, yimdb_test))

# Rendimiento en entrenamiento:  0.963
# Rendimiento en validación:  0.83125
# Mejor combinación de hiperparámetros:  (1.0, 512, True, False, 1000)
# Rendimiento en test:  0.825

# Es decir, el modelo con mejor rendimiento para este conjunto de datos,
# es el que tiene los siguientes hiperparámetros:
# - Tasa de aprendizaje inicial de 1.0
# - Tamaño de batch: 512
# - Número de epochs: 1000

# ===================================
# PARTE III: CLASIFICACIÓN MULTICLASE
# ===================================

# Se pide implementar un algoritmo de regresión logística para problemas de
# clasificación en los que hay más de dos clases, usando  la técnica
# de "One vs Rest" (OvR)

# ------------------------------------
# III.1) Implementación de One vs Rest
# ------------------------------------


#  En concreto, se pide implementar una clase python RL_OvR con la siguiente
#  estructura, y que implemente un clasificador OvR usando como base el
#  clasificador binario del apartado anterior.


# class RL_OvR():

#     def __init__(self,clases,rate=0.1,rate_decay=False,batch_tam=64):

#        ......

#     def entrena(self,X,y,n_epochs):

#        .......

#     def clasifica(self,ejemplo):

#        ......
            



#  Los parámetros de los métodos significan lo mismo que en el apartado
#  anterior, excepto que ahora "clases" puede ser una lista con más de dos
#  elementos. 

#  Un ejemplo de sesión, con el problema del iris:


# --------------------------------------------------------------------
# >>> from sklearn.datasets import load_iris
# >>> iris=load_iris()
# >>> X_iris=iris.data
# >>> y_iris=iris.target
# >>> Xe_iris,Xt_iris,ye_iris,yt_iris=train_test_split(X_iris,y_iris)

# >>> rl_iris=RL_OvR([0,1,2],rate=0.001,batch_tam=20)

# >>> rl_iris.entrena(Xe_iris,ye_iris,n_epochs=1000)

# >>> rendimiento(rl_iris,Xe_iris,ye_iris)
# 0.9732142857142857

# >>> rendimiento(rl_iris,Xt_iris,yt_iris)
# >>> 0.9736842105263158
# --------------------------------------------------------------------

class RL_OvR():

    def __init__(self,clases,rate=0.1,rate_decay=False,batch_tam=64):

        # Dimensión del problema
        self.classes = clases

        # Tasa de aprendizaje
        self.rate = rate
        self.rate_decay = rate_decay

        # Tamaño de los mini batches
        self.batch_tam = batch_tam

        # Variables internas
        self.modelo_entrenado = False

        # Diccionario de clasificadores binarios (uno por clase)
        # usando los que ya teníamos implementados en el apartado anterior
        self.binary_classifiers = {}
        for clase in clases:
            binary_classifier = RegresionLogisticaMiniBatch(clases=[0.0, 1.0], rate=rate, rate_decay=rate_decay, batch_tam=batch_tam)
            self.binary_classifiers[clase] = binary_classifier

    def entrena(self,X,y,n_epochs):

        for clase in self.classes:

            # Ajustamos y a valores numéricos de 0.0 y 1.0
            y_clase = np.array([0.0 if y[i] != clase else 1.0 for i in range(len(y))])

            # Entrenamos el clasificador binario para la clase actual
            self.binary_classifiers[clase].entrena(X, y_clase, n_epochs)

        self.modelo_entrenado = True

    def clasifica(self,ejemplo):
        if not self.modelo_entrenado:
            raise ClasificadorNoEntrenado
        
        # Calculamos la probabilidad de que el ejemplo pertenezca a cada clase
        probabilidades = {}
        for clase in self.classes:
            probabilidades[clase] = self.binary_classifiers[clase].clasifica_prob(ejemplo)[1.0]
        
        # Devolvemos la clase con mayor probabilidad
        return max(probabilidades, key=probabilidades.get)

# Se comprueba su correcto funcioamiento con el conjunto de datos del
# iris

# from sklearn.datasets import load_iris
# iris=load_iris()
# X_iris=iris.data
# y_iris=iris.target
# Xe_iris,Xt_iris,ye_iris,yt_iris=train_test_split(X_iris,y_iris, test_size=0.2, random_state=1, stratify=y_iris)

# rl_multiclase = RL_OvR([0,1,2],rate=0.001,batch_tam=20)
# rl_multiclase.entrena(Xe_iris,ye_iris,n_epochs=1000)
# print('Rendimiento en el conjunto de entrenamiento: ', rendimiento(rl_multiclase,Xe_iris,ye_iris))
# print('Rendimiento en el conjunto de prueba:', rendimiento(rl_multiclase,Xt_iris,yt_iris))

# RESULTADOS
# Rendimiento en el conjunto de entrenamiento: 0.9666666666666667
# Rendimiento en el conjunto de prueba: 0.9666666666666667

# ------------------------------------------------------------
# III.2) Clasificación de imágenes de dígitos escritos a mano
# ------------------------------------------------------------


#  Aplicar la implementación del apartado anterior, para obtener un
#  clasificador que prediga el dígito que se ha escrito a mano y que se
#  dispone en forma de imagen pixelada, a partir de los datos que están en el
#  archivo digidata.zip que se suministra.  Cada imagen viene dada por 28x28
#  píxeles, y cada pixel vendrá representado por un caracter "espacio en
#  blanco" (pixel blanco) o los caracteres "+" (borde del dígito) o "#"
#  (interior del dígito). En nuestro caso trataremos ambos como un pixel negro
#  (es decir, no distinguiremos entre el borde y el interior). En cada
#  conjunto las imágenes vienen todas seguidas en un fichero de texto, y las
#  clasificaciones de cada imagen (es decir, el número que representan) vienen
#  en un fichero aparte, en el mismo orden. Será necesario, por tanto, definir
#  funciones python que lean esos ficheros y obtengan los datos en el mismo
#  formato numpy en el que los necesita el clasificador. 

#  Los datos están ya separados en entrenamiento, validación y prueba. Si el
#  tiempo de cómputo en el entrenamiento no permite terminar en un tiempo
#  razonable, usar menos ejemplos de cada conjunto.

# Ajustar los parámetros de tamaño de batch, tasa de aprendizaje y
# rate_decay para tratar de obtener un rendimiento aceptable (por encima del
# 75% de aciertos sobre test). 


# Definimos una función para cargar las imágenes binarias, según hemos visto
# en el ejercicio 1 del tema 2 en clase:

def cargaImagenes(fichero,ancho,alto):

    # Si el caracter es distingo de un espacio en blanco devolvemos un 1
    def convierte_0_1(c):
        if c==" ":
            return 0
        else:
            return 1

    # Abrimos el fichero con los dígitos
    with open(fichero) as f:

        lista_imagenes=[]
        ejemplo=[]
        cont_lin=0

        for lin in f:
            
            # Metemos lo que hay en cada línea del fichero
            ejemplo.extend(list(map(convierte_0_1,lin[:ancho])))
            cont_lin+=1

            # Si se llega al alto de la imagen se carga en la lista de imágenes
            if cont_lin == alto:
                lista_imagenes.append(ejemplo)
                ejemplo=[]
                cont_lin=0

    # Se devuelve la imagen en un array de numpy tal y como lo exigen nuestras implementaciones
    # de los clasificadores realizados
    return np.array(lista_imagenes)

def cargaClases(fichero):
   with open(fichero) as f:
       return np.array([int(c) for c in f])

# Paths a los ficheros de datos
trainingdigits="digitdata/trainingimages"
validationdigits="digitdata/validationimages"
testdigits="digitdata/testimages"
trainingdigitslabels="digitdata/traininglabels"
validationdigitslabels="digitdata/validationlabels"
testdigitslabels="digitdata/testlabels"

# Cargamos los datos
X_train_dg=cargaImagenes(trainingdigits,28,28)
y_train_dg=cargaClases(trainingdigitslabels)
X_valid_dg=cargaImagenes(validationdigits,28,28)
y_valid_dg=cargaClases(validationdigitslabels)
X_test_dg=cargaImagenes(testdigits,28,28)
y_test_dg=cargaClases(testdigitslabels)

# Tasas de aprendizaje:
rates = [1.0, 0.1, 0.01]
# Tamaño de batch
batch_tams = [128, 256, 512]
# Uso de rate_decay
rate_decays = [True]
# Uso de normalización
normalizaciones = [False]
# Número de epochs
n_epochs = [1000]

# Generamos todas las combinaciones de hiperparámetros
opciones = [(rate, batch_tam, rate_decay, normalizacion, n_epoch)
            for rate in rates
            for batch_tam in batch_tams
            for rate_decay in rate_decays
            for normalizacion in normalizaciones
            for n_epoch in n_epochs]

# DESCOMENTAR PARA EJECUTAR #
# print("\nDigitos escritos a mano")
# rl_multiclase = evaluacion_hiperparametros(opciones, X_train_dg, y_train_dg, X_valid_dg, y_valid_dg, clases=[0,1,2,3,4,5,6,7,8,9], Model=RL_OvR, debug=True)
# print("Rendimiento en test: ", rendimiento(rl_multiclase, X_test_dg, y_test_dg))

# RESULTADOS
# Rendimiento en entrenamiento:  0.931
# Rendimiento en validación:  0.881
# Mejor combinación de hiperparámetros:  (0.1, 512, True, False, 1000)
# Rendimiento en test:  0.85

# Es decir, el modelo con mejor rendimiento para este conjunto de datos,
# es el que tiene los siguientes hiperparámetros:
# - Tasa de aprendizaje inicial de 0.1
# - Tamaño de batch: 512
# - Número de epochs: 1000
