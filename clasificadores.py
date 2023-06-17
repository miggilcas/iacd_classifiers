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
        # para guardar las probabilidades de cada atributo, al no saber el número de valores que puede tomar cada atributo se guarda en un diccionario para facilitar el acceso
        # a la hora de clasificar, su acceso más tarde será: self.prob_atributos[clase][atributo][valor]
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
from jugar_tenis import *

nb_tenis=NaiveBayes(k=0.5)
nb_tenis.entrena(X_tenis,y_tenis)
print("Entrenando el clasificador con el dataset de jugar al tenis...")
ej_tenis=np.array(['Soleado','Baja','Alta','Fuerte'])
print("Clasificando el ejemplo: ",ej_tenis)
print("Probabilidades de cada clase: ",nb_tenis.clasifica_prob(ej_tenis))
print("Clase predicha: ",nb_tenis.clasifica(ej_tenis))

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

print("Rendimiento del clasificador NB con el conjunto de jugar tenis: ",rendimiento(nb_tenis,X_tenis,y_tenis),"\n \n")
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
  

## importamos los sencillitos
import credito 
import votos

## recolección de datos
X_credito = np.array([d[:-1] for d in credito.datos_con_la_clase])
y_credito = np.array([d[-1] for d in credito.datos_con_la_clase])

X_votos = votos.datos
y_votos = votos.clasif

## separación en train y test
# credito
from sklearn.model_selection import train_test_split # usamos sklearn en una primera instancia
X_credito_train, X_credito_test, y_credito_train, y_credito_test = train_test_split(X_credito, y_credito, test_size=0.2, random_state=42)

# votos
X_votos_train = X_votos[:279]
y_votos_train = y_votos[:279]

X_votos_valid = X_votos[279:346]
y_votos_valid = y_votos[279:346]

X_votos_test = X_votos[346:]
y_votos_test = y_votos[346:]

### Conjunto de datos IMDB
import random as rd
from sklearn.datasets import load_files
reviews_train = load_files("aclImdb/train/")
muestra_entr=rd.sample(list(zip(reviews_train.data,
                                reviews_train.target)),k=2000)
text_train=[d[0] for d in muestra_entr]
text_train = [doc.replace(b"<br />", b" ") for doc in text_train]
yimdb_train=np.array([d[1] for d in muestra_entr])


reviews_test = load_files("aclImdb/test/")
muestra_test=rd.sample(list(zip(reviews_test.data,
                                    reviews_test.target)),k=400)
text_test=[d[0] for d in muestra_test]
text_test = [doc.replace(b"<br />", b" ") for doc in text_test]
yimdb_test=np.array([d[1] for d in muestra_test])


# Vectorizamos los textos como se indica
from sklearn.feature_extraction.text import CountVectorizer

vectTrain = CountVectorizer(min_df = 50, stop_words="english").fit(text_train)
#vectTest = CountVectorizer(min_df = 50, stop_words="english").fit(text_test)

text_train = vectTrain.transform(text_train)
#text_test = vectTest.transform(text_test)
text_test = vectTrain.transform(text_test)

Ximdb_train = (text_train.toarray()>0).astype(int)
Ximbd_test = (text_test.toarray()>0).astype(int)

print("Tamaño del vocabulario de aclimdb train: {}".format(len(vectTrain.vocabulary_)))
#print("Tamaño del vocabulario de aclimdb test: {}".format(len(vectTest.vocabulary_)))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# USO DEL CLAS NB CON LOS DATOS DE CREDITO
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
rend_credito = [0]
k_values = np.arange(0.01,10,0.01)
for x in k_values:#for x in range(1,20):
    nb_credito=NaiveBayes(k=x)
    nb_credito.entrena(X_credito_train,y_credito_train)
    rend = rendimiento(nb_credito,X_credito_test,y_credito_test)
    #tratamos de maximizar el rendimiento:
    if all(rend > r for r in rend_credito):
        rend_credito.append(rend)
        nb_credito_max = nb_credito
        x_credito_max = x

# Con el mejor clasificador, calculamos el rendimiento 
print("El mejor hiperparámetro k es: ",x_credito_max)
print("El mejor rendimiento del clasificador NB con el conjunto train de creditos es: ",rendimiento(nb_credito_max,X_credito_train,y_credito_train))
print("El mejor rendimiento del clasificador NB con el conjunto test de creditos es: ",max(rend_credito),"\n \n")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# USO DEL CLAS NB CON LOS DATOS DE VOTOS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
rend_votos = [0]
k_values = np.arange(0.01,10,0.01) # np.arange(0.1,1,0.1)
for x in k_values: # tras ver que 1 era el mejor en el rango de 1 a 20, probamos con valores cercanos
    nb_votos=NaiveBayes(k=x)
    nb_votos.entrena(X_votos_train,y_votos_train)
    rend = rendimiento(nb_votos,X_votos_test,y_votos_test)
    #tratamos de maximizar el rendimiento:
    if all(rend > r for r in rend_votos):
        rend_votos.append(rend)
        nb_votos_max = nb_votos
        x_votos_max = x

# Con el mejor clasificador, calculamos el rendimiento
print("El mejor hiperparámetro k es: ",x_votos_max)
print("El mejor rendimiento del clasificador NB con el conjunto test de votos es: ",max(rend_votos))
print("El rendimiento del clasificador NB con el conjunto validación de votos es: ",rendimiento(nb_votos_max,X_votos_valid,y_votos_valid),"\n \n")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# USO DEL CLAS NB CON LOS DATOS DE IMDB
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
rend_imdb = [0]
k_values = np.arange(0.01,0.5,0.01)
for x in k_values: # tras ver que 1 era el mejor en el rango de 1 a 20, probamos con valores cercanos
    nb_imdb=NaiveBayes(k=x)
    nb_imdb.entrena(Ximdb_train,yimdb_train)
    rend = rendimiento(nb_imdb,Ximbd_test,yimdb_test)
    #tratamos de maximizar el rendimiento:
    if all(rend > r for r in rend_imdb):
        rend_imdb.append(rend)
        nb_imdb_max = nb_imdb
        x_imdb_max = x

# Con el mejor clasificador, calculamos el rendimiento
print("El mejor hiperparámetro k es: ",x_imdb_max)
print("El mejor rendimiento del clasificador NB con el conjunto test de IMDB es: ",max(rend_imdb))




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

