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

class NaiveBayes():
    def __init__(self,k=1): 
        self.k = k
        self.clases = None
        self.prob_clases = None
        self.prob_atributos = None
        self.num_atributos = None
        self.num_clases = None
        self.atributos = None
        self.atributos_clases = None
        self.atributos_clases_prob = None
        self.atributos_prob = None
        self.atributos_prob_log = None
        self.atributos_clases_prob_log = None
        self.atributos_clases_prob_log_sum = None
    
    def entrena(self,X,y):
        # We calculate the number of classes and the number of attributes
        self.clases = np.unique(y)
        self.num_clases = len(self.clases)
        self.prob_clases = np.zeros(self.num_clases)
        self.atributos = np.unique(X)
        self.num_atributos = len(self.atributos)
        

# * El constructor recibe como argumento la constante k de suavizado (por
#   defecto 1) 
# * Método entrena, recibe como argumentos dos arrays de numpy, X e y, con los
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
import jugar_tenis


# ----------------------------------------------
# I.2) Implementación del cálculo de rendimiento
# ----------------------------------------------

# Definir una función "rendimiento(clasificador,X,y)" que devuelve la
# proporción de ejemplos bien clasificados (accuracy) que obtiene el
# clasificador sobre un conjunto de ejemplos X con clasificación esperada y. 

# Ejemplo:

# >>> rendimiento(nb_tenis,X_tenis,y_tenis)
# 0.9285714285714286

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
  

# importing dummy datasets
# import credito
# import votos












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

# Se comprueba su correcto funcioamiento con el conjunto de datos del 
# cáncer de mama

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
# Importamos esta función de sklearn para dividir los datos en tres subconjuntos, de
# modo que usaremos un conjunto de entramiento para entrenar el modelo, uno de validación
# para ajustar los hiperparámetros y uno de test para evaluar el rendimiento final del modelo
from sklearn.model_selection import train_test_split

# Dado que se debe ajustar el modelo de regresión logística a los conjuntos de datos,
# y se nos pide que el modelo ha de tener el mejor rendimiento posible, vamos a
# realizar una búsqueda de hiperparámetros para cada conjunto de datos.

# Las opciones que vamos a evaluar son las siguientes:
################################
# Búsqueda de hiperparámetros

# Se van a probar distintos valores de tasa de aprendizaje, tamaños de batch y rate_decay
# para obtener el mejor rendimiento posible

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

def evaluacion_hiperparametros(opciones, Xe, ye, Xv, yv):

    print('Evaluando %d combinaciones hiperparámetros...' % len(opciones))

    # Evaluamos el rendimiento de cada combinación de hiperparámetros
    rendimiento_opt = []
    lr_entrenados = []
    for opt in opciones:

        # Entrenamos un modelo para la combinación de hiperparámetros opt
        lr = RegresionLogisticaMiniBatch(rate=opt[0], batch_tam=opt[1], rate_decay=opt[2], normalizacion=opt[3])
        lr.entrena(Xe, ye, opt[4])

        # Guardamos el rendimiento en la validación y el modelo entrenado
        rendimiento_opt.append(rendimiento(lr, Xv, yv))
        lr_entrenados.append(lr)

    # Obtenemos el mejor rendimiento y la mejor combinación de hiperparámetros
    mejor_rendimiento_index = np.argmax(rendimiento_opt)
    mejor_opt = opciones[mejor_rendimiento_index]
    mejor_lr = lr_entrenados[mejor_rendimiento_index]

    print("Rendimiento en entrenamiento: ", rendimiento(mejor_lr, Xe, ye))
    print("Rendimiento en validación: ", rendimiento(mejor_lr, Xv, yv))
    print("Mejor combinación de hiperparámetros: ", mejor_opt)

    return mejor_lr

################################
### Votos de congresistas US ###
################################

# Imporatmos los datos y los dividimos sando train_test_split de Scikit Learn con estratificación
# en una proporción 50-20-30 respectivamente
import votos
X_votos = votos.datos
y_votos = votos.clasif
Xe_votos, Xaux_votos, ye_votos, yaux_votos = train_test_split(X_votos, y_votos, test_size=0.5, random_state=41, stratify=y_votos)
Xv_votos, Xt_votos, yv_votos, yt_votos = train_test_split(Xaux_votos, yaux_votos, test_size=0.6, random_state=41, stratify=yaux_votos)

# DESCOMENTAR PARA EJECUTAR #
# print("\nVotos de congresistas US")
# # Llamamos a la función anterior para obtener el modelo con mejor rendimiento sobre el conjunto de validación
# lr_votos = evaluacion_hiperparametros(opciones, Xe_votos, ye_votos, Xv_votos, yv_votos)
# print("Rendimiento en test: ", rendimiento(lr_votos, Xt_votos, yt_votos))
# # Guardamos los pesos como resultado del modelo entrenado
# pesos_votos = lr_votos.w
# np.savetxt("pesos_votos.txt", pesos_votos)

# RESULTADOS
# Rendimiento en entrenamiento:  0.9953917050691244
# Rendimiento en validación:  0.9540229885057471
# Mejor combinación de hiperparámetros:  (1.0, 32, True, True, 5000)
# Rendimiento en test:  0.9312977099236641

# Es decir, el modelo con mejor rendimiento para este conjunto de datos,
# es el que tiene los siguientes hiperparámetros:
# - Tasa de aprendizaje inicial de 1.0 con rate_decay
# - Tamaño de batch: 32
# - Normalización: True
# - Número de epochs: 5000

######################
### Cáncer de Mama ###
######################

# Imporatmos los datos y los dividimos sando train_test_split de Scikit Learn con estratificación
# en una proporción 50-20-30 respectivamente
from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
X_cancer,y_cancer=cancer.data,cancer.target
Xe_cancer, Xaux_cancer, ye_cancer, yaux_cancer = train_test_split(X_cancer, y_cancer, test_size=0.5, random_state=41, stratify=y_cancer)
Xv_cancer, Xt_cancer, yv_cancer, yt_cancer = train_test_split(Xaux_cancer, yaux_cancer, test_size=0.6, random_state=41, stratify=yaux_cancer)

# Evaluamos el rendimiento de cada combinación de hiperparámetros
print("\nCáncer de mama")
lr_cancer = evaluacion_hiperparametros(opciones, Xe_cancer, ye_cancer, Xv_cancer, yv_cancer)
print("Rendimiento en test: ", rendimiento(lr_cancer, Xt_cancer, yt_cancer))

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

######################
### Críticas IMDB ###
######################































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

