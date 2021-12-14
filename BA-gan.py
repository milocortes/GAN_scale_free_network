# BA-GAN
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from keras.datasets import mnist
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Reshape,BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import pandas as pd
import powerlaw
import networkx as nx
import seaborn as sns
import time

A_dim = 250
channels = 1

# Dimesión de la matriz de adyacencia de entrada
A_shape = (A_dim, A_dim, channels)

# Tamaño del vector de ruido utilizado como entrada en el generador
z_dim = 1200

## Definimos el Generador
def build_generator(A_shape, z_dim):

    model = Sequential()

    # Capa fully connected
    model.add(Dense(1000, input_dim=z_dim))

    # Batch normalization
    model.add(BatchNormalization())

    # Función de activación ReLU
    model.add(tf.keras.layers.LeakyReLU(alpha=0.01))

    # Capa fully connected
    model.add(Dense(1000))

    # Batch normalization
    model.add(BatchNormalization())

    # Función de activación ReLU
    model.add(tf.keras.layers.LeakyReLU(alpha=0.01))

    # Capa de salida con función de activación sigmoide
    model.add(Dense(A_dim * A_dim * 1, activation='tanh'))

    # Reshape the Generator output to image dimensions
    model.add(Reshape(A_shape))

    return model

## Definimos el Discriminador
def build_discriminator(A_shape):

    model = Sequential()

    # Aplanamos la matriz de adyacencia de entrada
    model.add(Flatten(input_shape=A_shape))

    # Capa fully connected
    model.add(Dense(1000))

    # Función de activación ReLU
    model.add(tf.keras.layers.LeakyReLU(alpha=0.01))

    # Capa de salida con función de activación sigmoide
    model.add(Dense(1, activation='sigmoid'))

    return model

## Definimos el modelo
def build_gan(generator, discriminator):

    model = Sequential()

    # Agregamos el Generador y el Discriminador
    model.add(generator)
    model.add(discriminator)

    return model

# Construimos y compilamos el Discriminador
discriminator = build_discriminator(A_shape)
discriminator.compile(loss='binary_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])

# Construimos el Generador
generator = build_generator(A_shape, z_dim)

# Mantenemos los parámetros del Discriminador contantes durante el entrenamiento del Generador
discriminator.trainable = False

# Construimos y compilamos el GAN manteniendo fijo el Discriminador para entrenar al Generador
gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=Adam())

# Listas de pérdidas y accuracies del entrenamiento
losses = []
accuracies = []
iteration_checkpoints = []


def train(iterations, batch_size, sample_interval):

    # Generamos modelos BA
    X_train = []
    for ba in range(3000):
        print(ba)
        BA =nx.barabasi_albert_graph(A_dim,3)
        X_train.append(np.array(nx.adjacency_matrix(BA).todense()))

    X_train = np.array(X_train)
    X_train = np.expand_dims(X_train, axis=3)

    # Etiquetas para las matrices de adyacencia del modelo BA: todas son uno
    real = np.ones((batch_size, 1))

    # Etiquetas para las matrices generadas por el Discriminador: todas son cero
    fake = np.zeros((batch_size, 1))

    for iteration in range(iterations):

        # -------------------------
        #  Entrenamos al Discriminador
        # -------------------------

        # Obtenemos un lote aleatorio de las matrices de adyancencia reales
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]

        # Generamos un lote aleatorio de las matrices de adyancencia fake
        z = np.random.normal(0, 1, (batch_size, 1200))
        gen_imgs = generator.predict(z)

        # Entrenamiento del Discriminador
        d_loss_real = discriminator.train_on_batch(imgs, real)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss, accuracy = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Entrenamos al Generador
        # ---------------------

        # Generamos un lote aleatorio de las matrices de adyancencia fake
        z = np.random.normal(0, 1, (batch_size, 1200))
        gen_imgs = generator.predict(z)

        # Entrenamiento del Generador
        g_loss = gan.train_on_batch(z, real)

        if (iteration + 1) % sample_interval == 0:

            # Guardamos las pérdidas y accuracies
            losses.append((d_loss, g_loss))
            accuracies.append(100.0 * accuracy)
            iteration_checkpoints.append(iteration + 1)

            # Imprimimos el progreso del proceso de entrenamiento
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
                  (iteration + 1, d_loss, 100.0 * accuracy, g_loss))


# Entrenamos el modelo GAN
# Hiperparámetros
iterations = 2000
batch_size = 100
sample_interval = 10

start_time = time.time()
train(iterations, batch_size, sample_interval)
print("--- Entrenamiento: %s seconds ---" % (time.time() - start_time))


# Generamos matrices de adyacencia a partir de un ruido aleatorio
# para calcular la pendiente de la distribución de grado en la
# escala log-log
muestra_gan = 1000
z = np.random.normal(0, 1, (muestra_gan, 1200))

gen_imgs = generator.predict(z)
resultados = []

for i in range(muestra_gan):
    BA_mat = np.where(gen_imgs[i] < 0.2, gen_imgs[i],1)
    BA_mat = np.where(BA_mat== 1, BA_mat,0)
    BA_mat = BA_mat.reshape((A_dim,A_dim))

    G = nx.from_numpy_matrix(BA_mat)

    degrees = G.degree() # dictionary node:degree
    values = sorted(set(dict(degrees).values()))
    hist = [list(dict(degrees).values()).count(x)/A_dim for x in values]

    try:
        X = np.array([np.log(x+1) for x in values]).reshape((-1, 1))
        y = np.array([np.log(y) for y in hist])
        reg = LinearRegression().fit(X, y)
        reg.score(X, y)
        results = powerlaw.Fit(hist)
        print(results.power_law.alpha)
        resultados.append(results.power_law.alpha)
    except:
        print("No se  puede calcular")

np.mean(resultados)

# Generamos matrices de adyacencia del modelo BA para calcular la pendiente
# de la distribución de grado en la escala log-log
BA_pendiente = []

for i in range(muestra_gan):
    BA =nx.barabasi_albert_graph(A_dim,3)

    degrees = BA.degree()
    values = sorted(set(dict(degrees).values()))
    hist = [list(dict(degrees).values()).count(x)/A_dim for x in values]

    try:
        X = np.array([np.log(x+1) for x in values]).reshape((-1, 1))
        y = np.array([np.log(y) for y in hist])
        reg = LinearRegression().fit(X, y)
        reg.score(X, y)
        results = powerlaw.Fit(hist)
        print(results.power_law.alpha)
        BA_pendiente.append(results.power_law.alpha)
    except:
        print("No se  puede calcular")

np.mean(BA_pendiente)

## Guardamos los resultados de las pendientes para graficarlas en ggplot2
pendiente_pd = pd.DataFrame.from_dict({'BA':BA_pendiente,'GAN':resultados})
pendiente_pd.to_csv("BA_GAN_pendientes.csv",index = False)

'''
## Graficamos
BA_mat = np.where(gen_imgs[19] < 0.25,gen_imgs[19],1)
BA_mat = np.where(BA_mat== 1, BA_mat,0)
BA_mat = BA_mat.reshape((A_dim,A_dim))
plt.imshow(BA_mat)
plt.show()

plt.imshow(nx.adjacency_matrix(BA).todense())
plt.show()


G = nx.from_numpy_matrix(BA_mat)

degrees = G.degree() # dictionary node:degree
values = sorted(set(dict(degrees).values()))
hist = [list(dict(degrees).values()).count(x)/250 for x in values]


plt.figure()
plt.loglog(values,hist,'ro-')
plt.xlabel('Degree')
plt.ylabel('Number of nodes')
plt.title('network of places in Cambridge')
plt.show()

X = np.array([np.log(x+1) for x in values]).reshape((-1, 1))
# y = 1 * x_0 + 2 * x_1 + 3
y = np.array([np.log(y) for y in hist])
reg = LinearRegression().fit(X, y)
reg.score(X, y)

reg.coef_
sns.kdeplot(gen_imgs[1].reshape((A_dim,A_dim)).reshape(1,62500).tolist()[0])
'''
