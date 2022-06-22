from networks.first_vae_net import FirstVAENet
import tensorflow as tf
from networks.net_factory import NetFactory
from util.loggingutil import LoggingUtil as Lu


class FirstVaeNetFactory(NetFactory):
    """Subclase de NetFactory encargada de implementar un variatonal autoencoder."""

    def __init__(self, input_shape, optimizer, loss, test_name):
        """
        Parameters
        ----------
        input_shape : tuple
            tupla con las imensiones de la entrada

        optimizer : Optimizer
            Optimizador a utilizar

        loss : Loss
            Loss a utilizar

        test_name : str
            nombre del test concreto que se está realizando sobre este modelo. Sirve para nombrar las carpetas.
        """

        self.input_shape = input_shape
        self.latent_dim = 4096
        self.optimizer = optimizer
        self.loss = loss
        self.model_name = FirstVaeNetFactory.__name__
        self.test_name = test_name
        self.logger = Lu.get_looger("Neural Network Factory")

    def create_nn(self):
        """ Crea un variatonal autoencoder

        Returns
        -------
        Network
            Devuelve una FirstVAENet ya configurada con la que poder gestionar la red neuronal ya construida.
        """
        g = tf.Graph()
        run_meta = tf.compat.v1.RunMetadata()
        with g.as_default():
            encoder, conv_shape = self._encoder_model(batch_size=1)
            decoder = self._decoder_model(conv_shape=conv_shape, batch_size=1)
            vae = self._vae_model(encoder, decoder, batch_size=1)

        flops = NetFactory._estimate_flops(g, run_meta)
        self.logger.info("Se ha creado una red neuronal cuyos flops totales son: " + str(flops))

        encoder, conv_shape = self._encoder_model()
        decoder = self._decoder_model(conv_shape=conv_shape)
        vae = self._vae_model(encoder, decoder)

        return FirstVAENet(self.input_shape, self.optimizer, self.loss,
                           self.model_name, self.test_name, encoder, decoder, vae)

    @staticmethod
    def get_custom_objects():
        """
        Returns
        -------
        Network
            Devuelve un diccionario con los objetos personalizados utilizados al construir la red.
        """
        return {'Sampling': FirstVaeNetFactory.Sampling}


    class Sampling(tf.keras.layers.Layer):
        def __init__(self):
            super().__init__()

        def call(self, inputs):
            """ Genera una muestra aleatoria de la distribución, media y desviación tipica, recibidas.

            Parameters
            ----------
            inputs : tuple
                tupla con 2 tensors uno con la media y otro con la deviación típica.


            Returns
            -------
            Tensor
                tensor con una muestra aleatoria de la distribución recibida (espacio latente).
            """

            mu, sigma = inputs
            batch = tf.shape(mu)[0]
            dim = tf.shape(mu)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            z = mu + tf.exp(sigma) * epsilon

            return z

    def _encoder_layers(self, inputs):
        """ Define las capas del codificador.

        Parameters
        ----------
        inputs : Tensor
            tensor con los datos de entrada


        Returns
        -------
        tuple
            tupla conteniendo el tensor con la media (mu), el tensor con la desviación tipica (sigma) y el tamaño de la
            ultima capa previa a las completamente conectadas de mu y sigma.
        """
        # add the Conv2D layers followed by BatchNormalization
        x = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=2, padding="same", activation='relu',
                                   name="encode_conv1")(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same', activation='relu',
                                   name="encode_conv2")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same', activation='relu',
                                   name="encode_conv3")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='same', activation='relu',
                                   name="encode_conv4")(x)

        # assign to a different variable so you can extract the shape later
        batch_3 = tf.keras.layers.BatchNormalization()(x)

        # flatten the features and feed into the Dense network
        x = tf.keras.layers.Flatten(name="encode_flatten")(batch_3)

        # we arbitrarily used 20 units here but feel free to change and see what results you get
        x = tf.keras.layers.Dense(4096, activation='relu', name="encode_dense")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        # add output Dense networks for mu and sigma, units equal to the declared latent_dim.
        mu = tf.keras.layers.Dense(self.latent_dim, name='latent_mu')(x)
        sigma = tf.keras.layers.Dense(self.latent_dim, name='latent_sigma')(x)

        # revise `batch_3.shape` here if you opted not to use 3 Conv2D layers
        return mu, sigma, batch_3.shape

    def _encoder_model(self, batch_size=None):
        """ Define el modelo del codificador.

        Parameters
        ----------
        batch_size : int, optional
            permite indicar al modelo cual va a ser el tamaño del batch.

        Returns
        -------
        tuple
            el modelo representando el codificador del autoencoder y el tamaño de la última capa de convolución.
        """
        inputs = tf.keras.layers.Input(batch_size=batch_size, shape=self.input_shape)
        mu, sigma, conv_shape = self._encoder_layers(inputs)
        z = FirstVaeNetFactory.Sampling().call((mu, sigma))
        model = tf.keras.Model(inputs, outputs=[mu, sigma, z])
        self.logger.info(NetFactory.show_model(model))
        return model, conv_shape

    def _decoder_layers(self, inputs, conv_shape):
        """ Define las capas del decodificador.

        Parameters
        ----------
        inputs : Tensor
            tensor con los datos de entrada


        Returns
        -------
        tuple
            tupla conteniendo el tensor con la media (mu), el tensor con la desviación tipica (sigma) y el tamaño de la
            ultima capa previa a las completamente conectadas de mu y sigma.
        """

        # feed to a Dense network with units computed from the conv_shape dimensions
        units = conv_shape[1] * conv_shape[2] * conv_shape[3]
        x = tf.keras.layers.Dense(units, activation='relu', name="decode_dense1")(inputs)
        x = tf.keras.layers.BatchNormalization()(x)

        # reshape output using the conv_shape dimensions
        x = tf.keras.layers.Reshape((conv_shape[1], conv_shape[2], conv_shape[3]), name="decode_reshape")(x)

        # upsample the features back to the original dimensions
        x = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same', activation='relu',
                                            name="decode_conv2d_2")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu',
                                            name="decode_conv2d_3")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu',
                                            name="decode_conv2d_4")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same', activation='relu',
                                            name="decode_conv2d_5")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=3, strides=1, padding='same', activation='sigmoid',
                                            name="decode_final")(x)


        return x

    def _decoder_model(self, conv_shape, batch_size=None):
        """ Define el modelo del decodificador.

        Parameters
        ----------
        conv_shape : tuple
            dimensiones de la última capa de convolución del codificador.

        batch_size : int, opcional
            permite indicar al modelo cual va a ser el tamaño del batch.

        Returns
        -------
        Model
            el modelo representando el decodificador del autoencoder.
        """
        inputs = tf.keras.layers.Input(batch_size=batch_size, shape=(self.latent_dim,))
        outputs = self._decoder_layers(inputs, conv_shape)
        model = tf.keras.Model(inputs, outputs)

        self.logger.info(NetFactory.show_model(model))
        return model

    def _kl_reconstruction_loss(self, mu, sigma):
        """ Calcula la divergencia Kullback-Leibler

        Parameters
        ----------
        mu : tensor
            tensor conteniendo las medias

        sigma : tensor
            tensor conteniendo las desviaciones típicas

        Returns
        -------
        float
            el resultado de calcular la divergencia KL.
        """
        return -0.5 * (tf.reduce_mean(1 + sigma - tf.square(mu) - tf.exp(sigma)))

    def _vae_model(self, encoder, decoder, batch_size=None):
        """ Define el modelo del VAE.

        Parameters
        ----------
        encoder : Model
            el codificador del VAE

        decoder : Model
            el decodificador del VAE

        batch_size : int, opcional
            permite indicar al modelo cual va a ser el tamaño del batch.

        Returns
        -------
        Model
            el modelo representando el decodificador del autoencoder.
        """
        # set the inputs
        inputs = tf.keras.layers.Input(batch_size=batch_size, shape=self.input_shape)

        # get mu, sigma, and z from the encoder output
        mu, sigma, z = encoder(inputs)

        # get reconstructed output from the decoder
        reconstructed = decoder(z)

        # define the inputs and outputs of the VAE
        model = tf.keras.Model(inputs=inputs, outputs=reconstructed)

        # add the KL loss
        loss = self._kl_reconstruction_loss(mu, sigma)
        model.add_loss(loss)

        return model
