import tensorflow as tf
from networks.ae_net import AeNet
from networks.net_factory import NetFactory
from util.loggingutil import LoggingUtil as Lu


class PersonasAeFactory(NetFactory):
    """Subclase de NetFactory encargada de implementar un autoencoder."""

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
        self.latent_dim = (64, 64, 4)
        self.optimizer = optimizer
        self.loss = loss
        self.model_name = PersonasAeFactory.__name__
        self.test_name = test_name
        self.logger = Lu.get_looger("Personas AE Network Factory")

    def create_nn(self):
        """ Crea un autoencoder

        Returns
        -------
        Network
            Devuelve una AeNet ya configurada con la que poder gestionar la red neuronal ya construida.
        """
        g = tf.Graph()
        run_meta = tf.compat.v1.RunMetadata()
        with g.as_default():
            encoder = self._encoder_model(batch_size=1)
            decoder = self._decoder_model(batch_size=1)
            vae = self._vae_model(encoder, decoder, batch_size=1)

        flops = NetFactory._estimate_flops(g, run_meta)
        self.logger.info("Se ha creado una red neuronal cuyos flops totales son: " + str(flops))

        encoder = self._encoder_model()
        decoder = self._decoder_model()
        vae = self._vae_model(encoder, decoder)

        return AeNet(self.input_shape, self.optimizer, self.loss,
                     self.model_name, self.test_name, encoder, decoder, vae)

    @staticmethod
    def get_custom_objects():
        """
        Returns
        -------
        Network
            Devuelve un diccionario con los objetos personalizados utilizados al construir la red.
        """
        return {}


    def _encoder_layers(self, inputs):
        """ Define las capas del codificador.

        Parameters
        ----------
        inputs : Tensor
            tensor con los datos de entrada


        Returns
        -------
        Layer
            Capas de codificador.
        """

        x = tf.keras.layers.Conv2D(filters=8, kernel_size=3, strides=2, padding='same', activation='relu',
                                   name="encode_conv1")(inputs)
        x = tf.keras.layers.Conv2D(filters=4, kernel_size=3, strides=2, padding='same', activation='relu',
                                   name="encode_conv2")(x)
        return x

    def _encoder_model(self, batch_size=None):
        """ Define el modelo del codificador.

        Parameters
        ----------
        batch_size : int, optional
            permite indicar al modelo cual va a ser el tamaño del batch.

        Returns
        -------
        Model
            el modelo representando el codificador del autoencoder.
        """
        inputs = tf.keras.layers.Input(batch_size=batch_size,shape=self.input_shape)
        encoder_layers = self._encoder_layers(inputs)
        model = tf.keras.Model(inputs, outputs=encoder_layers)
        self.logger.info(NetFactory.show_model(model))

        return model

    def _decoder_layers(self, inputs):
        """ Define las capas del decodificador.

        Parameters
        ----------
        inputs : Tensor
            tensor con los datos de entrada


        Returns
        -------
        Layer
            las capas del decodificador
        """

        x = tf.keras.layers.Conv2DTranspose(filters=4, kernel_size=3, strides=2, padding="same", activation='relu',
                                   name="encode_conv1")(inputs)
        x = tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=3, strides=2, padding='same', activation='relu',
                                   name="encode_conv2")(x)
        x = tf.keras.layers.Conv2D(filters=3, kernel_size=3, strides=1, padding='same', activation='sigmoid',
                                   name="encode_conv3")(x)

        return x

    def _decoder_model(self, batch_size=None):
        """ Define el modelo del decodificador.

        Parameters
        ----------
        batch_size : int, opcional
            permite indicar al modelo cual va a ser el tamaño del batch.

        Returns
        -------
        Model
            el modelo representando el decodificador del autoencoder.
        """
        inputs = tf.keras.layers.Input(batch_size=batch_size,shape=self.latent_dim)
        outputs = self._decoder_layers(inputs)
        model = tf.keras.Model(inputs, outputs)
        self.logger.info(NetFactory.show_model(model))
        return model

    def _vae_model(self, encoder, decoder,  batch_size=None):
        """ Define el modelo del autoencoder.

        Parameters
        ----------
        encoder : Model
            el codificador del AE

        decoder : Model
            el decodificador del AE

        batch_size : int, opcional
            permite indicar al modelo cual va a ser el tamaño del batch.

        Returns
        -------
        Model
            el modelo representando el decodificador del autoencoder.
        """

        # set the inputs
        inputs = tf.keras.layers.Input(batch_size=batch_size,shape=self.input_shape)

        code = encoder(inputs)

        # get reconstructed output from the decoder
        reconstructed = decoder(code)

        # define the inputs and outputs of the VAE
        model = tf.keras.Model(inputs=inputs, outputs=reconstructed)

        return model
