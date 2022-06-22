from interfaces.network import NeuralNetwork
import tensorflow as tf
from util.plot_util import Display as Dis
from globalprops import GlobalProps as Gp
from util.loggingutil import LoggingUtil as Lu
import numpy as np


class AeNet(NeuralNetwork):
    """Clase encargada de gestionar los autoencoders"""

    def __init__(self, input_shape, optimizer, loss, model_name, test_name, encoder, decoder, ae):
        """
        Parameters
        ----------
        input_shape : tuple
            tupla con las imensiones de la entrada

        optimizer : Optimizer
            Optimizador a utilizar

        loss : Loss
            Loss a utilizar

        model_name : str
            nombre del modelo. Este se utilizará para nombrar las carpetas en la que se guardan los resultados.

        test_name : str
            nombre del test concreto que se está realizando sobre este modelo. Sirve para nombrar las carpetas.

        encoder : Model
            Encoder a utilizar por el autoencoder

        decoder : Model
            decoder a utilizar por el autoencoder

        ae : Model
            modelo completo con el encoder y el decoder
        """

        self.klass = self.__class__

        self.input_shape = input_shape
        self.latent_dim = encoder.output_shape[1:]
        self.optimizer = optimizer
        self.loss = loss
        self.model_name = model_name
        self.test_name = test_name

        self.encoder = encoder
        self.decoder = decoder
        self.ae = ae
        self.logger = Lu.get_looger("Neural network")

        self.history = {}

        self.train_mean_loss = tf.keras.metrics.Mean("Train Mean loss")
        self.valid_mean_loss = tf.keras.metrics.Mean("Valid Mean loss")



    @property
    def metrics(self):
        """ Lista con todas las metricas que se quieren visualizar durante el entrenamiento."""
        return [self.train_mean_loss, self.valid_mean_loss]

    def train(self, train_dataset, num_epochs, test_dataset=None):
        """ Entrena la red neuronal durante el número de épocas indicado y con los datos de entranamiento indicados.
        Se pueden incorporar datos de validación para que se testee la red durante el entrenamiento.

        Parameters
        ----------
        train_dataset : BatchDataset
            dataset con los datos de entrenamiento.

        num_epochs : int
            número de épocas del entrenamiento.

        test_dataset : BatchDataset
            dataset con los datos de validación.
        """
        self._init_history()
        for epoch in range(num_epochs):
            self.logger.info('Start of epoch %d' % (epoch,))

            # Iterate over the batches of the dataset.

            for step, x_batch_train in enumerate(train_dataset):
                self._train_step(x_batch_train)
                if test_dataset is not None:
                    for y_batch_test in test_dataset:
                        self._test_step(y_batch_test)
                        break
                self._show_step_metrics(epoch, step)

            self._update_history()
            self.predict(train_dataset, f"result_train_{epoch}", 1)
            if epoch % (num_epochs // min(5,num_epochs)) == 0 or epoch == num_epochs-1:
                self.save_model(Gp.get_instance().paths["network_path"] + "\\" + str(epoch))
                Dis.generateHistoricLossGraphic(self.history, Gp.get_instance().paths["final_result_path"],
                                                "Evolución_del_error.png")


    def _train_step(self, batch_data):
        with tf.GradientTape() as tape:
            reconstructed = self.ae(batch_data)
            loss = self.loss(batch_data, reconstructed)

        grads = tape.gradient(loss, self.ae.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.ae.trainable_weights))

        self.train_mean_loss.update_state(loss)

    def _test_step(self, batch_data):
        prediction = self.ae(batch_data)
        loss = self.loss(batch_data, prediction)
        self.valid_mean_loss.update_state(loss)

    def predict(self, dataset, result_name, num_batch, is_final=False):
        """ Precide el número de lotes indicado de las imágenes del dataset indicado. Guarda los resultados con el
        nombre indicado.

        Parameters
        ----------
        dataset : BatchDataset
            dataset con los datos que se quieren predecir.

        result_name : str
            Nombre con el que se guardarán los resultados.

        num_batch : int
            número de batch que se van a predecir.

        is_final : bool
            bandera con la indicar en que carpeta se quieren guardar los resultados, en la carpeta final o en la parcial
        """
        partial_psnr = 0
        partial_ssim = 0
        total = 0
        id1 = 0
        id2 = 0
        for subset in dataset.take(num_batch):
            predictions = self.ae.predict(subset)
            psnr = np.mean(tf.image.psnr(subset, predictions, max_val=1.0).numpy())
            ssim = np.mean(tf.image.ssim(subset, predictions, max_val=1.0).numpy())
            partial_psnr += psnr
            partial_ssim += ssim
            self.logger.debug("El PSNR ES: " + str(psnr))
            self.logger.debug("El SSIM ES: " + str(ssim))
            idxs = np.random.choice(len(subset), size=10)
            Dis.display_results([subset[i] for i in idxs], [predictions[i] for i in idxs], subset.shape[1:],
                                Gp.get_instance().paths["final_result_path" if is_final else "partial_result_path"],
                                result_name)

            for prediction in predictions:
                Dis.save_image(Gp.get_instance().paths["partial_result_path"] + "\\" + str(id1) + ".png", prediction)
                id1 += 1
            for original in subset:
                Dis.save_image(Gp.get_instance().paths["partial_result_path"] + "\\" + str(id2) + "_original.png", original)
                id2 += 1
            total += 1
        self.logger.debug("El PSNR MEDIO ES: " + str(partial_psnr/total))
        self.logger.debug("El SSIM MEDIO ES: " + str(partial_ssim/total))

    def crop(self, images, x, y, width, height):
        """ Corta las imagenes en la coordenadas indicadas. Se utiliza para poder hacer zoom a partes concretas de
        una imagen.

        Parameters
        ----------
        images : BatchDataset
            dataset con las imagenes que se quieren recortar.

        x : int
            coordenada X del punto de la esquina superior izquierda del rectángulo a recortar.

        y : int
            coordenada Y del punto de la esquina superior izquierda del rectángulo a recortar.

        width : int
            ancho del área a recortar

        height : int
            alto del área a recortar
        """
        id = 0
        for batch in images:
            cropped_images = tf.image.crop_to_bounding_box(batch,  y, x, height, width)
            for image in cropped_images:
                Dis.save_image(Gp.get_instance().paths["partial_result_path"] + "\\" + str(id) + ".png", image)
                id += 1


    def _get_class_data(self):
        return {
            "variables": [
                self.klass,
                self.input_shape,
                tf.keras.optimizers.serialize(self.optimizer),
                tf.keras.losses.serialize(self.loss),
                self.model_name,
                self.test_name
            ],
            "modelos": {
                "encoder": self.encoder,
                "decoder": self.decoder,
                "ae": self.ae
            }
        }

    @staticmethod
    def _create_from_file(class_data):
        input_shape, ser_optimizer, ser_loss, compressor_name, test_name = class_data["variables"]
        modelos = class_data["modelos"]
        return AeNet(input_shape, tf.keras.optimizers.deserialize(ser_optimizer),
                     tf.keras.losses.deserialize(ser_loss),
                     compressor_name, test_name, modelos["encoder.h5"], modelos["decoder.h5"], modelos["ae.h5"])