from interfaces.network import NeuralNetwork
import tensorflow as tf
from util.plot_util import Display as Dis
from globalprops import GlobalProps as Gp
from util.loggingutil import LoggingUtil as Lu
import numpy as np


class FirstVAENet(NeuralNetwork):
    """Clase encargada de gestionar los variatonal autoencoders"""

    def __init__(self, input_shape, optimizer, loss, model_name, test_name, encoder, decoder, vae):
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

        vae : Model
            modelo completo con el encoder y el decoder
        """
        self.klass = self.__class__

        self.input_shape = input_shape
        self.latent_dim = 4096
        self.optimizer = optimizer
        self.loss = loss
        self.model_name = model_name
        self.test_name = test_name

        self.encoder = encoder
        self.decoder = decoder
        self.vae = vae
        self.logger = Lu.get_looger("Neural network")

        self.history = {}

        self.train_mean_loss = tf.keras.metrics.Mean("Train Mean loss")
        self.valid_mean_loss = tf.keras.metrics.Mean("Valid Mean loss")

    @property
    def metrics(self):
        """ Lista con todas las metricas que se quieren visualizar durante el entrenamiento."""
        return [self.train_mean_loss, self.valid_mean_loss]

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
                "vae": self.vae
            }
        }
    @staticmethod
    def _create_from_file(class_data):
        input_shape, ser_optimizer, ser_loss, model_name, test_name = class_data["variables"]
        modelos = class_data["modelos"]
        return FirstVAENet(input_shape, tf.keras.optimizers.deserialize(ser_optimizer),
                           tf.keras.losses.deserialize(ser_loss),
                           model_name, test_name, modelos["encoder.h5"], modelos["decoder.h5"], modelos["vae.h5"])


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

        result_path = Gp.get_instance().paths["partial_result_path"]
        random_vector_for_generation = tf.random.normal(shape=[16, self.latent_dim])
        Dis.generate_and_save_images(self.decoder, random_vector_for_generation, result_path, "epoch_0_etep_0")

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
            Dis.generate_and_save_images(self.decoder, random_vector_for_generation, result_path,
                                         f"epoch_{epoch}_etep_{step}")

            if epoch % (num_epochs // min(5, num_epochs)) == 0 or epoch == num_epochs - 1:
                self.save_model(Gp.get_instance().paths["network_path"] + "\\" + str(epoch))

                Dis.generateHistoricLossGraphic(self.history, Gp.get_instance().paths["final_result_path"],
                                                "Evolución_del_error.png")

    def _train_step(self, batch_data):
        with tf.GradientTape() as tape:
            reconstructed = self.vae(batch_data)
            # Compute reconstruction loss
            flattened_inputs = tf.reshape(batch_data, shape=[-1])
            flattened_outputs = tf.reshape(reconstructed, shape=[-1])

            loss = self.loss(flattened_inputs, flattened_outputs)
            k1_loss = sum(self.vae.losses)
            total_loss = (loss + k1_loss)/2

        grads = tape.gradient(total_loss, self.vae.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.vae.trainable_weights))

        self.train_mean_loss.update_state(total_loss)

    def _test_step(self, batch_data):
        prediction = self.vae(batch_data)
        flattened_inputs = tf.reshape(batch_data, shape=[-1])
        flattened_outputs = tf.reshape(prediction, shape=[-1])

        loss = self.loss(flattened_inputs, flattened_outputs)
        k1_loss = sum(self.vae.losses)
        total_loss = (loss + k1_loss) / 2

        self.valid_mean_loss.update_state(total_loss)

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
        for num, subset in enumerate(dataset.take(num_batch)):
            predictions = self.vae.predict(subset)
            psnr = np.mean(tf.image.psnr(subset, predictions, max_val=1.0).numpy())
            ssim = np.mean(tf.image.ssim(subset, predictions, max_val=1.0).numpy())
            partial_psnr += psnr
            partial_ssim += ssim
            self.logger.debug("El PSNR ES: " + str(psnr))
            self.logger.debug("El SSIM ES: " + str(ssim))
            idxs = np.random.choice(len(subset) - 1, size=10)
            Dis.display_results([subset[i] for i in idxs], [predictions[i] for i in idxs], subset.shape[1:],
                                Gp.get_instance().paths["final_result_path" if is_final else "partial_result_path"], result_name+"_"+str(num))
            if is_final:
                for prediction in predictions:
                    Dis.save_image(Gp.get_instance().paths["partial_result_path"] + "\\" + str(id1) + ".png", prediction)
                    id1 += 1
                for original in subset:
                    Dis.save_image(
                        Gp.get_instance().paths["partial_result_path"] + "\\" + str(id2) + "_original.png", original)
                    id2 += 1
            total += 1
        self.logger.debug("El PSNR MEDIO ES: " + str(partial_psnr / total))
        self.logger.debug("El SSIM MEDIO ES: " + str(partial_ssim / total))

