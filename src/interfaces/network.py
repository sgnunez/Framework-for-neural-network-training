from abc import ABC, abstractmethod
import pickle
import tensorflow as tf
import os


class NeuralNetwork(ABC):
    """Clase abstracta que se encargada del trabajo con un tipo de red neuronal. Las subclases deben implementar los
    métodos abstractos para permitir el correcto manejo de la red neuronal."""

    @staticmethod
    def load_model(path, custom_objects):
        """ Carga en memoria la red neuronal guardada en la ruta indicada.

        Parameters
        ----------
        path : str
            carpeta en la que se encuentran toods los ficheros necesarios para cargar la red neuronal

        custom_objects: dict
            diccionario con el nombre y la clase de todas las clase personalizadas que utilice la red neuronal, es decir,
            todas las clases que no pertenezcan a TensorFlow.

        Returns
        -------
        NeuralNetwork
            Devuelve una instancia de la subclase que de NeuralNetwork a la que correspondan los ficheros cargados.
        """

        files = os.listdir(path)
        models = {}
        for file in files:
            if file.endswith('.h5'):
                full_path = path + "\\" + file
                models[file] = tf.keras.models.load_model(full_path, custom_objects=custom_objects)

        with open(path + "\\variables.nn", 'rb') as load_file:
            variables = pickle.load(load_file)

            return variables[0]._create_from_file({"modelos": models, "variables": variables[1:]})

    def save_model(self, path):
        """ Guarda la red neuronal en la ruta indicada.

        Parameters
        ----------
        path : str
            carpeta en la que deben guardar todos los ficheros que se generan al guardar la red neuronal.

        Returns
        -------
        NeuralNetwork
            Devuelve una instancia de la subclase que de NeuralNetwork a la que correspondan los ficheros cargados.
        """
        class_data = self._get_class_data()
        models = class_data["modelos"]
        variables = class_data["variables"]

        for model in models.items():
            model[1].save(f"{path}\\{model[0]}.h5")

        with open(path + "\\variables.nn", 'wb') as save_file:
            pickle.dump(variables, save_file)

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def _get_class_data(self):
        pass

    @staticmethod
    @abstractmethod
    def _create_from_file(class_data):
        pass

    def _init_history(self):
        for metric in self.metrics:
            self.history[metric.name] = []


    def _update_history(self):
        for metric in self.metrics:
            self.history[metric.name].append(metric.result().numpy())
            metric.reset_state()

    def _show_step_metrics(self, epoch, step):
        tolog= f"Epoch: {epoch} step: {step}"
        for metric in self.metrics:
            tolog += f" {metric.name}: {metric.result().numpy()}"
        self.logger.info(tolog)