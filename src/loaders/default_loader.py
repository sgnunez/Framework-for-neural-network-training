from interfaces import data_loader as dl
import os
import tensorflow as tf
from globalprops import GlobalProps as Gp


class DefaultLoader(dl.DataLoader):
    """Clase encargada de cargar datos de un dataset. En concreto esta clase se encarga de cargar de forma genérica
    cualquier dataset localizado en una determinada ruta"""

    def __init__(self, dataset_name=None, train_percentage=None):
        """
        Parameters
        ----------
        dataset_name : str
            ruta del dataset a cargar

        train_percentage : float
            porcentaje del dataset que se desea utilizar para entrenar la red. El resto se destinará a tests.
        """
        self.datasetroot_path = Gp.get_instance().paths["dataset_path"] + "\\" + dataset_name
        self.train_percentage = train_percentage

    def get_data(self):
        """Devuelve los datos obtenidos del dataset cargado.

        Returns
        -------
        tuple
            una tupla con los datos de entrenamiento y los datos de test, en ese mismo orden.
        """
        image_paths = [os.path.join(self.datasetroot_path, fname) for fname in os.listdir(self.datasetroot_path)]
        train_paths = image_paths[:int(len(image_paths) * self.train_percentage)]
        test_paths = image_paths[int(len(image_paths) * (self.train_percentage)):]
        train_dataset = tf.data.Dataset.from_tensor_slices(train_paths)
        test_dataset = tf.data.Dataset.from_tensor_slices(test_paths)
        return train_dataset, test_dataset
