from interfaces import data_loader as dl
import tensorflow as tf


class SingleImageLoader(dl.DataLoader):
    """Clase encargada de cargar datos de un dataset. En concreto esta clase se encarga de cargar una única imagen
    localizada en la ruta indicada"""
    def __init__(self, image_path=None):
        """
        Parameters
        ----------
        image_path : str
            ruta de la imagen a cargar
        """
        self.datasetroot_path = image_path

    def get_data(self):
        """Devuelve los datos obtenidos del dataset cargado.

        Returns
        -------
        tuple
            una tupla con los datos de entrenamiento y los datos de test, en ese mismo orden.
        """
        train_paths = []
        test_paths = [self.datasetroot_path]
        train_dataset = tf.data.Dataset.from_tensor_slices(train_paths)
        test_dataset = tf.data.Dataset.from_tensor_slices(test_paths)
        return train_dataset, test_dataset
