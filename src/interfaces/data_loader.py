from abc import ABC, abstractmethod


class DataLoader(ABC):
    """Clase abstracta de la que deben heredar todas las clases encargadas de cargar los datos de un dataset"""

    @abstractmethod
    def get_data(self):
        """Devuelve los datos obtenidos del dataset cargado. Ha de ser implementado por las subclases.

        Returns
        -------
        tuple
            una tupla con los datos de entrenamiento y los datos de test, en ese mismo orden.
        """
        pass
