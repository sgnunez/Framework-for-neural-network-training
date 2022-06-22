from abc import ABC, abstractmethod
import tensorflow as tf


class NetFactory(ABC):
    """Clase abstracta que permite construir una red neuronal. Las subclases se deben encargar de definir la red
    neuronal a crear implementado los diferentes métodos abstractos."""

    @abstractmethod
    def create_nn(self):
        """ Crea la red neuronal

        Returns
        -------
        Network
            Devuelve una Network ya configurada con la que poder gestionar la red neuronal ya construida.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_custom_objects():
        """
        Returns
        -------
        Network
            Devuelve un diccionario con los objetos personalizados utilizados al construir la red.
        """
        pass

    @staticmethod
    def show_model(model):
        """ Extrae el summary de un modelo como cadena de texto.

        Parameters
        ----------
        model : Model
            modelo al que se quiere extraer el summary

        Returns
        -------
        str
            cadena de texto con el sumarry del modelo.
        """
        summary = ""

        def concat(msg):
            nonlocal summary
            summary = summary + msg + "\n"

        model.summary(print_fn=concat)
        return summary

    @staticmethod
    def _estimate_flops(graph, run_meta):
        flops = tf.compat.v1.profiler.profile(graph,run_meta=run_meta, cmd= 'op',
                                              options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
        return flops.total_float_ops

