import importlib


class ClassLoader:
    """Clase de utilidad que permite la carga dinámica de clases."""

    @staticmethod
    def load(modulename, classname):
        """ Permite cargar de forma dinámica una clase a partir del nombre del módulo y de la clase.

        Parameters
        ----------
        modulename : str
            nombre del módulo en el que se encuentra la clase que se quiere cargar.

        classname : str
            nombre de la clase que se quiere cargar.

        Returns
        ---------
        object
            la clase que se ha cargado dinámicamente. No devuelve una instancia de la clase sino la clase en si misma.

        """
        module = importlib.import_module(modulename)
        class_object = getattr(module, classname)
        return class_object
