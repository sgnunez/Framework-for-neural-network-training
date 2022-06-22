import logging
import sys


class LoggingUtil:
    """Clase de utilidad con la que se configuran los logs de la aplicación."""

    std_handler = logging.StreamHandler(stream=sys.stdout)
    std_handler.setLevel(logging.DEBUG)

    file_handler = None

    @staticmethod
    def get_looger(name, disable_file=False):
        """ Permite a cualquier clase de la aplicación obtener un logger en el que poder escribir logs.

        Parameters
        ----------
        name : str
            nombre de la clase que solicita el logger

        disable_file : bool
            parámetro con el que se deshabilita la escritura en fichero de los logs

        Returns
        ---------
        Logger
            logger en el que poder registrar los logs de la aplicación.

        """
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        for handler in list(logger.handlers):
            logger.removeHandler(handler)
        logger.addHandler(LoggingUtil.std_handler)
        if not disable_file:
            logger.addHandler(LoggingUtil.file_handler)

        return logger

    @staticmethod
    def set_file_handler(path, filename):
        """ Permite configurar el fichero en el que se escribirán los logs

        Parameters
        ----------
        path : str
            ruta del fichero

        filename : bool
            nombre del fichero
        """
        LoggingUtil.file_handler = logging.FileHandler(path + "\\" + filename, mode='a', encoding='UTF-8')
        LoggingUtil.file_handler.setLevel(logging.DEBUG)
        LoggingUtil.file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
