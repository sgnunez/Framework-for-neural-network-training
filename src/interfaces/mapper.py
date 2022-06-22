from abc import ABC, abstractmethod
import tensorflow as tf


class DataMapper(ABC):
    """Clase abstracta de la que deben heredar todas las clases encargadas procesar datos de un dataset.
    Esta clase genera una cadena de DataMapper, cada uno de ellos encargados de una tarea concreta en el procesado de
    datos."""

    def __init__(self, data_mapper=None):
        """
        Parameters
        ----------
        data_mapper : DataMapper
            El mapeador que transforma los datos sin procesar en datos procesados
        """
        self.dataMapper = data_mapper

    def process_data(self, raw_data):
        """ Procesa los datos en crudo con el mapeador y devuelve un dataset con el batch indicado.

        Parameters
        ----------
        raw_data : tuple
            una tupla con los datos de entrenamiento y los datos de test, en ese mismo orden.
        Returns
        -------
        tuple
            una tupla con los datos de entrenamiento y los datos de test parcialmente procesados.
        """
        if self.dataMapper is not None:
            return self.dataMapper.process_data(self.__process_data__(raw_data))
        else:
            return self.__process_data__(raw_data)

    @abstractmethod
    def __process_data__(self, raw_data):
        pass


class NormaMapper(DataMapper):
    """Clase encargada de procesar los datos de entrada. Se asume que son imágenes y
    estas se normalizan para que tomen valores entre 0 y 1."""

    def __init__(self, data_mapper=None):
        """
        Parameters
        ----------
        data_mapper : DataMapper
            El mapeador que transforma los datos sin procesar en datos procesados
        """
        super().__init__(data_mapper)

    def __process_data__(self, raw_data):
        return tf.cast(raw_data, dtype=tf.float32) / 255.0


class FileReadJpegMapper(DataMapper):
    """Clase encargada de procesar los datos de entrada. Se asume que los datos de entrada son rutas a imagenes JPEG, y
    esta clase transforma las rutas en las imágenes."""

    def __init__(self, data_mapper=None):
        """
        Parameters
        ----------
        data_mapper : DataMapper
            El mapeador que transforma los datos sin procesar en datos procesados
        """
        super().__init__(data_mapper)

    def __process_data__(self, raw_data):
        img_raw = tf.io.read_file(raw_data)
        return tf.image.decode_jpeg(img_raw)


class FileReadPngMapper(DataMapper):
    """Clase encargada de procesar los datos de entrada. Se asume que los datos de entrada son rutas a imagenes JPEG, y
    esta clase transforma las rutas en las imágenes."""

    def __init__(self, data_mapper=None):
        """
        Parameters
        ----------
        data_mapper : DataMapper
            El mapeador que transforma los datos sin procesar en datos procesados
        """
        super().__init__(data_mapper)

    def __process_data__(self, raw_data):
        img_raw = tf.io.read_file(raw_data)
        return tf.image.decode_png(img_raw)


class ResizeMapper(DataMapper):
    """Clase encargada de procesar los datos de entrada. Para ello esta clase reescala los datos de entrada
    para que tengan la dimensión indicada en el constructor."""

    def __init__(self, img_size, data_mapper=None):
        """
        Parameters
        ----------
        img_size : int
            Entero indicando el ancho y el alto al que se debe reescalar la imagen.
        data_mapper : DataMapper
            El mapeador que transforma los datos sin procesar en datos procesados
        """
        super().__init__(data_mapper)
        self.img_size = int(img_size)

    def __process_data__(self, raw_data):
        return tf.reshape(tf.image.resize(raw_data, (self.img_size, self.img_size)),
                          shape=(self.img_size, self.img_size, 3,))


class FlatMapper(DataMapper):
    """Clase encargada de procesar los datos de entrada. La clase reduce los datos entrada a una única dimensión"""
    def __init__(self, data_mapper=None):
        """
        Parameters
        ----------
        data_mapper : DataMapper
            El mapeador que transforma los datos sin procesar en datos procesados
        """
        super().__init__(data_mapper)

    def __process_data__(self, raw_data):
        return tf.reshape(raw_data, shape=(28 * 28,))


class ToGrayImages(DataMapper):
    """Clase encargada de procesar los datos de entrada. Transforma las imágenes a color en imágenes en escala de
    grises"""
    def __init__(self, data_mapper=None):
        """
        Parameters
        ----------
        data_mapper : DataMapper
            El mapeador que transforma los datos sin procesar en datos procesados
        """
        super().__init__(data_mapper)

    def __process_data__(self, raw_data):
        return tf.image.rgb_to_grayscale(raw_data)


class ToRGBImages(DataMapper):
    """Clase encargada de procesar los datos de entrada. Transforma las imágenes en escala de grises a imágenes RGB.
    No añade color a las imágenes tan solo incrementa el número de calases de 1 a 3."""
    def __init__(self, data_mapper=None):
        """
        Parameters
        ----------
        data_mapper : DataMapper
            El mapeador que transforma los datos sin procesar en datos procesados
        """
        super().__init__(data_mapper)

    def __process_data__(self, raw_data):
        return tf.image.grayscale_to_rgb(raw_data) if tf.shape(raw_data)[2] == 1 else raw_data


class CropMapper(DataMapper):
    """Clase encargada de procesar los datos de entrada. Se asume que la entrada son imágenes y las recorta con el
    tamaño indicado en el constructor."""
    def __init__(self, crop_size, data_mapper=None):
        """
        Parameters
        ----------
        crop_size : int
            Entero indicando el tamaño al que se debe recortar la imagen.
        data_mapper : DataMapper
            El mapeador que transforma los datos sin procesar en datos procesados
        """
        super().__init__(data_mapper)
        self.crop_size = int(crop_size)

    def __process_data__(self, raw_data):
        return tf.image.resize_with_crop_or_pad(raw_data, self.crop_size, self.crop_size)
