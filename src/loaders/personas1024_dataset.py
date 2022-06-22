from interfaces import data_loader as dl
import os
import zipfile
import tensorflow as tf
from PIL import Image
from numpy import asarray
from skimage.transform import resize
import numpy as np
from multiprocessing.pool import ThreadPool as Pool


class Personas1024Dataset(dl.DataLoader):
    """Clase encargada de cargar datos de dataset. En concreto esta clase se encarga de cargar el dataset FFHQ.
    Está clase fue creada especificamente para descomprimir y realizar un pequeño procesado de las images descargadas
    de la página oficial"""

    def __init__(self):
        # Se debe indicar la ruta del dataset
        self.datasetroot_path = ''

    def get_data(self):
        """Devuelve los datos obtenidos del dataset cargado.

        Returns
        -------
        tuple
            una tupla con los datos de entrenamiento y los datos de test, en ese mismo orden.
        """
        if not self._check_data_exists():
            self._unzip_data_and_resize()
        return self._split_data()

    def _check_data_exists(self):
         return os.path.isdir(self.datasetroot_path + "images256x256")

    def _unzip_data_and_resize(self):

        extract_path = self.datasetroot_path
        try:
            os.mkdir(self.datasetroot_path)
        except OSError:
            pass

        # extract the zip file
        # for i in range(1,46):
        #     print(str(i))
        #     zip_path =self.datasetroot_path + "images1024x1024-20220603T151742Z-{:03d}.zip".format(i)
        #
        #     zip_ref = zipfile.ZipFile(zip_path, 'r')
        #     zip_ref.extractall(extract_path)
        #     zip_ref.close()

        try:
            os.mkdir(self.datasetroot_path + "images256x256")
        except:
            pass
        pool_size = 8
        def resize_folder(folder):
            if os.path.isdir(self.datasetroot_path + "images1024x1024\\" +folder):
                for file in os.listdir(self.datasetroot_path + "images1024x1024\\" + folder):
                        img = Image.open(self.datasetroot_path + "images1024x1024\\" + folder + "\\" + file)
                        imgarray = asarray(img)
                        resized_img = resize(imgarray, (256, 256))
                        try:
                            os.mkdir(self.datasetroot_path + "images256x256\\" + folder)
                        except:
                            pass
                        im = Image.fromarray((resized_img * 255).astype(np.uint8))
                        im.save(self.datasetroot_path + "images256x256\\" + folder + "\\" + file)

        pool = Pool(pool_size)

        for folder in os.listdir(self.datasetroot_path + "images1024x1024"):
            pool.apply_async(resize_folder, (folder,))

        pool.close()
        pool.join()

    def _split_data(self):
        images = self._read_images(self.datasetroot_path + "images256x256")

        train_paths = images[:int(len(images)*0.8)]
        test_paths = images[int(len(images)*0.8):]

        # load the training image paths into tensors, create batches and shuffle
        train_dataset = tf.data.Dataset.from_tensor_slices(train_paths)
        test_dataset = tf.data.Dataset.from_tensor_slices(test_paths)
        return train_dataset, test_dataset



    def _read_images(self, path):
        image_paths = [os.path.join(path, dir, fname) for dir in os.listdir(path) if os.path.isdir(path + "\\" + dir) for fname in os.listdir(path + "\\" + dir)]
        image_paths.sort()
        return image_paths