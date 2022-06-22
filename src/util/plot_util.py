import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class Display:
    """Clase de utilidad con la que se crean imágenes que se guardan en disco"""

    @staticmethod
    def display_one_row(disp_images, offset, shape=(28, 28, 3)):
        """ Genera una fila de imágenes

        Parameters
        ----------
        disp_images : Dataset
            imagenes a generar

        offset : int
            parámetro indicando la fila en la que se quiere generar las imágenes

        shape : tuple
            tamaño de las imágenes
        """
        for idx, image in enumerate(disp_images):
            plt.subplot(3, 10, offset + idx + 1)
            plt.xticks([])
            plt.yticks([])
            image = np.clip(image, 0, 1)
            image = np.reshape(image, shape)
            plt.imshow(image, cmap=('gray' if shape[2] == 1 else 'viridis' ))

    @staticmethod
    def display_results(disp_input_images, disp_predicted, image_size, path, name):
        """ Genera las imagenes de entrada y sus predicciones y las guarda en la ruta indicada.

        Parameters
        ----------
        disp_input_images : Dataset
            imagenes de entrada

        disp_predicted : Dataset
            predicciones de las imágenes de entrada

        image_size : tuple
            tamaño de las imágenes

        path : str
            ubicación en la que guardar las imágenes

        name : str
            nombre del fichero con las imágenes.
        """
        plt.figure(figsize=(15, 5))
        Display.display_one_row(disp_input_images, 0, shape=image_size)
        Display.display_one_row(disp_predicted, 20, shape=image_size)
        plt.savefig(path + "\\" + name)
        plt.clf()
        plt.close()

    @staticmethod
    def generate_and_save_images(model, test_input, path, name):
        """ Genera 16 imágenes

        Parameters
        ----------
        model : Dataset
            Modelo con el que predecir las imágenes

        test_input : Dataset
            imágenes a predecir

        path : str
            ubicación en la que guardar las imágenes

        name : str
            nombre del fichero con las imágenes.
        """

        predictions = model.predict(test_input)

        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            img = np.clip(predictions[i, :, :, :], 0, 1)
            img = img * 255
            img = img.astype('int32')
            plt.imshow(img, cmap=('gray' if predictions.shape[2] == 1 else 'viridis' ))
            plt.axis('off')

        # tight_layout minimizes the overlap between 2 sub-plots
        fig.suptitle(name)
        plt.savefig(path + "\\" + name)
        plt.clf()
        plt.close()

    @staticmethod
    def generateHistoricLossGraphic(history,path, name):
        """ Genera un gráfico de la evolución del loss a lo largo de las épocas

        Parameters
        ----------
        history : dict
            diccionario con todos los datos a representar en el gráfico.

        path : str
            ubicación en la que guardar el gráfico.

        name : str
            nombre del fichero del gráfico.
        """
        fig, axs = plt.subplots(figsize=(15, 15))
        x_axis= list(range(0,len(history['Train Mean loss'])))
        axs.plot(x_axis, history['Train Mean loss'])
        axs.plot(x_axis, history['Valid Mean loss'])
        axs.title.set_text('Training Loss vs Validation Loss')
        axs.set_xlabel('Epochs')
        axs.set_ylabel('Loss')
        axs.legend(['Train', 'Test'])

        plt.savefig(path + "\\" + name)
        plt.clf()
        plt.close()

    @staticmethod
    def save_image(path, prediction):
        """ Guarda una imagen en la ruta indicada.

        Parameters
        ----------
        path : str
            ubicación en la que guardar el gráfico.

        prediction : Tensor
            tensor con la imágen que se desea guardar.
        """
        image = np.clip(prediction, 0, 1)
        im = Image.fromarray((image * 255).astype(np.uint8))
        im.save(path)


