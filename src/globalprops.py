from datetime import datetime
import os


class GlobalProps:
    """ Clase Singleton en la que se almacen variable globales de la aplicación, en concreto la ruta a las diferentes
    carpetas de la aplicación"""

    instance = None

    def __init__(self, model_name, test_name, create_train_folder):
        """
        Parameters
        ----------
        model_name : str
            Nombre del modelo
        test_name : str
            Nombre del test que se está realizando
        create_train_folder : bool
            bandera para indicar si se debe generar o no una carpeta para almacenar la red neuronal entrenada. No es
            necesario generar esta carpeta si no se va a entrenar la red neuronal.
        """

        self.model_name = model_name
        self.test_name = test_name
        self.time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        sesion_id_path = f"{self.model_name}\\{self.time}-{self.test_name}"
        self.paths = {"final_result_path": f"results\\{sesion_id_path}\\final",
                      "partial_result_path": f"results\\{sesion_id_path}\\parcial",
                      "download_path": "downloads",
                      "dataset_path": "datasets"}

        if create_train_folder:
            self.paths["network_path"] = f"trainnedNN\\{sesion_id_path}"

        for path in self.paths.values():
            os.makedirs(path, exist_ok=True)

    @staticmethod
    def create_instance(model_name, test_name, create_train_folder):
        """ Crea una instancia de esta clase.

        Parameters
        ----------
        model_name : str
            Nombre del modelo
        test_name : str
            Nombre del test que se está realizando
        create_train_folder : bool
            bandera para indicar si se debe generar o no una carpeta para almacenar la red neuronal entrenada. No es
            necesario generar esta carpeta si no se va a entrenar la red neuronal.

        """
        GlobalProps.instance = GlobalProps(model_name, test_name, create_train_folder)

    @staticmethod
    def get_instance():
        """ Permite obtener de manera global la única instancia que existe de esta clase.

        Returns
        -------
        GlobalProps
            la única instancia de esta clase.
        """
        return GlobalProps.instance
