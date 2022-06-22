from abc import ABC, abstractmethod

from data_preprocessor import DataPreprocessor
from interfaces.network import NeuralNetwork
from util.loggingutil import LoggingUtil as lu

from globalprops import GlobalProps


class Task(ABC):
    """ Clase encarga de ejecutar tareas. Una tarea puede ser cualquier tipo de prueba o entrenamiento de una red
    neuronal y de la que se loggea diferentes tipos de información."""

    def __init__(self):
        self.logger = None

    def execute(self, params):
        """ Ejecuta la tarea que corresponda según la clase que extienda a esta. De forma común en todos los
        casos se configuran logs y se cargan y procesan los datos de entrada.

        Parameters
        ----------
        params : ParametersContainner
            contenedor con todos los parámetros necesarios para ejecutar la tarea: loader a  utilizar,
            mapeadores de datos, red neuronal a utilizar ...
        """

        GlobalProps.create_instance(params.net_factory_data[0].__name__ if params.pretrainned_network_data is None else
                                    params.pretrainned_network_data[1].__name__, params.test_name, self._create_trainnednn_dir())
        lu.set_file_handler(GlobalProps.get_instance().paths["final_result_path"], "resultados.txt")
        self.logger = lu.get_looger("Task")
        self.logger.info("Inicio del programa")
        self.logger.info("Parámetros utilizados: \n"+params.raw_content)

        data_loader = params.dataset_loader_data[0](**params.dataset_loader_data[1])
        raw_data = data_loader.get_data()

        data_preprocessor = DataPreprocessor(params.mapper)
        train_data, test_data = data_preprocessor.get_processed_data(raw_data, params.batch_size)
        net_factory_data = params.net_factory_data

        if params.pretrainned_network_data is not None:
            neural_network = NeuralNetwork.load_model(params.pretrainned_network_data[0],
                                                      params.pretrainned_network_data[1].get_custom_objects())
            neural_network.test_name = params.test_name
        else:
            neural_network = net_factory_data[0](
                train_data.take(1).as_numpy_iterator().next().shape[1:],
                net_factory_data[1], net_factory_data[2], params.test_name).create_nn()

        self._execute(params, neural_network, (train_data,test_data))

    @abstractmethod
    def _execute(self, params, neural_network, data):
        pass

    @abstractmethod
    def _create_trainnednn_dir(self):
        pass