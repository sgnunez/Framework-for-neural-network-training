import xml.etree.ElementTree as ET

from tasks.croptask import CropTask
from tasks.predicttask import PredictTask
from tasks.traintask import TrainTask
from util.classloader import ClassLoader
from re import sub


class ParametersLoader:
    """ Clase encargada de procesar los ficheros de configuración y de construir un objecto de tipo
    ParametersContainner que contenga toda la información extraída"""

    task_dict = {
        "train": TrainTask(),
        "test": PredictTask(),
        "crop": CropTask()
    }

    def __init__(self):
        pass

    def get_parameters(self, filename):
        """ Lee el fichero de configuración indicado y extrae toda la información.

        Parameters
        ----------
        filename : str
           nombre del fichero de configuración

        Returns
        -------
        ParametersContainner
            ParametersContainner conteniendo todos los datos del fichero de configuración.
        """
        root = ET.parse(filename).getroot()

        testname = root.find("test_name").text

        task_elem = root.find("task")
        task = ParametersLoader.task_dict[task_elem.get("name")]
        task_params = {child.tag: child.text for child in task_elem.getchildren()}
        task_data = (task, task_params)

        pretrainned_network_data = None
        try:
            pretrainned_network_elem = root.find("load_network")
            pretrainned_network = pretrainned_network_elem.text
            pretrainned_network_str = pretrainned_network_elem.get("name")
            pretrainned_network_class = ClassLoader.load("networks.factories." + pretrainned_network_str,
                                                sub(r"(_|-)+", " ", pretrainned_network_str).title().replace(" ", ""))
            pretrainned_network_data = (pretrainned_network, pretrainned_network_class)
        except:
            pass

        batch_size = int(root.find("batch_size").text)
        shuffle_size = int(root.find("shuffle_size").text)

        dataset_loader_elem = root.find("dataset_loader").getchildren()[0]
        dataset_loader_str = dataset_loader_elem.tag
        dataset_loader_params = dataset_loader_elem.attrib
        dataset_loader = ClassLoader.load("loaders." + dataset_loader_str,
                                          sub(r"([_\-])+", " ", dataset_loader_str).title().replace(" ", ""))
        dataset_loader_data = (dataset_loader, dataset_loader_params)

        mapper_elems = root.find("mappers").getchildren()
        mapper = None
        for m in reversed(mapper_elems):
            mapper_name = m.tag
            mapper_params = list(m.attrib.values())
            mapper = ClassLoader.load("interfaces.mapper", mapper_name)(*mapper_params, data_mapper=mapper)

        net_factory_data = None
        try:
            net_factory_elem = root.find("network_factory")
            net_factory_str = net_factory_elem.get("name")

            optimizer_element = net_factory_elem.find("optimizer").getchildren()[0]
            loss_element = net_factory_elem.find("loss").getchildren()[0]

            optimizer_tag = optimizer_element.tag
            loss_tag = loss_element.tag

            optimizer_params = {key: float(value) for key, value in optimizer_element.attrib.items()}
            loss_params = loss_element.attrib

            optimizer = ClassLoader.load("tensorflow.python.keras.optimizer_v2." + optimizer_tag.lower(),
                                         optimizer_tag)(**optimizer_params)

            loss = ClassLoader.load("tensorflow.python.keras.losses", loss_tag)(*loss_params)

            net_factory = ClassLoader.load("networks.factories." + net_factory_str,
                                           sub(r"(_|-)+", " ", net_factory_str).title().replace(" ", ""))

            net_factory_data = (net_factory, optimizer, loss)
        except:
            pass

        raw_content = ET.tostring(root, encoding='unicode', method='xml')

        return ParametersLoader.ParametersContainner(testname, task_data, pretrainned_network_data, batch_size, shuffle_size,
                                                     dataset_loader_data, mapper, net_factory_data, raw_content)

    class ParametersContainner():
        """Clase encargada de almacenar información."""

        def __init__(self, testname, task_data, pretrainned_network_data, batch_size, shuffle_size, dataset_loader_data,
                     mapper, net_factory_data, raw_content):
            """
            Parameters
             ----------
             testname : str
               Nombre del test que se está realizando

             task_data : tuple
               tupla conteniendo la tarea a ejecutar y sus parámetros

             pretrainned_network_data : tuple
                tupla conteniendo la NetFactory ya entrenada en caso de haberla y sus parametros

            batch_size : int
               tamaño del paso a utilizar en el dataset

            shuffle_size : int
               tamaño del shuffle a utilizar

             mapper : Mapper
               mapper ya configurado con toda la cadena de transformaciones según lo indicado en el fichero de
               configuración.

            net_factory_data : tupla
               En caso de que no se utilice una red ya entrenada contiene la NetFactory, el loss y el optimizer

            raw_content : str
               string con el contenido del fichero de configuración. Se utiliza para mostrarlo en el log.

            """


            self.test_name = testname
            self.task_data = task_data
            self.pretrainned_network_data = pretrainned_network_data
            self.batch_size = batch_size
            self.shuffle_size = shuffle_size

            self.dataset_loader_data = dataset_loader_data
            self.mapper = mapper
            self.net_factory_data = net_factory_data
            self.raw_content = raw_content
