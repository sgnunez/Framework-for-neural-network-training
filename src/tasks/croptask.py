from interfaces.task import Task


class CropTask(Task):
    """Define la tarea de extraer recortes de las imágenes del dataset indicas en la configuración."""

    def _execute(self, params, neural_network, data):
        neural_network.crop(data[1],int(params.task_data[1]["x"]),int(params.task_data[1]["y"]),
                            int(params.task_data[1]["width"]),int(params.task_data[1]["height"]))

    def _create_trainnednn_dir(self):
        return False