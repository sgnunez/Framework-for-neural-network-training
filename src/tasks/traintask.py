from interfaces.task import Task


class TrainTask(Task):
    """Define la tarea de entrenar una red neuronal. Dicha tarea consiste en entrenar la red neuronal
    segun los parámetros indicados en la configuración y realizar un prediccion con los datos de test."""

    def _execute(self, params, neural_network, data):
        neural_network.train(data[0], int(params.task_data[1]["epochs"]), test_dataset=data[1])
        neural_network.predict(data[1], "resultado_test_dataset", 10, is_final=True)

    def _create_trainnednn_dir(self):
        return True