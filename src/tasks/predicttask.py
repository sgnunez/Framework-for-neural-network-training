from interfaces.task import Task


class PredictTask(Task):
    """Define la tarea de predecir con red neuronal. Dicha tarea consiste en predecir todas las imágener que contiene el
    dataset de test indicado en la configuración."""

    def _execute(self, params, neural_network, data):
        neural_network.predict(data[1], "resultado_test_dataset",-1, is_final=True)

    def _create_trainnednn_dir(self):
        return False