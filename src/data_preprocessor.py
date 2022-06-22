

class DataPreprocessor:
    """Clase encargada de preprocesar los datos y crear los datasets."""

    def __init__(self, data_mapper):
        """
        Parameters
        ----------
        data_mapper : DataMapper
            El mapeador que transforma los datos sin procesar en datos procesados.
        """
        self.dataMapper = data_mapper

    def get_processed_data(self, raw_data, batch_size):
        """ Procesa los datos en crudo con el mapeador y devuelve un dataset con el batch indicado.

        Parameters
        ----------
        raw_data : tuple
            una tupla con los datos de entrenamiento y los datos de test, en ese mismo orden.
        batch_size : int
            Tamaño del batch a utilizar en el dataset.

        Returns
        -------
        tuple
            una tupla con los datos de entrenamiento y los datos de test procesados.
        """
        raw_train_data, raw_test_data = raw_data
        if self.dataMapper is not None:
            train_data = raw_train_data.map(self.dataMapper.process_data) if raw_train_data.cardinality().numpy() != 0 else raw_train_data
            test_data = raw_test_data.map(self.dataMapper.process_data) if raw_test_data.cardinality().numpy() != 0 else raw_test_data
        else:
            train_data = raw_train_data
            test_data = raw_test_data

        train_data = train_data.batch(batch_size)
        test_data = test_data.batch(batch_size)

        print(f'number of batches in the training set: {len(train_data)}')
        print(f'number of batches in the test set: {len(test_data)}')
        return train_data, test_data
