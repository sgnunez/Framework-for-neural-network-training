import sys
from parametersloader import ParametersLoader


class VAETrainner:

    @staticmethod
    def main():
        """ Método principal de la aplicación se encarga de ejecutar todas las tareas que se le hayan pasado como
        argumento al ejecutar el programa. Los argumentos se deben corresponder a los nombres de
        los ficheros de configuracion"""
        params_loader = ParametersLoader()
        for conf in sys.argv[1:]:
            params = params_loader.get_parameters(conf)
            task = params.task_data[0]
            task.execute(params)


VAETrainner().main()
