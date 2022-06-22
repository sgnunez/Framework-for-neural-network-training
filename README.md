Descripción
--------------------

A continuación se incluye un árbol de directorios del proyecto con una descripción de cada una de las partes.

├───datasets									Carpeta en la que se deben incluir los datasets (inicialmente vacía, los datasets no se incluyen en la entrega, se deben descargar de forma externa, consultar referencias en la documentación).
├───results									Carpeta en la que se almacenan los resultados obtenidos tras la ejecución de una prueba.
│   ├───FirstVaeNetFactory							Carpeta en la que se almacen los resultados obtenidos en todas las pruebas realizadas con LVAE.
│   │   ├───2022_06_13_18_34_55-tercer_entrenamiento_personas			Resultados del entrenamiento de LVAE con la poderación original.
│   │   │   ├───final								Carpeta con los resultados obtenidos al final del entrenamiento. (Esta carpeta se incluye en todos los demás resultados pero se omite en el resto del arbol).
│   │   │   └───parcial								Carpeta con los resultados obtenidos durante el entrenamiento.	(Esta carpeta se incluye en todos los demás resultados pero se omite en el resto del arbol).
│   │   ├───2022_06_13_22_38_58-tercer_entrenamiento_personas_generacion	Resultados del entrenamiento de LVAE con la poderación propuesta.
│   │   ├───2022_06_14_11_30_50-test_entrenamiento_personas_generacion		Resultado de testear LVAE con la ponderación original utilizando el dataset de entrenamiento (subconjunto de pruebas).
│   │   ├───2022_06_14_11_37_26-test_entrenamiento_personas			Resultado de testear LVAE con la ponderación propuesta utilizando el dataset de entrenamiento (subconjunto de pruebas).
│   │   ├───2022_06_19_16_51_49-Test_CROP_Urban100				Resultado de testear LVAE con la ponderación propuesta utilizando el dataset de test Urban100.
│   │   ├───2022_06_19_16_52_01-Test_CROP_set14					Resultado de testear LVAE con la ponderación propuesta utilizando el dataset de test Set14.
│   │   ├───2022_06_19_16_52_06-Test_CROP_set5					Resultado de testear LVAE con la ponderación propuesta utilizando el dataset de test Set5.
│   │   ├───2022_06_19_16_52_11-Test_CROP_Manga109				Resultado de testear LVAE con la ponderación propuesta utilizando el dataset de test Manga109.
│   │   └───2022_06_19_16_54_25-Test_CROP_BSDS100				Resultado de testear LVAE con la ponderación propuesta utilizando el dataset de test BSDS100.
│   ├───MascotasAeFactory							Carpeta en la que se almacen los resultados obtenidos en todas las pruebas realizadas con LCAE1.
│   │   ├───2022_06_04_00_24_46-Segundo_entrenamiento_con_personas_256x256	Resultados del entrenamiento de LCAE1.
│   │   ├───2022_06_12_18_16_25-CROP_IMAGE_soldado				Resultado de recortar una imagen de un soldado del dataset BSDS100 generada por LCAE1
│   │   ├───2022_06_12_18_53_30-CROP_IMAGE_soldado_metrics			Resultado en el que se obtienen las métricas de calidad de la imagen del soldado anterior.
│   │   ├───2022_06_12_19_31_34-CROP_IMAGE_animals				Resultado de recortar una imagen de un animal del dataset BSDS100 generada por LCAE1.
│   │   ├───2022_06_12_19_33_00-CROP_IMAGE_animals_metrics			Resultado en el que se obtienen las métricas de calidad de la imagen del animal anterior.
│   │   ├───2022_06_13_11_11_17-Test_Crop_Entrenamiento				Resultado de testear LCAE1 utilizando el dataset de entrenamiento (subconjunto de pruebas).
│   │   ├───2022_06_14_12_35_37-CROP_IMAGE_compresion1				Resultado de recortar una imagen del dataset FFHQ generada por LCAE1.
│   │   ├───2022_06_14_12_55_49-CROP_IMAGE_compresion2				Resultado de recortar una imagen del dataset FFHQ generada por LCAE1.
│   │   ├───2022_06_19_16_15_05-Test_CROP_Urban100				Resultado de testear LCAE1 utilizando el dataset de test Urban100.
│   │   ├───2022_06_19_16_15_26-Test_CROP_Set14					Resultado de testear LCAE1 utilizando el dataset de test Set14.
│   │   ├───2022_06_19_16_15_30-Test_CROP_Set5					Resultado de testear LCAE1 utilizando el dataset de test Set5.
│   │   ├───2022_06_19_16_15_33-Test_CROP_Manga109				Resultado de testear LCAE1 utilizando el dataset de test Manga109.
│   │   └───2022_06_19_16_15_44-Test_CROP_BSDS100				Resultado de testear LCAE1 utilizando el dataset de test BSDS100.
│   └───PersonasAeFactory							Carpeta en la que se almacen los resultados obtenidos en todas las pruebas realizadas con LCAE2.
│       ├───2022_06_04_12_48_45-Segundo_entrenamiento_con_personas_256x256	Resultados del entrenamiento de LCAE2.
│       ├───2022_06_12_18_11_37-CROP_IMAGE_soldado_original			Resultado de recortar una imagen de un soldado del dataset BSDS100 sobre la imagen original
│       ├───2022_06_12_18_12_25-CROP_IMAGE_soldado				Resultado de recortar una imagen de un soldado del dataset BSDS100 generada por LCAE2
│       ├───2022_06_12_18_39_25-CROP_IMAGE_soldado_metrics			Resultado en el que se obtienen las métricas de calidad de la imagen del soldado anterior.
│       ├───2022_06_12_19_27_22-CROP_IMAGE_animals_original			Resultado de recortar una imagen de un animal del dataset BSDS100 sobre la imagen original.
│       ├───2022_06_12_19_28_44-CROP_IMAGE_animals				Resultado de recortar una imagen de un animal del dataset BSDS100 generada por LCAE2.
│       ├───2022_06_12_19_29_40-CROP_IMAGE_animals_metrics			Resultado en el que se obtienen las métricas de calidad de la imagen del animal anterior.
│       ├───2022_06_13_11_31_21-Test_Crop_Entrenamiento				Resultado de testear LCAE2 utilizando el dataset de entrenamiento (subconjunto de pruebas).
│       ├───2022_06_14_12_28_32-CROP_IMAGE_compresion1_original			Resultado de recortar una imagen del dataset FFHQ sobre la imagen original
│       ├───2022_06_14_12_30_32-CROP_IMAGE_compresion1				Resultado de recortar una imagen del dataset FFHQ generada por LCAE2.
│       ├───2022_06_14_12_54_37-CROP_IMAGE_compresion2_original			Resultado de recortar una imagen del dataset FFHQ sobre la imagen original
│       ├───2022_06_14_12_58_35-CROP_IMAGE_compresion2				Resultado de recortar una imagen del dataset FFHQ generada por LCAE2.
│       ├───2022_06_19_16_31_42-Test_CROP_Urban100				Resultado de testear LCAE1 utilizando el dataset de test Urban100.
│       ├───2022_06_19_16_31_53-Test_CROP_Set14					Resultado de testear LCAE1 utilizando el dataset de test Set14.
│       ├───2022_06_19_16_31_56-Test_CROP_Set5					Resultado de testear LCAE1 utilizando el dataset de test Set5.
│       ├───2022_06_19_16_31_59-Test_CROP_Manga109				Resultado de testear LCAE1 utilizando el dataset de test Manga109.
│       └───2022_06_19_16_32_08-Test_CROP_BSDS100				Resultado de testear LCAE1 utilizando el dataset de test BSDS100.
├───src										Carpeta en la que se encuentra todo el código fuente.
│   ├───interfaces								Carpeta con los modulos de Python de las clases abstractas
│   ├───loaders									Carpeta con los modulos de Python encargados de cargar datos de datasets
│   ├───networks								Carpeta con los modulos de Python encargados de gestionar el entrenamiento y predicciones de los diferentes tipos de redes neuronales.
│   │   └───factories								Carpeta con los modulos de Python encargados de definir e instanciar los modelos de redes neuronales.
│   ├───tasks									Carpeta con los modulos de Python encargados de definir las posibles tareas a ejecutar (entrenar o testear una red neuronal, recortar una imagen...)
│   └───util									Carpeta con los modulos de Python que contienen utilidades global de la aplicaión.
└───trainnedNN									Carpeta en la que se almacenan las redes neuronales entrenadas.
    ├───FirstVaeNetFactory							Carpeta en la que se almacenan las redes neuronales entrenadas correspondientes a una VAE.
    │   ├───2022_06_13_18_34_55-tercer_entrenamiento_personas			Carpeta en la que se almacenan las redes neuronales de un entrenamiento concreto (LVAE con la ponderación original).
    │   │   └───99								Carpeta en la que se almacenan la red neuronal con los parámetros calculados en la época 99 de un entrenamiento concreto.
    │   └───2022_06_13_22_38_58-tercer_entrenamiento_personas_generacion	Carpeta en la que se almacenan las redes neuronales de un entrenamiento concreto (LVAE con la ponderación propuesta).
    │       └───99								Carpeta en la que se almacenan la red neuronal con los parámetros calculados en la época 99 de un entrenamiento concreto.
    ├───MascotasAeFactory							Carpeta en la que se almacenan las redes neuronales entrenadas correspondientes a un autoencoder.
    │   └───2022_06_04_00_24_46-Segundo_entrenamiento_con_personas_256x256	Carpeta en la que se almacenan las redes neuronales de un entrenamiento concreto (LCAE1).
    │       └───99								Carpeta en la que se almacenan la red neuronal con los parámetros calculados en la época 99 de un entrenamiento concreto.
    └───PersonasAeFactory							Carpeta en la que se almacenan las redes neuronales entrenadas correspondientes a un autoencoder.
        └───2022_06_04_12_48_45-Segundo_entrenamiento_con_personas_256x256	Carpeta en la que se almacenan las redes neuronales de un entrenamiento concreto (LCAE2).
            └───99								Carpeta en la que se almacenan la red neuronal con los parámetros calculados en la época 99 de un entrenamiento concreto.


En la raíz del proyecto tambien se incluyen los ficheros de configuración de todas las pruebas que han dado lugar a los resultados que muestran en el árbol. 
Además también encontramos este fichero y el fichero requirements.txt con las dependencias de Python.

Requisitos
--------------------

Es obligatoria la instalación de python. Este proyecto se ha probado con la versión 3.8.

Además es necesario instalar todas las librerías que contiene el fichero requirements.txt

Para ello se debe ejecutar desde la ruta del proyecto el comando:

> pip install -r requirements.txt

Es importante destacar que en este proyecto no se incluyen los dataset utilizados, estos se deben descargar de forma externa. En la documentación se encuentra 
información más detallada sobre los dataset y cómo obtenerlos.

Por último, se debe tener en cuenta que en la versión de esta documentación subida a moovi no se incluyen las redes neuronales entrenadas por limitaciones de espacio.
En caso de necesitarse estas se pueden descargar desde el siguiente enlance: https://github.com/sgnunez/Framework-for-neural-network-training.git
Al añadir las redes neuronales se deben respetar las rutas originales o de lo contrario será necesario modificar los ficheros de configuración. 


Ejecucción
-------------------

Para iniciar la ejecución del programa se debe ejecutar el archivo Python vae_trainner.py de la carpeta src.
Se debe indicar como parámetros los diferentes ficheros de configuración que se quieren ejecutar. Por ejemplo, para ejecutar
el fichero de configuración conf_vae_bsds100.xml se podría ejecutar el siguiente comando:

> python src\vae_trainner.py conf_vae_bsds100.xml

Para más información detalla sobre como funciona este código ver anexo de la documentación aportada.
