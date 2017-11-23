# NeuralNetworksBasic
Fundamentos y ejemplos para iniciarse con redes de neuronas. Este código se usa como código de ejemplo en el taller [El cerebro de las IAs: Redes neuronales](https://cursos.gul.es/xxxi-jornadas-tecnicas-del-gul/el-cerebro-de-las-ias-redes-neuronales) impartido en las [XXXI Jornadas Técnicas del GUL](https://cursos.gul.es/xxxi-jornadas-tecnicas-del-gul).

## Uso

Para poder ejecutar los ejemplos se recomienda utilizar un virtualenv:

```
virtualenv -p python3 <venv-path>
```

Una vez creado el virtualenv, basta con activar el virtualenv e instalar los paquetes especificados en el fichero `requirements.txt`.

```
source <venv-path>/bin/activate
pip install -r requirements.txt
```

## Estructura

La jerarquía que se ha seguido ha sido la siguiente:

* `multilayer-perceptron/` contiene los ejemplos correspodientes al [Multilayer Perceptron](http://cs231n.github.io/neural-networks-1/#nn) sobre el conjunto [MNIST](http://yann.lecun.com/exdb/mnist/), siendo el ejecutable el fichero `run_mlp.py`
* `deep-learning/` contiene ejemplos correspondientes a la sección de Deep Learning. Concretamente representan una [Convolutional Neural Network](http://cs231n.github.io/convolutional-networks/) sobre el conjunto [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)