import tensorflow as tf

class NN(tf.keras.Model):
    def __init__(
        self,
        input_dim,
        architecture      = [1, 10, 1],
        hidden_activation = "sigmoid",
        output_activation = "sigmoid",
        nn_name           = "my neural network",
        **kwargs
    ):
        """
        Neural Network Model

        Arguments:
        input_dim         [int]    -> number of features in data               e.g. 2
        architecture      [list]   -> neural network architecture              e.g. [2, 3, 3, 1]
        hidden_activation [string] -> activation function for hidden layers    e.g. relu
        output_activation [string] -> activation function for the output layer e.g. sigmoid
        name              [string] -> model name 
        """
        # initialize parent class
        super().__init__(**kwargs)

        # store the model name
        self.nn_name = nn_name

        # store the number of features 
        self.input_dim = input_dim

        # create the input layer with input_dim neurons
        self.input_layer = tf.keras.layers.Input(shape=self.input_dim, name="input_layer")

        # create hidden layers following architecture
        self.hidden_layers = [
            tf.keras.layers.Dense(
                architecture[i+1], 
                input_shape = (architecture[i],), 
                activation  = hidden_activation,
                name        = f"hidden_{i}"
            )
            for i in range(len(architecture)-2)
        ]

        # create the output layer
        self.output_layer = tf.keras.layers.Dense(
            architecture[-1], 
            input_shape = (architecture[-2],), 
            activation  = output_activation, 
            name        = "output_layer"
        )

        # build the model 
        self.build(input_shape=(None, self.input_dim))

    def call(self, x):
        """the call method deals with shape computation"""

        # for each hiddel layer, feed it with the previous one
        for i, hidden_layer in enumerate(self.hidden_layers):
            x = hidden_layer(x)

        # feed the output layer with the last hidden layer
        x = self.output_layer(x)

        # returns the computed output layer
        return x

    def summary(self):
        """re-define summary method to fix the output_shape : multiple issue"""

        # create a temporary model with all the computed shapes (thanks to self.call method)
        model = tf.keras.Model(
            inputs  = [self.input_layer], 
            outputs = self.call(self.input_layer),
            name    = self.nn_name
        )

        # return the model summary with computed shapes
        return model.summary()
    