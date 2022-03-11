import tensorflow as tf

class NN(tf.keras.Model):
    def __init__(
        self,
        input_dim,
        architecture      = None,
        dropout_layers    = None,
        dropout_rates     = None,
        batch_norm_layers = None,
        initializer       = "glorot_uniform",
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
        dropout_layers    [list]   -> list of dropout layers                   e.g. [3, 4]
        dropout_rates     [list]   -> list of dropout rates                    e.g. [0.2, 0.5]
        batch_norm_layers [list]   -> list of batch normalization layers       e.g. [1, 2]
        initializers      [list]   -> list of weights initializers per layer   e.g. ["zeros", "ones", "glorot_uniform"]
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

        # store weights initializers
        self.w_init = initializer

        # store dropout architecture
        self.dropout_arc   = dropout_layers
        self.dropout_rates = dropout_rates

        if self.dropout_arc is not None:
            self.dropout       = [ 
                tf.keras.layers.Dropout(
                    self.dropout_rates[i], 
                    name=f"dropout_{self.dropout_arc[i]-1}"
                )
                for i in range(len(self.dropout_arc))
            ]

        # store batch normalization architecture
        self.batch_norm_arc = batch_norm_layers

        if self.batch_norm_arc is not None:
            self.batch_norm     = [
                tf.keras.layers.BatchNormalization(
                    name=f"batch_norm_{self.batch_norm_arc[i]-1}"
                )
                for i in range(len(self.batch_norm_arc))
            ]

        # create the input layer with input_dim neurons
        self.input_layer = tf.keras.layers.Input(shape=self.input_dim, name="input_layer")

        self.hidden_layers = [
            tf.keras.layers.Dense(
                architecture[i+1], 
                input_shape        = (architecture[i],), 
                activation         = hidden_activation,
                kernel_initializer = self.w_init,
                name               = f"hidden_{i}"
            )
            for i in range(len(architecture)-2)
        ]

        # create hidden layers following architecture
        #if len(self.w_init) == 1:
        #    self.hidden_layers = [
        #        tf.keras.layers.Dense(
        #            architecture[i+1], 
        #            input_shape        = (architecture[i],), 
        #            activation         = hidden_activation,
        #            kernel_initializer = self.w_init[0],
        #            name               = f"hidden_{i}"
        #        )
        #        for i in range(len(architecture)-2)
        #    ]
        #elif len(self.w_init) > 1:
        #    self.hidden_layers = [
        #        tf.keras.layers.Dense(
        #            architecture[i+1], 
        #            input_shape        = (architecture[i],), 
        #            activation         = hidden_activation,
        #            kernel_initializer = self.w_init[i],
        #            name               = f"hidden_{i}"
        #        )
        #        for i in range(len(architecture)-2)
        #    ]

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
        """the call method deals with model creation"""

        batch_norm_counter = 0
        dropout_counter    = 0

        # for each hidden layer, feed it with the previous one
        for i, hidden_layer in enumerate(self.hidden_layers):

            # build batch normalization layer before hidden layer
            if self.batch_norm_arc is not None and (i+1) in self.batch_norm_arc:
                x = self.batch_norm[batch_norm_counter](x)
                batch_norm_counter = batch_norm_counter +1

            # build the hidden layer
            x = hidden_layer(x)

            # build dropout layer after hidden layer
            if self.dropout_arc is not None and (i+1) in self.dropout_arc:
                x = self.dropout[dropout_counter](x)
                dropout_counter = dropout_counter + 1
            
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
        return model.summary(line_length=100)
    

def create_model(
    input_dim,
    architecture,
    dropout_layers    = None,
    dropout_rates     = None,
    batch_norm_layers = None,
    hidden_activation = "relu",
    output_activation = "sigmoid",
    initializer       = "glorot_uniform",
    loss              = "binary_crossentropy",
    optimizer         = "adam",
    metrics           = ["accuracy"],
    nn_name           = "model",
):

    # build the NN model
    model = NN(
        input_dim         = input_dim,
        architecture      = architecture,
        dropout_layers    = dropout_layers,
        dropout_rates     = dropout_rates,
        batch_norm_layers = batch_norm_layers,
        hidden_activation = hidden_activation,
        output_activation = output_activation,
        initializer       = initializer,
        nn_name           = nn_name,
    )
    # compile the NN model
    model.compile(
        loss      = loss,
        optimizer = optimizer,
        metrics   = metrics,
    )

    return model