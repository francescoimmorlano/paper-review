from tensorflow.keras.layers import Activation, LeakyReLU, PReLU, Conv2D, Dense, Conv2DTranspose, Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras import activations

def custom_CNN_transpose(n_filters, weight_initializer, regularizer, hidden_layers_activation_function, output_layer_activation_function, alpha_constant):
    '''
    Build a Deep Neural Network architecture.

    Parameters
    ----------
    n_filters : int
        Number that is assigned to 'filters' in each Conv2DTranspose
        layer and multiplied in the number of neurons in the Dense layer
    weight_initializer : string
    regularizer : string
    hidden_layers_activation_function : string
        String indicating the activation function to use in the hidden layers

        Possible values:
        * 'linear'
        * 'relu'
        * 'prelu'
        * 'leaky_relu'
        * 'sigmoid'
    output_layer_activation_function : string
    String indicating the activation function to use in the hidden layers

        Possible values:
        * 'linear'
        * 'relu'
        * 'prelu'
        * 'leaky_relu'
        * 'sigmoid'
    alpha_constant : float
        Alpha constant for Leaky ReLU

    Returns
    -------
    model : tf.keras.Model 

    '''
    input_dim = 1
    start_rows = 4
    start_cols = 8

    k_size = 10

    model = Sequential()
    model.add(Dense(n_filters*start_rows*start_cols, input_dim=input_dim, kernel_initializer=weight_initializer))
    
    if hidden_layers_activation_function == 'linear':
        model.add(Activation(activations.linear, name='linear_1'))
    elif hidden_layers_activation_function == 'relu':
        model.add(Activation(activations.relu, name='relu_1'))
    elif hidden_layers_activation_function == 'prelu':
        model.add(PReLU(name='prelu_1'))
    elif hidden_layers_activation_function == 'leaky_relu':
        model.add(LeakyReLU(alpha_constant, name='leaky_relu_1'))
    elif hidden_layers_activation_function == 'sigmoid':
        model.add(Activation(activations.sigmoid, name='sigmoid_1'))

    model.add(Reshape((start_rows,start_cols,n_filters)))

    for i in range(4):
        model.add(Conv2DTranspose(filters=n_filters, kernel_regularizer=regularizer, kernel_size=(k_size,k_size), strides=(2,2), padding='same', kernel_initializer=weight_initializer))
        
        if hidden_layers_activation_function == 'linear':
            model.add(Activation(activations.linear, name=f'linear_{i+2}'))
        elif hidden_layers_activation_function == 'relu':
            model.add(Activation(activations.relu, name=f'relu_{i+2}'))
        elif hidden_layers_activation_function == 'prelu':
            model.add(PReLU(name=f'prelu_{i+2}'))
        elif hidden_layers_activation_function == 'leaky_relu':
            model.add(LeakyReLU(alpha_constant, name=f'leaky_relu_{i+2}'))
        elif hidden_layers_activation_function == 'sigmoid':
            model.add(Activation(activations.sigmoid, name=f'sigmoid_{i+2}'))
    
    model.add(Conv2D(filters=1, kernel_size=(5,5), strides=(1,1), kernel_regularizer=regularizer, padding='same', kernel_initializer=weight_initializer))

    if output_layer_activation_function == 'linear':
        model.add(Activation(activations.linear, name='linear_out'))
    elif hidden_layers_activation_function == 'relu':
        model.add(Activation(activations.relu, name='relu_out'))
    elif output_layer_activation_function == 'prelu':
        model.add(PReLU(name='prelu_out'))
    elif output_layer_activation_function == 'leaky_relu':
        model.add(LeakyReLU(alpha_constant, name='leaky_relu_out'))
    elif output_layer_activation_function == 'sigmoid':
        model.add(Activation(activations.sigmoid, name='sigmoid_out'))

    return model