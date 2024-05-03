from tensorflow.keras import layers 
from tensorflow.keras.models import load_model

class ResidualUnit(layers.Layer):
    def __init__(self, filters, strides=1, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.activation = activation
        self.main_layers = [
            layers.Conv2D(filters, 3, strides=strides, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.Activation(activation),
            layers.Conv2D(filters, 3, strides=1, padding="same", use_bias=False),
            layers.BatchNormalization()
        ]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                layers.Conv2D(filters, 1, strides=strides, padding="same", use_bias=False),
                layers.BatchNormalization()
            ]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return layers.Activation(self.activation)(Z + skip_Z)
    

model = load_model('ASL_model.h5')