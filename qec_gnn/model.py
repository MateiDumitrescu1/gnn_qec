"""TensorFlow GCN model for graph-level logical-flip classification."""

from __future__ import annotations

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="qec_gnn")
class GCNLayer(tf.keras.layers.Layer):
    """Dense GCN layer: ``ReLU(A_norm @ X @ W + b)``."""

    def __init__(self, units: int, activation: str | None = "relu", **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation_name = activation
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shapes):
        feature_dim = int(input_shapes[0][-1])
        self.w = self.add_weight(
            name="w",
            shape=(feature_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="b",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )

    def call(self, inputs):
        x, a = inputs
        h = tf.matmul(x, self.w)
        h = tf.matmul(a, h)
        h = h + self.b
        return self.activation(h) if self.activation is not None else h

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "activation": self.activation_name})
        return config


@tf.keras.utils.register_keras_serializable(package="qec_gnn")
class MaskedMeanPool(tf.keras.layers.Layer):
    """Mean-pool node embeddings while ignoring padded nodes."""

    def call(self, inputs):
        h, mask = inputs
        mask = tf.cast(mask, h.dtype)
        mask = tf.expand_dims(mask, axis=-1)
        h = h * mask
        return tf.reduce_sum(h, axis=1) / (tf.reduce_sum(mask, axis=1) + tf.keras.backend.epsilon())


@tf.keras.utils.register_keras_serializable(package="qec_gnn")
class MaskedMaxPool(tf.keras.layers.Layer):
    """Max-pool node embeddings while ignoring padded nodes."""

    def call(self, inputs):
        h, mask = inputs
        mask = tf.cast(mask, h.dtype)
        mask = tf.expand_dims(mask, axis=-1)
        very_negative = tf.cast(-1e9, h.dtype)
        masked = tf.where(mask > 0, h, very_negative)
        return tf.reduce_max(masked, axis=1)


@tf.keras.utils.register_keras_serializable(package="qec_gnn")
class MaskedFlatten(tf.keras.layers.Layer):
    """Flatten masked node embeddings to make tiny-overfit memorization possible."""

    def call(self, inputs):
        h, mask = inputs
        mask = tf.cast(mask, h.dtype)
        h = h * tf.expand_dims(mask, axis=-1)
        return tf.reshape(h, (tf.shape(h)[0], -1))

    def compute_output_shape(self, input_shapes):
        h_shape = input_shapes[0]
        return (h_shape[0], h_shape[1] * h_shape[2])


def build_gnn_model(
    *,
    max_nodes: int,
    feature_dim: int,
    hidden_dim: int = 64,
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """Build and compile the minimal dense GCN decoder."""
    x_in = tf.keras.Input(shape=(max_nodes, feature_dim), name="X")
    a_in = tf.keras.Input(shape=(max_nodes, max_nodes), name="A")
    mask_in = tf.keras.Input(shape=(max_nodes,), name="mask")

    h = GCNLayer(hidden_dim, name="gcn1")([x_in, a_in])
    h = GCNLayer(hidden_dim, name="gcn2")([h, a_in])
    h = GCNLayer(hidden_dim, name="gcn3")([h, a_in])
    mean_pool = MaskedMeanPool(name="masked_mean_pool")([h, mask_in])
    max_pool = MaskedMaxPool(name="masked_max_pool")([h, mask_in])
    flat_pool = MaskedFlatten(name="masked_flatten")([h, mask_in])
    graph_embedding = tf.keras.layers.Concatenate(name="graph_summary")([mean_pool, max_pool, flat_pool])
    graph_embedding = tf.keras.layers.Dropout(dropout, name="dropout")(graph_embedding)
    graph_embedding = tf.keras.layers.Dense(hidden_dim, activation="relu", name="dense1")(graph_embedding)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="logical_flip")(graph_embedding)

    model = tf.keras.Model(inputs=[x_in, a_in, mask_in], outputs=output, name="gnn_decoder")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_gnn_model(path: str):
    """Load a saved model with registered custom GCN layers available."""
    return tf.keras.models.load_model(path)
