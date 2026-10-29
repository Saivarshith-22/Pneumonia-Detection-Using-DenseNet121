import tensorflow as tf

model = tf.keras.models.load_model("pneumonia_densenet121.keras")

print("\nMODEL LAYERS:")
for layer in model.layers:
    print(layer.name, type(layer).__name__)

    if isinstance(layer, tf.keras.Model):
        print("  INNER LAYERS:")
        for inner in layer.layers:
            if isinstance(inner, tf.keras.layers.Conv2D):
                print("   CONVOLUTION:", inner.name)