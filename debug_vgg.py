# debug_vgg.py
from vgg_face import get_vgg_face_model
import tensorflow as tf

# Load the VGG-Face model
base_model = get_vgg_face_model(include_top=False, input_shape=(224, 224, 3))

# Print all layer names to identify any 'flatten' layers
print("All layer names in VGG-Face model:")
for i, layer in enumerate(base_model.layers):
    print(f"{i}: {layer.name} - {type(layer).__name__}")

# Try a different approach for creating BMI model
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs)
x = tf.keras.layers.Flatten(name='bmi_flatten')(x)
x = tf.keras.layers.Dense(512, activation='relu', name='bmi_dense1')(x)
x = tf.keras.layers.Dropout(0.5, name='bmi_dropout1')(x)
x = tf.keras.layers.Dense(128, activation='relu', name='bmi_dense2')(x)
x = tf.keras.layers.Dropout(0.5, name='bmi_dropout2')(x)
outputs = tf.keras.layers.Dense(1, activation='linear', name='bmi_output')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Print the model summary
print("\nModel Summary:")
model.summary()

print("\nSuccessfully created model with unique layer names!")