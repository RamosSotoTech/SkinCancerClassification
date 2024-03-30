import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from dataset_reader import train_generator, validation_generator, num_classes, class_weights

load_model = False

# Load the VGG19 model
vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

# Freeze the layers
for layer in vgg19.layers:
    layer.trainable = False

# Add a custom output layer
x = Flatten()(vgg19.output)
x = Dense(1024, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
prediction = Dense(num_classes, activation='softmax')(x)


def focal_loss(gamma=2., alpha=.25, from_logits=False):
    """

    :param gamma: Exponent of the modulating factor (1 - p_t)^gamma
    :param alpha: Weight factor for the positive class
    :param from_logits: whether the input is a logit or a probability
    :return: Focal loss function
    """

    def focal_loss_with_logits(logits, targets):
        y_pred = tf.sigmoid(logits)
        loss = targets * (-alpha * tf.pow((1 - y_pred), gamma) * tf.math.log(y_pred)) + \
               (1 - targets) * (-alpha * tf.pow(y_pred, gamma) * tf.math.log(1 - y_pred))
        return tf.reduce_sum(loss)

    def focal_loss_with_probs(probs, targets):
        """
        References :Lee, J.-W., & Kang, H.-S. (2024). Three-Stage Deep Learning Framework for Video Surveillance. Applied Sciences (2076-3417), 14(1), 408. https://doi-org.lopes.idm.oclc.org/10.3390/app14010408

        :param probs: y_pred from the model (predicted probabilities)
        :param targets: y_true from the model (true labels)
        :return: Focal loss
        """
        eps = tf.keras.backend.epsilon()
        loss = (targets * (-alpha * tf.pow((1 - probs), gamma) * tf.math.log(probs + eps)) + (1 - targets) *
                (-alpha * tf.pow(probs, gamma) * tf.math.log(1 - probs + eps)))
        return tf.reduce_sum(loss)

    return focal_loss_with_logits if from_logits else focal_loss_with_probs

# Create a model object or load the model
# check if the model.keras file exists
import os

if os.path.exists('VGG19_tunned.keras') and load_model:
    model = tf.keras.models.load_model('VGG19_tunned.keras')

else:
    # Create a model object
    model = Model(inputs=vgg19.input, outputs=prediction)

    # Compile the model
    model.compile(optimizer=AdamW(learning_rate=0.01), loss=focal_loss(alpha=0.25),
                  metrics=['accuracy', 'AUC', 'Precision', 'Recall'])

# Calculate the initial values of the metrics
metrics_initial_values = model.evaluate(validation_generator,
                                        steps=validation_generator.samples // validation_generator.batch_size)
# calculate the initial values of loss, accuracy, AUC, Precision, and Recall

# Initialize the list for the metrics and loss
metrics_and_loss_values = dict(zip(model.metrics_names, metrics_initial_values))

print('Initial metrics values:', metrics_and_loss_values)

epochs = 100

# Callbacks
callbacks = [EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
             ModelCheckpoint('VGG19_tunned.keras', save_best_only=True, verbose=1, monitor='val_loss',
                             mode='min', initial_value_threshold=metrics_and_loss_values['loss']),
             TensorBoard(log_dir='logs/phase1', histogram_freq=1),
             ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='min')]

history_phase1 = model.fit(train_generator, validation_data=validation_generator, epochs=epochs,
                           steps_per_epoch=train_generator.samples // train_generator.batch_size,
                           validation_steps=validation_generator.samples // validation_generator.batch_size,
                           # class_weight=class_weights,
                           callbacks=callbacks)

print(model.summary())
print(history_phase1.history)

# retrains the model with a smaller learning rate and the last 5 layers of the VGG19 model trainable

# load the model weights with the best validation loss
# EarlyStopping will restore the best weights
# model = tf.keras.models.load_model('VGG19_tunned.keras', custom_objects={'focal_loss_with_logits': focal_loss,
#                                                                         'focal_loss_with_probs': focal_loss})

# Make last 5 layers of VGG19 model trainable
for layer in vgg19.layers[-5:]:
    layer.trainable = True

# Compile the model with a smaller learning rate
model.compile(optimizer=AdamW(learning_rate=0.00001), loss='categorical_crossentropy',
              metrics=['accuracy', 'AUC', 'Precision', 'Recall'])

best_val_loss = min(history_phase1.history['val_loss'])

callbacks = [EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
             ModelCheckpoint('VGG19_tunned.keras', save_best_only=True, verbose=1, monitor='val_loss',
                             mode='min', initial_value_threshold=best_val_loss),
             TensorBoard(log_dir='logs/phase2', histogram_freq=1),
             ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='min')]

# Train the model again
history_phase2 = model.fit(train_generator, validation_data=validation_generator, epochs=epochs,
                           steps_per_epoch=train_generator.samples // train_generator.batch_size,
                           validation_steps=validation_generator.samples // validation_generator.batch_size,
                           # class_weight=class_weights,
                           callbacks=callbacks)

# Print the model summary
print(model.summary())
# history
print(history_phase2.history)

if __name__ == '__main__':
    # Save the model
    model.save('VGG19_tunned.keras')

    # Save the history
    import pickle

    with open('history_phase1.pkl', 'wb') as f:
        pickle.dump(history_phase1.history, f)

    with open('history_phase2.pkl', 'wb') as f:
        pickle.dump(history_phase2.history, f)

    print('Model and history saved as VGG19_tunned.keras, history_phase1.pkl, and history_phase2.pkl')
