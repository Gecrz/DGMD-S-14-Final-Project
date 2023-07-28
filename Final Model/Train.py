# Define the standard CNN model
n_filters = 12  # Base number of convolutional filters

def make_standard_classifier(n_outputs=1, dropout_rate=0.2):
    Conv2D = partial(tf.keras.layers.Conv2D, padding='same', activation='relu') # Adding Conv2D layers to the model 
    BatchNormalization = tf.keras.layers.BatchNormalization # Define Batch Normalization Layer which can help with training stability and convergence.
    Flatten = tf.keras.layers.Flatten # Flatten the input into a 1D vector,
    Dense = partial(tf.keras.layers.Dense, activation='relu')

    model = tf.keras.Sequential([
        Conv2D(filters=1*n_filters, kernel_size=5, strides=2),
        BatchNormalization(),
        Dropout(dropout_rate),  # Dropout layer 
        Conv2D(filters=2*n_filters, kernel_size=5, strides=2),
        Conv2D(filters=4*n_filters, kernel_size=3, strides=2),
        Dropout(dropout_rate),  # Dropout layer 
        Conv2D(filters=8*n_filters, kernel_size=3, strides=2),
        Dropout(dropout_rate),  # Dropout layer 
        Flatten(),
        Dense(512),
        Dropout(dropout_rate),  # Dropout layer 
        Dense(n_outputs, activation='sigmoid')
    ])
    return model


# Create the standard CNN model with dropout (dropout_rate=0.2)
standard_classifier = make_standard_classifier(dropout_rate=0.2)

# Compile the model
standard_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define callbacks
early_stop = EarlyStopping(patience=5)
def lr_scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * 0.1
lr_schedule = LearningRateScheduler(lr_scheduler)

# Train the model
history=standard_classifier.fit(train_generator, validation_data=val_generator, epochs=15, batch_size = 32, callbacks=[early_stop, lr_schedule])
