import tensorflow as tf 

class Training:
    def __init__(self, config):
        self.config = config 
    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
    
    
    def train_valid_generator(self):

        # Defines how image pixel values should be modified before being fed to the model
        datagenerator_kwargs = dict(
            rescale = 1./255, # Scales pixel values from 0-255 down to 0-1 for better neural network performance
            validation_split=0.20 # Reserves 20% of the training images for validation to monitor overfitting
        )

        # Defines how the raw images should be loaded from the hard drive, resized, and batched
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1], # Resizes all images to fit VGG16's exact requirement
            batch_size=self.config.params_batch_size, # Number of images processed at once to save RAM
            interpolation="bilinear" # Smooths pixels when resizing images to prevent blocky artifacts
        )

        # The core engine that applies the pixel scaling and validation split rules defined above
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        # Automatically loads the 20% validation subset from folders and feeds them to the model
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False, # Keep order intact for validation so we can reliably calculate accuracy
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            # Creates a new engine for the training data that applies random augmentations to prevent overfitting
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.05,
                height_shift_range=0.05,
                zoom_range=0.1,
                brightness_range=[0.8, 1.2], # ±20% brightness for hospital variations
                fill_mode='constant',        # Fill newly created empty space
                cval=0,                      # Fill with pure black (0)
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        # Automatically loads the 80% training subset from folders and feeds them to the model
        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True, # Randomize order so the model doesn't memorize the sequence of images
            **dataflow_kwargs
        )
    
    
    def train(self):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=3
        )

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            callbacks=[early_stopping, reduce_lr]
        )

        self.model.save(self.config.trained_model_path)
