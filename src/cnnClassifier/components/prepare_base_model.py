import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam


class PrepareBaseModel:
    def __init__(self, prepare_base_model_config):
        self.config = prepare_base_model_config
    
    def get_model(self):
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        self.model.save(self.config.base_model_path)
    
    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            model.trainable = False
        elif (freeze_till is not None) and freeze_till > 0:
            for layer in model.layers[:freeze_till]:
                layer.trainable = False
        
        # Building the model using the Sequential API
        full_model = Sequential()
        full_model.add(model)
        full_model.add(Flatten())
        full_model.add(Dense(256, activation="relu"))
        full_model.add(BatchNormalization())
        full_model.add(Dropout(0.5))
        full_model.add(Dense(units=classes, activation="softmax"))

        full_model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model
    
    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=False,       # Don't freeze everything — we want to fine-tune
            freeze_till=15,         # Freeze layers 0-14 (blocks 1-4), unfreeze block5 (layers 15-18)
            learning_rate=self.config.params_learning_rate
        )
        self.full_model.save(self.config.updated_base_model_path)