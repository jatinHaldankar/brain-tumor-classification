from pathlib import Path
import tensorflow as tf 
import mlflow
import mlflow.keras
import mlflow.tensorflow
from urllib.parse import urlparse
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import read_yaml, save_json

class Evaluation:
    def __init__(self, config : EvaluationConfig):
        self.config = config 
    
    @staticmethod
    def load_model(path: Path):
        return tf.keras.models.load_model(path)
    
    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)
    
    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            
            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
            else:
                mlflow.keras.log_model(self.model, "model")

    def _valid_generator(self):

        # Defines how image pixel values should be modified before evaluation
        datagenerator_kwargs = dict(
            rescale = 1./255 # Scales pixel values from 0-255 down to 0-1 to match how the model was trained
            # No validation_split needed here since we use the dedicated Testing dataset
        )

        # Defines how the raw testing images should be loaded, resized, and batched
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1], # Resizes all testing images to match the model's required input size
            batch_size=self.config.params_batch_size, # Number of images evaluated at once
            interpolation="bilinear" # Smooths pixels when resizing images to prevent blocky artifacts
        )

        # The core engine that applies the pixel scaling rule defined above
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        # Automatically loads the unseen testing images from folders and feeds them to the model for evaluation
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.testing_data,
            shuffle=False, # Must be False for evaluation to match predictions with true labels accurately
            **dataflow_kwargs
        )

    
        