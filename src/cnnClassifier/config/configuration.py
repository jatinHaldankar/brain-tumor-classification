from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import DataIngestionConfig, PrepareBaseModelConfig, TrainingConfig, EvaluationConfig
import os
from dotenv import load_dotenv

load_dotenv()

class ConfigurationManager:
    def __init__(self,config_filepath = CONFIG_FILE_PATH,params_filepath = PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])
    
    def data_ingestion_config(self)-> DataIngestionConfig:
        data_ingestion_config = self.config.data_ingestion

        create_directories([data_ingestion_config.root_dir])

        return DataIngestionConfig(data_ingestion_config.root_dir, data_ingestion_config.source_url, data_ingestion_config.local_data_file, data_ingestion_config.unzip_dir)
    
    def get_prepare_base_model_config(self):
        prepare_base_model_config = self.config.prepare_base_model

        create_directories([prepare_base_model_config.root_dir])


        return PrepareBaseModelConfig(
            prepare_base_model_config.root_dir,
            prepare_base_model_config.base_model_path, 
            prepare_base_model_config.updated_base_model_path,  
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )
    
        
    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        training_data = self.config.training.training_data

        
        create_directories([Path(training.root_dir)])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE
        )

        return training_config
    
    def get_model_evaluation_config(self):
        return EvaluationConfig(
            path_of_model = self.config.training.trained_model_path,
            testing_data = self.config.testing.testing_data,
            all_params = self.params,
            mlflow_uri = os.getenv("MLFLOW_TRACKING_URI"),
            params_image_size = self.params.IMAGE_SIZE,
            params_batch_size = self.params.BATCH_SIZE
        )