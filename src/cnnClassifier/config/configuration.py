from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import DataIngestionConfig, PrepareBaseModelConfig

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