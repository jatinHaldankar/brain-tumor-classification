
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.prepare_base_model import PrepareBaseModel
from cnnClassifier import logger

STAGE_NAME = "prepare base model"

class PrepareBaseModelPipeline:
    def __init__(self):
        pass

    def main(self):
        configuration_manager = ConfigurationManager()
        base_model_config = configuration_manager.get_prepare_base_model_config()
        base_model = PrepareBaseModel(base_model_config)
        base_model.get_model()
        base_model.update_base_model()

if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareBaseModelPipeline()
        obj.main()
        logger.info(f">>>>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e