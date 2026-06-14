from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelPipeline
from cnnClassifier.pipeline.stage_03_model_trainer import ModelTrainerPipeline
from cnnClassifier.pipeline.stage_04_model_evaluation import ModelEvaluationPipeline


STAGES = [
    ("Data Ingestion", DataIngestionPipeline),
    ("Prepare Base Model", PrepareBaseModelPipeline),
    ("Model Training", ModelTrainerPipeline),
    ("Model Evaluation", ModelEvaluationPipeline),
]

if __name__ == "__main__":
    for stage_name, PipelineClass in STAGES:
        try:
            logger.info(f">>>>>> stage {stage_name} started <<<<<<")
            obj = PipelineClass()
            obj.main()
            logger.info(f">>>>>> stage {stage_name} completed <<<<<<\n\nx==========x")
        except Exception as e:
            logger.exception(e)
            raise e
