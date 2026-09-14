import os
from textSummarizer.logging import logger
from textSummarizer.entity import DataValidationConfig

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_files_exist(self) -> bool:
        try:
            validation_status = True
            dataset_dir = os.path.join("artifacts", "data_ingestion", "samsum_dataset")
            
            if not os.path.exists(dataset_dir):
                validation_status = False
            else:
                existing_files = set(os.listdir(dataset_dir))
                for required_file in self.config.ALL_REQUIRED_FILES:
                    if required_file not in existing_files:
                        validation_status = False
                        break

            os.makedirs(os.path.dirname(self.config.STATUS_FILE), exist_ok=True)
            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation status: {validation_status}")

            return validation_status
        
        except Exception as e:
            raise e
