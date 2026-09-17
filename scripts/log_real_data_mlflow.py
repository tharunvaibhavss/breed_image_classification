"""Log Real-Data Validation Audit Run into MLflow."""

import datetime
from ml.common.mlflow_tracker import MLflowTracker

def log_real_data_run():
    tracker = MLflowTracker()
    timestamp = datetime.datetime.now(datetime.UTC).isoformat()

    run_id = tracker.log_efficientnet_experiment(
        dataset_version="dataset_v001",
        hyperparameters={
            "dataset_path": "data/raw/dataset",
            "physical_images_found": 0,
            "evaluation_type": "real_data_validation_audit",
            "data_leakage_status": "NOT_VERIFIED_NO_PHYSICAL_IMAGES",
        },
        evaluation_metrics={
            "physical_images_count": 0.0,
            "test_images_count": 0.0,
        },
        model_version="1.0.0",
        run_name="real-data-validation-audit-run",
    )
    print(f"MLflow real-data validation audit run logged successfully. Run ID: {run_id}")

if __name__ == "__main__":
    log_real_data_run()
