"""Test model training workflow."""
from scripts.train import train_model

def test_training_metrics():
    """Verify AUC > 0.8."""
    with mlflow.start_run():
        train_model()
        run = mlflow.get_run(mlflow.active_run().info.run_id)
        assert run.data.metrics['auc'] > 0.8
