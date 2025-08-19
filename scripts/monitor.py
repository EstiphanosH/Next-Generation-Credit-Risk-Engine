"""EvidentlyAI monitoring for data/model drift."""
from evidently.dashboard import Dashboard
from evidently.tabs import DataDriftTab

def check_drift(current, reference):
    """Generates drift report."""
    drift_dashboard = Dashboard(tabs=[DataDriftTab()])
    drift_dashboard.calculate(current, reference)
    return drift_dashboard
