import os
from utils.enums import Tasks
from utils.utils import write_results


def _save(y_true, y_pred, model, results_folder):
    labels = ["NonGerm", "germ"]
    task_name = model

    write_results(
        y_true=y_true,
        y_pred=y_pred,
        cm_target_names=labels,
        task_name=task_name,
        results_folder=results_folder,
    )


def save_results(test_results, model, results_path, experiment_name):
    """Save experiment results."""
    results_folder = os.path.join(results_path, experiment_name)

    y_true, y_pred = test_results
    _save(y_true, y_pred, model, results_folder)
