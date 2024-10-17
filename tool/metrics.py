import numpy as np

def _compute_metrics(eval_pred, metrics):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    report = {}
    for metric in metrics:
        if metric.name == "f1":
            report.update(metric.compute(predictions=predictions, references=labels, average="weighted"))
        else:
            report.update(metric.compute(predictions=predictions, references=labels))
    return report