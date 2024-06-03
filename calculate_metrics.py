import os
import json
from collections import defaultdict
from typing import Optional, Any

def nested_dict(n: int = 2, type_: Any = list, obj: Optional[dict] = None):
    
    """
    Creates nested dicts (`defaultdict`s of any depth with any type).
    Additionally, can take an arbitrarily deep nested dictionary and
    turn it into its `defaultdict` correspondence.

    :param n: Number of depth levels.
    :param type_: Type of the nested dict
    :param obj: The nested dict to turn into a `defaultdict`.
    :return:
    """
    if n <= 0:  # Base case
        return obj if obj else type_()

    if obj:  # Recursive case
        return defaultdict(lambda: type_(), {k: nested_dict(n=n - 1, type_=type_, obj=v) for k, v in obj.items()})

    return defaultdict(lambda: nested_dict(n=n - 1, type_=type_))

def calc_metrics(
    ground_truth_path = "/cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/data/test.json",
    output_dir ="/cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it"
    ):
    results = nested_dict(n=2, type_=dict)
    with open(ground_truth_path, "r") as f:
        ground_truth = json.load(f)
    for element in ground_truth:
        question = element["source_question"]
        results[question]["ground_truth"] = element["target_result"]

    # Get prediction
    predicted_result_path = os.path.join(output_dir, "final_results.json")
    with open(predicted_result_path, "r") as f:
        predicted_result = json.load(f)

    for element in predicted_result:
        question = element["instruction"].split("\n\n### Input:\n")[1].split("\n\n### Response:\n")[0]
        results[question]["predicted"] = element["result"]

    # For those where there is no prediction, add "None" as a prediction to the results dict
    for question in results:
        if "predicted" not in results[question]:
            results[question]["predicted"] = "None"

    # Calculate error
    for question in results:
        results[question]["correct"] = results[question]["predicted"] == results[question]["ground_truth"]

    # Write results to file
    results_path = os.path.join(output_dir, "final_comparison.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    # Write metrics to file
    metrics = {
        "n": len(results),
        "correct": sum([results[question]["correct"] for question in results]),
        "incorrect": len(results) - sum([results[question]["correct"] for question in results]),
        "accuracy": sum([results[question]["correct"] for question in results]) / len(results)
    }

    metrics_path = os.path.join(output_dir, "final_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    print("Done.")