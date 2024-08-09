import os
import json

from utils import nested_dict

from typing import Optional
from collections import defaultdict
from argparse import ArgumentParser
from collections import Counter  

def evaluate(
        final_results: str,
        eval_folder_path: str,
        ground_truth_path: str,
        postfix: Optional[str] = "",
        type_: Optional[str] = "normal"
    ):

    # if type_=="oracle":
    #     comparison = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # else:
    comparison = defaultdict(lambda: defaultdict(list))
    
    # Predictions
    for final_result in final_results:

        with open(final_result, "r") as f:
            data = json.load(f)
        for element in data:

            if type_ in ["inference", "majority"]:
                question = element["instruction"].split("\n\n### Input:\nQuestion: ")[1].split("\n\nPossible first steps:")[0].split("\n<step ")[0]
            elif type_ in ["sample", "oracle", "normal"]:
                question = element["instruction"].split("\n\n### Input:\nQuestion: ")[1].split("\n\n### Response:")[0].split("\n<step ")[0].replace("Question: ", "")
            else:
                raise ValueError("You must pass a `type_`!")
            
            if "result" in element:
                prediction = element["result"]
            elif element["prediction"].startswith("<final answer>: "):
                prediction = element["prediction"].replace("<final answer>: ", "")
            else:
                split_by_answer = element["prediction"].split("<final answer>: ")
                if len(split_by_answer) > 1:
                    prediction = split_by_answer[-1]
                else:
                    continue

            if type_ in ["oracle", "majority"]:
                comparison[question]["prediction"].append(prediction)
            else:   
                comparison[question]["prediction"] = prediction
                
    # Ground truth
    with open(ground_truth_path, "r") as f:
        ground_truth = json.load(f)
    for element in ground_truth:
        question = element["source_question"]
        comparison[question]["ground_truth"] = element["target_result"] #.replace(",", "")
    
    # Write the comparison to a file
    with open(os.path.join(eval_folder_path, f"comparison{postfix}.json"), "w") as f:
        json.dump(comparison, f, indent=4, ensure_ascii=False)
    
    # Now calculate the accuracy
    correct = 0
    predicted = 0
    total = len(ground_truth)
    for question, data in comparison.items():
        if "prediction" not in data or "ground_truth" not in data:
            continue
        predicted += 1
        if type_=="oracle" or type_=="majority":
            majority_vote = Counter(data["prediction"]).most_common(1)[0][0]  
            print("Question:")
            print(question)
            print("--------")
            print(*data["prediction"], sep="\n")
            print("--------")
            print(f"The correct one was {data['ground_truth']}")
            print(f"which was {'not' if not majority_vote == data['ground_truth'] else ''} met")
            
            if any([x == data['ground_truth'] for x in data['prediction']]) and not majority_vote == data['ground_truth']:
                print(f"But the solution was in there!")

            # if any([x == data["ground_truth"] for x in data["prediction"]]):
            if majority_vote == data["ground_truth"]:
                correct += 1
        else:
            if data["prediction"] == data["ground_truth"]:
                correct += 1
    
    accuracy = correct / total
    
    # Write the accuracy to a file
    with open(os.path.join(eval_folder_path, f"accuracy{postfix}.json"), "w") as f:
        json.dump(
            {
                "accuracy": accuracy,
                "correct": correct,
                "false": total - correct,
                "predicted": predicted,
                "missed": total - predicted,
                "n": total,
            }
            , f, indent=4, ensure_ascii=False)
    
    print(f"Accuracy: {round(accuracy * 100, 4)}%")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--eval_folder_path", type=str, required=True)
    parser.add_argument("--ground_truth_path", type=str, required=True)
    parser.add_argument("--final_result", type=str, required=True, nargs='+')
    parser.add_argument("--postfix", type=str, default="")
    parser.add_argument("--type", type=str, default="training", choices=["training", "inference", "oracle", "sample", "majority", "normal"])
    args = parser.parse_args()
    print(args.final_result)
    evaluate(args.final_result, args.eval_folder_path, args.ground_truth_path, args.postfix, args.type)
