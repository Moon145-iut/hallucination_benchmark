import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from analysis.metrics import analyze_fine_grained_metrics, visualize_results
from hallucination_scorer import HallucinationScorer
from llm.free_llm import generate_text

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"

SCORER = HallucinationScorer()


def _invoke_llm(
    messages: Iterable[Dict[str, Any]],
    *,
    model_name: str,
    temperature: float = 0.0,
    max_new_tokens: int = 64,
    task: str = "default"
) -> str:
    # Truncate long messages to fit model's context window
    truncated_messages = []
    for msg in messages:
        content = msg['content']
        # Use lower word limit for summarization tasks due to longer documents
        word_limit = 250 if task == "summarization" else 450
        if len(content.split()) > word_limit:
            content = ' '.join(content.split()[:word_limit]) + '...'
        truncated_messages.append({**msg, 'content': content})
    
    return generate_text(
        truncated_messages,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        model_name=model_name,
    ).strip()


def _parse_binary_response(response: str) -> Optional[str]:
    tokens = re.findall(r"\b(yes|no)\b", response.lower())
    if not tokens:
        return None
    unique = set(tokens)
    if len(unique) == 1:
        return tokens[-1].capitalize()
    return None


def _round(value: float) -> float:
    if value != value:  # NaN check
        return 0.0
    return round(float(value), 4)


def _compute_binary_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    valid_predictions = [
        r for r in results if r["judgement"] in {"Yes", "No"}
    ]
    failures = total - len(valid_predictions)
    correct_valid = sum(
        1 for r in valid_predictions if r["judgement"] == r["ground_truth"]
    )
    overall_correct = correct_valid

    tp = sum(
        1
        for r in valid_predictions
        if r["ground_truth"] == "Yes" and r["judgement"] == "Yes"
    )
    fp = sum(
        1
        for r in valid_predictions
        if r["ground_truth"] == "No" and r["judgement"] == "Yes"
    )
    tn = sum(
        1
        for r in valid_predictions
        if r["ground_truth"] == "No" and r["judgement"] == "No"
    )
    fn = sum(
        1
        for r in valid_predictions
        if r["ground_truth"] == "Yes" and r["judgement"] == "No"
    )

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "total_samples": total,
        "valid_predictions": len(valid_predictions),
        "failure_count": failures,
        "coverage": _round(len(valid_predictions) / total if total else 0.0),
        "overall_accuracy": _round(overall_correct / total if total else 0.0),
        "accuracy": _round(correct_valid / len(valid_predictions) if valid_predictions else 0.0),
        "precision": _round(precision),
        "recall": _round(recall),
        "f1": _round(f1),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


def _build_report(results: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    analysis_data = analyze_fine_grained_metrics(results)
    binary_metrics = _compute_binary_metrics(results)

    component_means = {
        name: _round(sum(scores) / len(scores)) if scores else 0.0
        for name, scores in analysis_data["component_scores"].items()
    }
    rounded_correlations: Dict[str, Dict[str, float]] = {}
    for key, values in analysis_data["correlations"].items():
        rounded_correlations[key] = {sub: _round(val) for sub, val in values.items()}

    report = {
        "binary_metrics": binary_metrics,
        "fine_grained_metrics": {
            "average_score": _round(analysis_data["basic_stats"]["avg_score"]),
            "score_std": _round(analysis_data["basic_stats"]["std_score"]),
            "average_confidence": _round(analysis_data["basic_stats"]["avg_confidence"]),
            "severity_distribution": analysis_data["basic_stats"]["severity_distribution"],
            "component_means": component_means,
            "correlations": rounded_correlations,
            "accuracy_severity_correlation": _round(
                analysis_data["basic_stats"]["accuracy_severity_correlation"]
            ),
        },
    }
    return report, analysis_data


def _write_jsonl(records: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_report(report: Dict[str, Any], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)


def _print_summary(report: Dict[str, Any]) -> None:
    binary = report["binary_metrics"]
    fine = report["fine_grained_metrics"]
    print("Evaluation Summary")
    print("------------------")
    print(f"Samples processed: {binary['total_samples']}")
    print(f"Prediction coverage: {binary['coverage'] * 100:.2f}%")
    print(
        "Binary metrics - "
        f"overall_acc: {binary['overall_accuracy']:.3f}, "
        f"acc: {binary['accuracy']:.3f}, "
        f"precision: {binary['precision']:.3f}, "
        f"recall: {binary['recall']:.3f}, "
        f"f1: {binary['f1']:.3f}"
    )
    print("Severity distribution:")
    for label, count in fine["severity_distribution"].items():
        print(f"  {label}: {count}")


def evaluation_qa_dataset(
    model_name: str,
    file_path: Path,
    instruction: str,
    output_path: Path,
) -> List[Dict[str, Any]]:
    with file_path.open("r", encoding="utf-8-sig") as handle:
        data = [json.loads(line) for line in handle]

    results: List[Dict[str, Any]] = []

    system_prompt = (
        "You are a hallucination detector. Respond only with 'Yes' if the answer contains "
        "hallucinated or incorrect information given the knowledge and question. "
        "Otherwise respond only with 'No'."
    )

    for sample in data:
        knowledge = sample["knowledge"]
        question = sample["question"]
        right_answer = sample["right_answer"]
        hallucinated_answer = sample["hallucinated_answer"]

        if random.random() > 0.5:
            candidate = hallucinated_answer
            ground_truth = "Yes"
            candidate_type = "hallucinated"
        else:
            candidate = right_answer
            ground_truth = "No"
            candidate_type = "factual"

        user_prompt = (
            f"{instruction}\n\n"
            f"#Knowledge#: {knowledge}\n"
            f"#Question#: {question}\n"
            f"#Answer#: {candidate}\n"
            "#Your Judgement#:"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        detector_response = _invoke_llm(messages, model_name=model_name, max_new_tokens=16)
        judgement = _parse_binary_response(detector_response) or "failed!"

        metrics = SCORER.calculate_hallucination_score(
            knowledge=knowledge,
            true_response=right_answer,
            generated_response=candidate,
        )

        results.append(
            {
                "knowledge": knowledge,
                "question": question,
                "answer": candidate,
                "candidate_answer": candidate,
                "reference_answer": right_answer,
                "candidate_type": candidate_type,
                "ground_truth": ground_truth,
                "judgement": judgement,
                "detector_response": detector_response,
                "hallucination_metrics": metrics,
                "severity_label": metrics["severity"],
            }
        )

    _write_jsonl(results, output_path)
    return results


def evaluation_dialogue_dataset(
    model_name: str,
    file_path: Path,
    instruction: str,
    output_path: Path,
) -> List[Dict[str, Any]]:
    with file_path.open("r", encoding="utf-8-sig") as handle:
        data = [json.loads(line) for line in handle]

    results: List[Dict[str, Any]] = []

    system_prompt = (
        "You are a dialogue quality inspector. Respond only with 'Yes' if the candidate response "
        "contains hallucinated or incorrect information given the knowledge and dialogue history. "
        "Otherwise respond only with 'No'."
    )

    for sample in data:
        knowledge = sample["knowledge"]
        dialog = sample["dialogue_history"]
        right_response = sample["right_response"]
        hallucinated_response = sample["hallucinated_response"]

        if random.random() > 0.5:
            candidate = hallucinated_response
            ground_truth = "Yes"
            candidate_type = "hallucinated"
        else:
            candidate = right_response
            ground_truth = "No"
            candidate_type = "factual"

        user_prompt = (
            f"{instruction}\n\n"
            f"#Knowledge#: {knowledge}\n"
            f"#Dialogue History#: {dialog}\n"
            f"#Response#: {candidate}\n"
            "#Your Judgement#:"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        detector_response = _invoke_llm(messages, model_name=model_name, max_new_tokens=16)
        judgement = _parse_binary_response(detector_response) or "failed!"

        metrics = SCORER.calculate_hallucination_score(
            knowledge=knowledge,
            true_response=right_response,
            generated_response=candidate,
        )

        results.append(
            {
                "knowledge": knowledge,
                "dialogue_history": dialog,
                "response": candidate,
                "candidate_response": candidate,
                "reference_response": right_response,
                "candidate_type": candidate_type,
                "ground_truth": ground_truth,
                "judgement": judgement,
                "detector_response": detector_response,
                "hallucination_metrics": metrics,
                "severity_label": metrics["severity"],
            }
        )

    _write_jsonl(results, output_path)
    return results


def evaluation_summarization_dataset(
    model_name: str,
    file_path: Path,
    instruction: str,
    output_path: Path,
) -> List[Dict[str, Any]]:
    with file_path.open("r", encoding="utf-8-sig") as handle:
        data = [json.loads(line) for line in handle]

    results: List[Dict[str, Any]] = []

    system_prompt = (
        "You evaluate summaries for factual quality. Respond only with 'Yes' if the candidate summary "
        "contains hallucinated or incorrect information given the document. "
        "Otherwise respond only with 'No'."
    )

    for sample in data:
        document = sample["document"]
        right_summary = sample["right_summary"]
        hallucinated_summary = sample["hallucinated_summary"]

        if random.random() > 0.5:
            candidate = hallucinated_summary
            ground_truth = "Yes"
            candidate_type = "hallucinated"
        else:
            candidate = right_summary
            ground_truth = "No"
            candidate_type = "factual"

        user_prompt = (
            f"{instruction}\n\n"
            f"#Document#: {document}\n"
            f"#Summary#: {candidate}\n"
            "#Your Judgement#:"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        detector_response = _invoke_llm(messages, model_name=model_name, max_new_tokens=24, task="summarization")
        judgement = _parse_binary_response(detector_response) or "failed!"

        metrics = SCORER.calculate_hallucination_score(
            knowledge=document,
            true_response=right_summary,
            generated_response=candidate,
        )

        results.append(
            {
                "document": document,
                "summary": candidate,
                "candidate_summary": candidate,
                "reference_summary": right_summary,
                "candidate_type": candidate_type,
                "ground_truth": ground_truth,
                "judgement": judgement,
                "detector_response": detector_response,
                "hallucination_metrics": metrics,
                "severity_label": metrics["severity"],
            }
        )

    _write_jsonl(results, output_path)
    return results


def run_evaluation(
    task: str,
    model_name: str,
    visualize_dir: Optional[Path] = None,
) -> None:
    instruction_file = SCRIPT_DIR / task / f"{task}_evaluation_instruction.txt"
    with instruction_file.open("r", encoding="utf-8") as handle:
        instruction = handle.read().strip()

    data_path = DATA_DIR / f"{task}_data.json"
    results_path = SCRIPT_DIR / task / f"{task}_{model_name.replace('/', '_')}_results.jsonl"
    report_path = results_path.with_name(results_path.stem + "_report.json")

    if task == "qa":
        records = evaluation_qa_dataset(model_name, data_path, instruction, results_path)
    elif task == "dialogue":
        records = evaluation_dialogue_dataset(model_name, data_path, instruction, results_path)
    elif task == "summarization":
        records = evaluation_summarization_dataset(model_name, data_path, instruction, results_path)
    else:
        raise ValueError("The task must be qa, dialogue, or summarization!")

    report, analysis_data = _build_report(records)
    _write_report(report, report_path)

    if visualize_dir:
        visualize_results(analysis_data, visualize_dir)

    _print_summary(report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-grained hallucination evaluation")
    parser.add_argument("--task", default="qa", choices=["qa", "dialogue", "summarization"])
    parser.add_argument(
        "--model",
        default="google/flan-t5-base",
        help="Hugging Face model name to use for detection.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling hallucinated vs factual responses.",
    )
    parser.add_argument(
        "--visualize-dir",
        type=str,
        default=None,
        help="Optional directory to store analysis visualizations.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    visualize_dir = Path(args.visualize_dir) if args.visualize_dir else None
    run_evaluation(args.task, args.model, visualize_dir)


if __name__ == "__main__":
    main()
