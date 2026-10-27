from __future__ import annotations

from typing import Any, Dict, List, Optional


class ExtractionEvaluator:
    """Evaluates predicted extractions against ground truth dictionaries without schema assumptions."""

    def __init__(self, ground_truth: Dict[str, Any]):
        self.ground_truth = ground_truth

    def evaluate(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []
        self._compare(self.ground_truth, prediction, results)

        comparable = [item for item in results if item["ground_truth"] is not None]
        exact_matches = [item for item in comparable if item["match"]]
        missing_fields = [item for item in comparable if item["prediction"] is None]
        value_mismatches = [
            item for item in comparable if item["prediction"] is not None and not item["match"]
        ]

        accuracy = len(exact_matches) / len(comparable) if comparable else 0.0

        return {
            "total_fields": len(results),
            "comparable_fields": len(comparable),
            "correct_fields": len(exact_matches),
            "missing_fields": len(missing_fields),
            "value_mismatches": len(value_mismatches),
            "accuracy": round(accuracy, 4),
            "field_results": results,
        }

    def _compare(
        self,
        ground_truth: Any,
        prediction: Any,
        results: List[Dict[str, Any]],
        path: str = "",
    ) -> None:
        if isinstance(ground_truth, dict):
            for key, expected_value in ground_truth.items():
                current_path = f"{path}.{key}" if path else key
                actual_value = prediction.get(key) if isinstance(prediction, dict) else None

                if isinstance(expected_value, dict):
                    self._compare(expected_value, actual_value, results, current_path)
                else:
                    norm_expected = self._normalize(expected_value)
                    norm_actual = self._normalize(actual_value)
                    match = norm_expected == norm_actual
                    results.append({
                        "field": current_path,
                        "ground_truth": expected_value,
                        "prediction": actual_value,
                        "match": match,
                    })
        else:
            norm_expected = self._normalize(ground_truth)
            norm_actual = self._normalize(prediction)
            match = norm_expected == norm_actual
            results.append({
                "field": path,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "match": match,
            })

    @staticmethod
    def _normalize(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            cleaned = value.strip()
            return cleaned.lower() if cleaned else None
        return str(value).strip().lower()


__all__ = ["ExtractionEvaluator"]
