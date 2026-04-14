"""
Inter-annotator agreement computation.

Computes Cohen's Kappa and other agreement metrics.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class AgreementStats:
    """Inter-annotator agreement statistics."""
    
    cohens_kappa: float
    percent_agreement: float
    total_items: int
    dimension: str
    annotators: Tuple[str, str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cohens_kappa": round(self.cohens_kappa, 4),
            "percent_agreement": round(self.percent_agreement, 4),
            "percent_agreement_pct": round(self.percent_agreement * 100, 2),
            "total_items": self.total_items,
            "dimension": self.dimension,
            "annotators": self.annotators,
            "interpretation": interpret_kappa(self.cohens_kappa),
        }


def interpret_kappa(kappa: float) -> str:
    """Interpret Cohen's Kappa value."""
    if kappa < 0:
        return "poor (less than chance)"
    elif kappa < 0.20:
        return "slight"
    elif kappa < 0.40:
        return "fair"
    elif kappa < 0.60:
        return "moderate"
    elif kappa < 0.80:
        return "substantial"
    else:
        return "almost perfect"


def compute_agreement(
    annotations_a: List[Dict[str, Any]],
    annotations_b: List[Dict[str, Any]],
    dimension: str,
    annotator_a: str = "annotator_a",
    annotator_b: str = "annotator_b",
) -> AgreementStats:
    """
    Compute inter-annotator agreement for a dimension.
    
    Args:
        annotations_a: Annotations from first annotator
        annotations_b: Annotations from second annotator
        dimension: Dimension to compute agreement for
        annotator_a: Name of first annotator
        annotator_b: Name of second annotator
        
    Returns:
        AgreementStats with Cohen's Kappa and other metrics
    """
    scores_a = []
    scores_b = []
    
    for ann_a, ann_b in zip(annotations_a, annotations_b):
        if dimension in ann_a.get("scores", {}) and dimension in ann_b.get("scores", {}):
            scores_a.append(ann_a["scores"][dimension])
            scores_b.append(ann_b["scores"][dimension])
    
    if not scores_a:
        return AgreementStats(
            cohens_kappa=0.0,
            percent_agreement=0.0,
            total_items=0,
            dimension=dimension,
            annotators=(annotator_a, annotator_b),
        )
    
    agreements = sum(1 for a, b in zip(scores_a, scores_b) if a == b)
    percent_agreement = agreements / len(scores_a)
    
    kappa = _compute_cohens_kappa(scores_a, scores_b)
    
    return AgreementStats(
        cohens_kappa=kappa,
        percent_agreement=percent_agreement,
        total_items=len(scores_a),
        dimension=dimension,
        annotators=(annotator_a, annotator_b),
    )


def _compute_cohens_kappa(scores_a: List[int], scores_b: List[int]) -> float:
    """
    Compute Cohen's Kappa coefficient.
    
    Args:
        scores_a: Scores from annotator A
        scores_b: Scores from annotator B
        
    Returns:
        Cohen's Kappa value (-1 to 1)
    """
    n = len(scores_a)
    if n == 0:
        return 0.0
    
    all_categories = sorted(set(scores_a) | set(scores_b))
    k = len(all_categories)
    
    if k < 2:
        return 1.0
    
    cat_to_idx = {cat: i for i, cat in enumerate(all_categories)}
    
    matrix = [[0] * k for _ in range(k)]
    for a, b in zip(scores_a, scores_b):
        matrix[cat_to_idx[a]][cat_to_idx[b]] += 1
    
    po = sum(matrix[i][i] for i in range(k)) / n
    
    row_totals = [sum(matrix[i]) for i in range(k)]
    col_totals = [sum(matrix[i][j] for i in range(k)) for j in range(k)]
    
    pe = sum(row_totals[i] * col_totals[i] for i in range(k)) / (n * n)
    
    if pe == 1.0:
        return 1.0
    
    kappa = (po - pe) / (1 - pe)
    return kappa


def compute_all_agreements(
    annotations_by_annotator: Dict[str, List[Dict[str, Any]]],
    dimensions: List[str],
) -> Dict[Tuple[str, str], Dict[str, AgreementStats]]:
    """
    Compute agreement for all pairs of annotators across all dimensions.
    
    Args:
        annotations_by_annotator: Dict mapping annotator ID to their annotations
        dimensions: List of dimensions to compute agreement for
        
    Returns:
        Dict mapping annotator pairs to dimension agreements
    """
    annotators = list(annotations_by_annotator.keys())
    results = {}
    
    for i, ann_a in enumerate(annotators):
        for ann_b in annotators[i+1:]:
            pair = (ann_a, ann_b)
            results[pair] = {}
            
            for dim in dimensions:
                stats = compute_agreement(
                    annotations_by_annotator[ann_a],
                    annotations_by_annotator[ann_b],
                    dim,
                    ann_a,
                    ann_b,
                )
                results[pair][dim] = stats
    
    return results


def compute_fleiss_kappa(
    annotations: List[List[int]],
    n_categories: int,
) -> float:
    """
    Compute Fleiss' Kappa for multiple annotators.
    
    Args:
        annotations: List of lists, where each inner list contains
                    the category assignments from all annotators for one item
        n_categories: Number of possible categories
        
    Returns:
        Fleiss' Kappa value
    """
    n_items = len(annotations)
    if n_items == 0:
        return 0.0
    
    n_raters = len(annotations[0])
    if n_raters < 2:
        return 0.0
    
    counts = []
    for item_annotations in annotations:
        item_counts = [0] * n_categories
        for ann in item_annotations:
            if 0 <= ann < n_categories:
                item_counts[ann] += 1
        counts.append(item_counts)
    
    p_j = []
    for j in range(n_categories):
        total = sum(counts[i][j] for i in range(n_items))
        p_j.append(total / (n_items * n_raters))
    
    P_bar_e = sum(p ** 2 for p in p_j)
    
    P_i = []
    for i in range(n_items):
        sum_sq = sum(counts[i][j] ** 2 for j in range(n_categories))
        P_i.append((sum_sq - n_raters) / (n_raters * (n_raters - 1)))
    
    P_bar = sum(P_i) / n_items
    
    if P_bar_e == 1.0:
        return 1.0
    
    kappa = (P_bar - P_bar_e) / (1 - P_bar_e)
    return kappa
