from typing import List, Dict, Tuple
import math

def update_belief(prior: Dict[str, float],
                  concept_scores: Dict[str, float], threshold: float = 0.35,
                  alpha: float = 1, eps=1e-12) -> Dict[str, float]:
    """
    Principled log-linear belief update:
        posterior(c) ∝ prior(c) * exp(alpha * relevance(c))
    """
    posterior = prior.copy()
    for concept, score in concept_scores.items():
        if concept in posterior:                     # naive mapping
            posterior[concept] *= math.exp(alpha * score)
        else:
            posterior[concept] = math.exp(alpha * score)
        if posterior[concept] <= threshold:
            if concept in posterior: posterior.pop(concept)
    for k in posterior:
        posterior[k] = max(posterior[k], eps)
    # renormalise
    z = max(sum(posterior.values()), eps)
    for k in posterior:
        posterior[k] /= z
    return posterior

def update_belief_approx(prior: Dict[str, float],
                  concept_scores: Dict[str, float], threshold: float = 0.65,
                  alpha: float = 1, eps=1e-12) -> Dict[str, float]:
    """
    Simple multiplicative update:
        posterior(obj) ∝ prior(obj) * (1 + alpha * relevance(obj))
    We map concept names directly to objects if they match.
    """
    posterior = prior.copy()
    for concept, score in concept_scores.items():
        if concept in posterior:                     # naive mapping
            posterior[concept] *= (1 + alpha * score)
        else:
            posterior[concept] = (alpha * score)
        if posterior[concept] <= threshold:
            if concept in posterior: posterior.pop(concept)
    for k in posterior:
        posterior[k] = max(posterior[k], eps)
    # renormalise
    z = max(sum(posterior.values()), eps)
    for k in posterior:
        posterior[k] /= z
    return posterior

def kl_divergence(p: dict, q: dict, eps: float = 1e-12) -> float:
    return sum(
        pk * math.log(pk / max(q.get(k, 0.0), eps))
        for k, pk in p.items()
        if pk > eps
    )

def js_divergence(p: dict, q: dict, eps: float = 1e-12) -> float:
    return 0.5*sum(
        qk * math.log(qk / (0.5* (max(p.get(k, 0.0), eps) + qk)))
        for k, qk in q.items()
        if qk > eps
    ) + 0.5*sum(
        pk * math.log(pk / (0.5* (max(q.get(k, 0.0), eps) + pk)))
        for k, pk in p.items()
        if pk > eps
    )