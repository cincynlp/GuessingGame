ig_records = []  # each entry: {'ig': float, 'questions_remaining': int}
# Logging setup
import logging
import os
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------
# Existing Helper Functions
# -------------------------------

def clean_candidate(candidate: str) -> str:
    """
    Cleans the candidate string to extract a valid physical object name by removing extraneous leading words and punctuation.
    This function:
      1. Strips whitespace and surrounding quotes/punctuation.
      2. If a colon is present, keeps only the text after the colon.
      3. Removes extraneous leading words (e.g., 'okay', 'ok', 'since', 'it's likely', 'um', 'uh', 'hmm', 'well').
      4. Removes leading articles such as 'a' or 'an'.
      5. Performs final cleanup of any residual punctuation.
    """
    import re
    candidate = candidate.strip().strip('\"\'“”‘’')
    if ':' in candidate:
        candidate = candidate.split(':', 1)[1].strip()
    # Remove extraneous leading words and punctuation.
    pattern = r'^(okay|ok|since|it\'?s likely|um|uh|hmm|well)[\s,\-.:;]+'
    candidate = re.sub(pattern, '', candidate, flags=re.IGNORECASE)
    # Remove leading articles.
    candidate = re.sub(r'^(a|an)\s+', '', candidate, flags=re.IGNORECASE)
    candidate = candidate.strip().strip('.,;:!?\'\"')
    candidate = candidate.replace(" ", "_")
    return candidate

import random
TOTAL_RELATIONS = [
  "Antonym",
  "AtLocation",
  "CapableOf",
  "Causes",
  "CausesDesire",
  "CreatedBy",
  "DefinedAs",
  "DerivedFrom",
  "Desires",
  "DistinctFrom",
  "Entails",
  "FormOf",
  "HasA",
  "HasContext",
  "HasFirstSubevent",
  "HasLastSubevent",
  "HasPrerequisite",
  "HasProperty",
  "HasSubevent",
  "InstanceOf",
  "IsA",
  "LocatedNear",
  "MadeOf",
  "MannerOf",
  "MotivatedByGoal",
  "NotCapableOf",
  "NotDesires",
  "NotHasProperty",
  "PartOf",
  "ReceivesAction",
  "RelatedTo",
  "SimilarTo",
  "SymbolOf",
  "Synonym",
  "UsedFor",
  "dbpedia/capital",
  "dbpedia/field",
  "dbpedia/genre",
  "dbpedia/genus",
  "dbpedia/influencedBy",
  "dbpedia/knownFor",
  "dbpedia/language",
  "dbpedia/leader",
  "dbpedia/occupation",
  "dbpedia/product"
]

# -------------------------------
# Shared requests session with retry/backoff for ConceptNet API
# -------------------------------
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure a shared session with retry/backoff
session = requests.Session()
retry = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)
session.mount("https://", HTTPAdapter(max_retries=retry))

# -------------------------------
# ConceptNet semantic mapping setup
# -------------------------------
import pandas as pd

try:
    # Only load columns we need for mapping
    edges_df = pd.read_csv('ConceptInfoGain/conceptnet_english_edges.csv', usecols=['relation','end_node'])
except Exception as e:
    logger.error("Failed to load ConceptNet edges CSV: %s", e)
    edges_df = None

try:
    rel_edges_df = pd.read_csv('ConceptInfoGain/conceptnet_english_edges.csv', usecols=['relation','start_node','end_node'])
except Exception as e:
    logger.error("Failed to load ConceptNet relation CSV: %s", e)
    rel_edges_df = None

# Precompute mappings for speed
if rel_edges_df is not None:
    # map relation -> set of start_nodes (domain)
    domain_map = {
        rel: set(group['start_node'])
        for rel, group in rel_edges_df.groupby('relation')
    }
    # map (relation, end_node) -> set of start_nodes (yes-set)
    yes_map = {
        (rel, end_node): set(group['start_node'])
        for (rel, end_node), group in rel_edges_df.groupby(['relation','end_node'])
    }
else:
    domain_map, yes_map = {}, {}

# --- Semantic similarity model setup for mapping responses to ConceptNet URIs ---
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import pickle

# Preload embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")

# Cache embeddings so we compute only once
EMBED_CACHE = 'ConceptInfoGain/conceptnet_label_embeddings.pkl'
if edges_df is not None:
    if os.path.exists(EMBED_CACHE):
        # Load from disk
        with open(EMBED_CACHE, 'rb') as f:
            choices, labels, label_embeddings = pickle.load(f)
        # Derive uris from choices if needed
        uris = [uri for uri in choices['end_node'].tolist()]
    else:
        # Compute and cache
        choices = edges_df[['relation','end_node']].dropna().drop_duplicates()
        uris = choices['end_node'].tolist()
        labels = [uri.split('/')[3].replace('_',' ') for uri in uris]
        label_embeddings = embed_model.encode(labels, convert_to_tensor=True)
        with open(EMBED_CACHE, 'wb') as f:
            pickle.dump((choices, labels, label_embeddings), f)
else:
    choices = None
    uris = []
    labels = []
    label_embeddings = None

# -----------------------------------------
# Semantic mapping: response to (relation, URI) using embeddings
# -----------------------------------------
def map_response_to_uris(response: str, threshold: float = 0.6):
    """
    Map Oracle response to one or more (relation, URI) pairs via embedding similarity.
    Returns all matches with similarity >= threshold; if none, returns the single best match.
    """
    if edges_df is None or label_embeddings is None:
        concept = response.split()[-1].lower()
        return [(None, f"/c/en/{concept}")]
    import torch
    query_emb = embed_model.encode(response, convert_to_tensor=True)
    sims = torch.nn.functional.cosine_similarity(query_emb, label_embeddings)
    sims_list = sims.cpu().tolist()
    # find indices above threshold
    matched = [i for i, s in enumerate(sims_list) if s >= threshold]
    if not matched:
        best_idx = max(range(len(sims_list)), key=lambda i: sims_list[i])
        matched = [best_idx]
    results = []
    for idx in matched:
        row = choices.iloc[idx]
        results.append((row['relation'], row['end_node']))
    return results


import functools
@functools.lru_cache(maxsize=None)
def queryConceptNet(candidate, allowed_rels_key=None):
    # allowed_rels_key is a comma-separated string key for caching
    allowed_rels = set(allowed_rels_key.split(',')) if allowed_rels_key else TOTAL_RELATIONS
    base_url = "https://api.conceptnet.io/c/en/"
    candidate_url = base_url + candidate.lower()
    try:
        resp = session.get(candidate_url, timeout=5)
        data = resp.json()
    except Exception as e:
        logger.error("Failed ConceptNet query for %s: %s", candidate, e)
        return ""
    attributes = []
    for edge in data.get("edges", []):
        rel = edge.get("rel", {}).get("label", "")
        if rel in allowed_rels:
            end_label = edge.get("end", {}).get("label", "")
            if end_label and end_label.lower() != candidate.lower():
                attributes.append(end_label.lower())
    unique_attrs = list(set(attributes))
    return ", ".join(unique_attrs)

from collections import Counter
import math

def aggregateAttributes(candidates, allowed_rels={"MadeOf", "UsedFor", "HasProperty"}, k=None, use_entropy=False):
    # Build a count of how many candidates contain each attribute
    counter = Counter()
    for candidate in candidates:
        cleaned_candidate = clean_candidate(candidate)
        if not cleaned_candidate:
            continue
        attrs = queryConceptNet(cleaned_candidate)
        # unique per candidate
        attrs_set = {x.strip() for x in attrs.split(",") if x.strip()}
        for attr in attrs_set:
            counter[attr] += 1

    num_candidates = len(candidates)
    if num_candidates == 0:
        if use_entropy:
            return "", 0.0
        return ""

    if use_entropy:
        best_attr = None
        best_ig = -1.0
        # Compute information gain for each attr based on split among candidates
        for attr, m in counter.items():
            # candidates with attr = m, without = num_candidates - m
            if m == 0 or m == num_candidates:
                ig = 0.0
            else:
                # IG = H(C) - expected H(C|attr)
                # H(C) = log2(N)
                Hc = math.log2(num_candidates)
                # H(C|present)=log2(m), H(C|absent)=log2(N-m)
                Hp = math.log2(m)
                Ha = math.log2(num_candidates - m)
                ig = Hc - (m/num_candidates)*Hp - ((num_candidates-m)/num_candidates)*Ha
            if ig > best_ig:
                best_ig = ig
                best_attr = attr
        return best_attr or "", best_ig

    # fallback to frequency-based top-k
    if k:
        attrs_list = [attr for attr, _ in counter.most_common(k)]
    return ", ".join(attrs_list)

# -------------------------------
# Alternative: Use ConceptNet relation sampling
# -------------------------------
import random

from concurrent.futures import ThreadPoolExecutor, as_completed

def generateCandidatesByRelations(positive_pairs, negative_pairs, n=None, threshold=10):
    """
    Pulls concepts from ConceptNet that satisfy positive (relation, concept) pairs (scoring by how many pairs each candidate matches)
    and excludes those matching any negative pairs. Returns top-n candidates by score.
    Parallelized for faster API queries.
    """
    from collections import Counter
    score_counter = Counter()

    # Helper for fetching terms for a (rel, concept) pair
    def fetch_terms(rel, concept):
        mapped_uri = concept if concept.startswith('/c/') else concept
        url = f'https://api.conceptnet.io/query?rel=/r/{rel}&end={mapped_uri}'
        try:
            resp = session.get(url, timeout=5)
            edges = resp.json().get('edges', [])
            return rel, {edge['start']['label'].lower() for edge in edges if edge.get('start',{}).get('label')}
        except Exception as e:
            logger.error('Relation fetch failed for %s,%s: %s', rel, concept, e)
            return rel, set()

    # Parallel positive fetch
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(fetch_terms, rel, concept) for rel, concept in positive_pairs]
        for future in as_completed(futures):
            rel, terms = future.result()
            for term in terms:
                score_counter[term] += 1

    candidates = set(score_counter)

    # Parallel negative fetch
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(fetch_terms, rel, concept) for rel, concept in negative_pairs]
        for future in as_completed(futures):
            rel, neg_terms = future.result()
            candidates -= neg_terms

    # Only keep candidates matching a strict majority of positive relations
    if positive_pairs:
        threshold_val = (len(positive_pairs)//threshold)-1
        return [term for term, score in score_counter.items() if score >= threshold_val]
    return []



# -------------------------------
# Retroactive Information Gain Measurement
# -------------------------------
import csv
import math
from functools import lru_cache

# Cache ConceptNet queries for speed
@lru_cache(maxsize=None)
def has_relation(obj: str, rel: str, concept: str):
    """Return True if ConceptNet reports (obj) --rel--> (concept)"""
    attrs = queryConceptNet(clean_candidate(obj), allowed_rels_key=rel)
    return concept.lower() in [x.strip() for x in attrs.split(',') if x.strip()]

# Compute information gain for a given relation/concept over the domain
def compute_ig(rel: str, concept: str, domain: list):
    """
    Compute information gain for (rel,concept) over the domain using the preloaded CSV.
    """
    N = len(domain)
    if N == 0:
        return 0.0
    if len(rel_edges_df):
        rel_edges_df = rel_edges_df

    # If CSV loaded, use it; else fallback to API-based has_relation
    if rel_edges_df is not None:
        rel_uri = f"{rel}"
        concept_uri = f"/c/en/{concept}"
        # Build set of domain URIs
        start_uris = {f"/c/en/{clean_candidate(obj)}" for obj in domain}
        # Filter CSV for matching edges
        subset = rel_edges_df[
            (rel_edges_df['relation'] == rel_uri) &
            (rel_edges_df['end_node'] == concept_uri)
        ]
        m_set = set(subset['start_node'].tolist())
        m = sum(1 for uri in start_uris if uri in m_set)
    else:
        # fallback to API lookup
        m = sum(1 for obj in domain if has_relation(obj, rel, concept))
    # if no split
    if m == 0 or m == N:
        return 0.0
    # H(C) = log2(N)
    Hc = math.log2(N)
    # H(C|present) = log2(m), H(C|absent)=log2(N-m)
    Hp = math.log2(m) if m > 0 else 0.0
    Ha = math.log2(N - m) if N - m > 0 else 0.0
    # expected conditional entropy
    Hcond = (m / N) * Hp + ((N - m) / N) * Ha
    ig = Hc - Hcond
    return ig

# Retroactively compute IG for a game history file
def retro_compute_ig(history_file: str, output_csv: str, threshold=.6):
    # prepare output header if needed
    if not os.path.exists(output_csv):
        with open(output_csv, 'w', newline='', encoding='utf8') as outf:
            writer = csv.writer(outf)
            writer.writerow(['secret', 'question', 'ig'])

    # Open output file once for appending
    outf = open(output_csv, 'a', newline='', encoding='utf8')
    writer = csv.writer(outf)

    # process each game
    with open(history_file, 'r', encoding='utf8') as hf:
        for line_num, line in enumerate(hf):
            parts = line.strip().split(',', 2)
            if len(parts) < 3:
                continue
            secret, num, rest = parts
            fields = rest.split('\t')
            if num == 50:
                continue
            # iterate question/answer pairs
            for i in range(0, len(fields) - 2, 2):
                question = fields[i+1].strip()
                answer = fields[i+2].replace('Oracle said:', '').strip()
                mappings = map_response_to_uris(answer, threshold=threshold)
                # Build per-question domain and yes-set via precomputed maps
                domain_nodes = set()
                yes_nodes = set()
                for rel, uri in mappings:
                    domain_nodes |= domain_map.get(rel, set())
                    yes_nodes |= yes_map.get((rel, uri), set())
                N = len(domain_nodes)
                m = len(yes_nodes)
                # compute information gain: IG = H(C) - expected H(C|Q)
                if m == 0 or m == N:
                    ig = 0.0
                else:
                    import math
                    Hc = math.log2(N)
                    Hp = math.log2(m) if m > 0 else 0.0
                    Ha = math.log2(N - m) if N - m > 0 else 0.0
                    ig = Hc - (m/N)*Hp - ((N-m)/N)*Ha
                # append result immediately
                writer.writerow([secret, question, ig])
    outf.close()
    logger.info(f"Retro IG records written to {output_csv}")

# Example usage:
# for threshold in [.5,.6,.65,.7,.75,.8,.85]:
for threshold in [.85]:
    retro_compute_ig('Raw Text Results Final/LLamaNaturalOpenResults.txt', f'ConceptInfoGain/naturalOpen_ig_{threshold}_output.csv', threshold=threshold)
