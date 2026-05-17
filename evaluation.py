from utilities import *
from modules import *
from framework_stg import *
import time
import os
import re
import string
import pandas as pd
from collections import Counter
from sacrebleu.metrics import BLEU
from nltk.translate.bleu_score import sentence_bleu


RE_ART  = re.compile(r'\b(a|an|the)\b')
RE_PUNC = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')


def _remove_articles(t):
    return RE_ART.sub(' ', t)


def _white_space_fix(t):
    return ' '.join(t.split())


def _remove_punc(t):
    return RE_PUNC.sub(' ', t)


def _lower(t):
    return t.lower()


def normalize(text: str) -> str:
    """
    Lower text and remove punctuation,
    articles and extra whitespace.
    """
    return _white_space_fix(
        _remove_articles(
            _remove_punc(
                _lower(text)
            )
        )
    )


# ========================================================
# BLEU NORMAL
# ========================================================

def bleu_sacre_normal(reference: str, hypothesis: str) -> float:

    bleu = BLEU(effective_order=True)

    result = bleu.sentence_score(
        hypothesis,
        [reference]
    )

    return result.score / 100.0


# ========================================================
# SGPT F1
# ========================================================

def sgpt_f1(reference: str, hypothesis: str) -> dict:

    ref_tokens = normalize(reference).split()
    hyp_tokens = normalize(hypothesis).split()

    if not ref_tokens and not hyp_tokens:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0
        }

    if not ref_tokens or not hyp_tokens:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }

    common_tokens = sum(
        (
            Counter(ref_tokens)
            &
            Counter(hyp_tokens)
        ).values()
    )

    precision = common_tokens / len(hyp_tokens)
    recall    = common_tokens / len(ref_tokens)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (
            2 * precision * recall
        ) / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def test_pipeline(a:str):
  try:
    df = pd.read_json(a)
  except:
    df = pd.read_csv(a)
  SPARQL_query = []
  NL_questions = []
  for items in df["sparql"]:
    SPARQL_query.append(items)
  for items in df["question_en"]:
    NL_questions.append(items)
  return SPARQL_query, NL_questions

if __name__ == "__main__":
    pass