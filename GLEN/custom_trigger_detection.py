"""
custom_trigger_detection.py
============================
Thin adapter around the GLEN model code (model/predict_sentence.py) that
reformats GLEN's output into the compact structure used by this pipeline:

    [
        {
            "sentence": "...",
            "triggers": [
                {"text": "died", "confidence": 0.98},
                ...
            ]
        },
        ...
    ]

Must be called with the working directory set to the GLEN/ folder so that the
relative imports inside model/ resolve correctly.  main.py and pipeline_steps.py
both do  os.chdir("GLEN")  before importing this module.

Requirements
------------
    pip install torch transformers
    # Checkpoint: GLEN/ckpts/type_classifier.bin
    # Download from: https://drive.google.com/file/d/1UU1UVPpYypRh5dPUhQ8TreAJd-uoLEh7/view
"""

from __future__ import annotations

import os
import sys
from typing import List, Dict, Any

# Ensure the GLEN root (this file's directory) is on sys.path so that
# `from model.xxx import ...` works regardless of caller's cwd.
_GLEN_ROOT = os.path.dirname(os.path.abspath(__file__))
if _GLEN_ROOT not in sys.path:
    sys.path.insert(0, _GLEN_ROOT)


def detect_triggers(
    sentences: List[str],
    ckpt_path: str,
    bs_TI: int = 32,
    bs_TC: int = 32,
    bs_TR: int = 4,
    k: int = 10,
) -> List[Dict[str, Any]]:
    """Run GLEN trigger detection on a list of sentences.

    Parameters
    ----------
    sentences:
        Plain-text sentences to analyse.
    ckpt_path:
        Path to the directory containing ``type_classifier.bin``.
        Can be relative (resolved from GLEN/) or absolute.
    bs_TI, bs_TC, bs_TR:
        Batch sizes for GLEN's three internal modules (Trigger Identification,
        Type Ranking, Type Classification).  Lower values use less GPU/CPU RAM.
    k:
        Number of candidate event types considered by the type classifier.

    Returns
    -------
    List of dicts, one per input sentence::

        {
            "sentence": str,
            "triggers": [
                {"text": str, "confidence": float},
                ...
            ]
        }

    Sentences with no detected triggers will have an empty ``triggers`` list.
    """
    from model.predict_sentence import setup, predict  # type: ignore
    from model.params import define_arguments          # type: ignore
    import argparse

    # Build params dict the same way parse_arguments() would, but without
    # reading sys.argv (which breaks when called from a larger pipeline).
    parser = argparse.ArgumentParser()
    define_arguments(parser)
    params = parser.parse_args([])  # parse empty args → all defaults
    params = vars(params)

    # Override the settings we care about
    params["path_to_ckpt"] = ckpt_path
    params["bs_TI"] = bs_TI
    params["bs_TC"] = bs_TC
    params["bs_TR"] = bs_TR
    params["k"] = k

    # Initialise models (loads checkpoints)
    tokenizer, trigger_identifier, type_ranking, type_classifier, device, \
        cand_encs, used_cand = setup(params)

    # Run prediction
    raw_results: List[Dict] = predict(
        sentences,
        params,
        tokenizer,
        trigger_identifier,
        type_ranking,
        type_classifier,
        device,
        cand_encs,
        used_cand,
    )

    # Reformat to compact pipeline format
    output: List[Dict[str, Any]] = []
    for item in raw_results:
        sentence: str = item.get("sentence", "")
        mentions: List[Dict] = item.get("predicted_mentions", [])

        triggers = []
        for mention in mentions:
            text: str = mention.get("trigger_words", "")
            confidence: float = float(mention.get("trigger_confidence", 0.0))
            if text:
                triggers.append({"text": text, "confidence": confidence})

        output.append({"sentence": sentence, "triggers": triggers})

    return output
