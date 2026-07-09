from __future__ import annotations

import pandas as pd

from ethiclens_api.agent.demo_datasets import list_demo_datasets


def test_every_demo_dataset_csv_exists_and_parses():
    for dataset in list_demo_datasets():
        assert dataset.csv_path.exists(), f"{dataset.key}: missing CSV at {dataset.csv_path}"
        df = pd.read_csv(dataset.csv_path)
        assert not df.empty


def test_every_demo_proposal_columns_exist_in_its_csv():
    for dataset in list_demo_datasets():
        df = pd.read_csv(dataset.csv_path)
        proposal = dataset.proposal
        referenced = [
            proposal.outcome_column,
            *proposal.protected_attribute_columns,
            *proposal.feature_columns,
        ]
        if proposal.true_label_column:
            referenced.append(proposal.true_label_column)
        if proposal.score_column:
            referenced.append(proposal.score_column)
        missing = [c for c in referenced if c not in df.columns]
        assert not missing, f"{dataset.key}: proposal references missing columns {missing}"


def test_demo_dataset_keys_are_unique_and_match_registry():
    from ethiclens_api.agent.demo_datasets import DEMO_DATASETS

    for key, dataset in DEMO_DATASETS.items():
        assert dataset.key == key
