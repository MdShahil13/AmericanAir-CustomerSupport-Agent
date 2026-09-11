import re
import pandas as pd
from src.retrieval.retriever import AmericanAirRetriever


GOLDEN_PATH = "data/american_air_golden_set_final.csv"
RETRIEVAL_PATH = "data/american_air_support_pairs.csv"


def normalize_text(text: str) -> str:
    """Normalize text for exact historical-case matching."""
    text = str(text).replace("\\n", "\n").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_customer_message(conversation: str) -> str:
    """
    Current evaluation assumption:
    the final message in the conversation is the customer message
    we evaluate.
    """
    text = str(conversation).replace("\\n", "\n").strip()

    messages = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if not messages:
        return ""

    return messages[-1]


def main():

    golden = pd.read_csv(GOLDEN_PATH)
    historical = pd.read_csv(RETRIEVAL_PATH)

    print(f"Golden examples      : {len(golden)}")
    print(f"Historical cases     : {len(historical)}")

    # Create normalized lookup of historical customer messages
    historical["normalized_customer"] = (
        historical["customer_text"]
        .fillna("")
        .astype(str)
        .map(normalize_text)
    )

    historical_lookup = {}

    for idx, row in historical.iterrows():

        key = row["normalized_customer"]

        if key:
            historical_lookup.setdefault(key, []).append(idx)

    retriever = AmericanAirRetriever(RETRIEVAL_PATH)

    recall_1 = 0
    recall_5 = 0
    recall_10 = 0

    covered_examples = 0

    print("\nRunning retrieval evaluation...\n")

    for i, row in golden.iterrows():

        customer_message = extract_customer_message(
            row["conversation"]
        )

        normalized_query = normalize_text(
            customer_message
        )

        # Find the historical case corresponding
        # to this golden customer message.
        gold_indices = historical_lookup.get(
            normalized_query,
            [],
        )

        if not gold_indices:
            print(
                f"[{i + 1:03d}/{len(golden)}] "
                "NO HISTORICAL MATCH"
            )
            continue

        covered_examples += 1

        # Keep retrieval intent-aware, matching current pipeline.
        intent = row["intent"]

        cases = retriever.retrieve(
            customer_message,
            intent=None if intent == "NON_SUPPORT" else intent,
            top_k=10,
        )

        retrieved_indices = list(cases.index)

        gold_set = set(gold_indices)

        hit_1 = any(
            idx in gold_set
            for idx in retrieved_indices[:1]
        )

        hit_5 = any(
            idx in gold_set
            for idx in retrieved_indices[:5]
        )

        hit_10 = any(
            idx in gold_set
            for idx in retrieved_indices[:10]
        )

        if hit_1:
            recall_1 += 1

        if hit_5:
            recall_5 += 1

        if hit_10:
            recall_10 += 1

        print(
            f"[{i + 1:03d}/{len(golden)}] "
            f"R@1={'✓' if hit_1 else '✗'} "
            f"R@5={'✓' if hit_5 else '✗'} "
            f"R@10={'✓' if hit_10 else '✗'}"
        )

    print("\n" + "=" * 65)
    print("RETRIEVAL EVALUATION")
    print("=" * 65)

    print(f"\nGolden examples             : {len(golden)}")
    print(f"Examples with exact history : {covered_examples}")

    if covered_examples == 0:
        print(
            "\nNo exact historical matches were found. "
            "Recall@K cannot be calculated."
        )
        return

    print(
        f"\nRecall@1   : "
        f"{recall_1 / covered_examples:.4f}"
    )

    print(
        f"Recall@5   : "
        f"{recall_5 / covered_examples:.4f}"
    )

    print(
        f"Recall@10  : "
        f"{recall_10 / covered_examples:.4f}"
    )

    coverage = covered_examples / len(golden)

    print(
        f"\nHistorical coverage : "
        f"{coverage:.4f}"
    )

    print("\nIMPORTANT:")
    print(
        "These metrics use exact customer-message matches "
        "between the golden set and the historical support-pairs "
        "dataset. Because both originate from the same source "
        "dataset, this evaluation may contain source overlap/"
        "leakage and must be disclosed in the final report."
    )


if __name__ == "__main__":
    main()