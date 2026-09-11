import re

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.classifier.hybrid_intent_classifier import HybridIntentClassifier


GOLDEN_SET = "data/american_air_golden_set_final.csv"


def extract_latest_customer_message(conversation: str) -> str:
    """
    Our reconstructed conversations end with the latest customer message.
    The CSV stores '\\n' as literal characters, so convert them first.
    """

    text = str(conversation).replace("\\n", "\n")

    messages = []

    for line in text.splitlines():
        line = re.sub(r"^\s*\d+\.\s*", "", line).strip()

        if line:
            messages.append(line)

    if not messages:
        return ""

    return messages[-1]


def main():
    df = pd.read_csv(GOLDEN_SET)
    
    classifier = HybridIntentClassifier()

    customer_messages = []
    predictions = []

    for conversation in df["conversation"]:
        customer_message = extract_latest_customer_message(
            conversation
        )

        customer_messages.append(customer_message)

        predictions.append(
            classifier.predict(customer_message)
        )

    df["customer_message"] = customer_messages
    df["predicted_intent"] = predictions

    accuracy = accuracy_score(
        df["intent"],
        df["predicted_intent"]
    )

    print("=" * 70)
    print("AMERICANAIR INTENT CLASSIFIER EVALUATION")
    print("=" * 70)

    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            df["intent"],
            df["predicted_intent"],
            zero_division=0
        )
    )

    labels = sorted(
        set(df["intent"]) |
        set(df["predicted_intent"])
    )

    matrix = confusion_matrix(
        df["intent"],
        df["predicted_intent"],
        labels=labels
    )

    print("\nConfusion Matrix:")
    print(
        pd.DataFrame(
            matrix,
            index=labels,
            columns=labels
        )
    )

    print("\nSample Inputs:")
    print(
        df[
            [
                "customer_message",
                "intent",
                "predicted_intent"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()