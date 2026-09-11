import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

from src.classifier.hybrid_intent_classifier import HybridIntentClassifier
from src.retrieval.retriever import AmericanAirRetriever
from src.escalation.decision import EscalationDecision


GOLDEN_PATH = "data/american_air_golden_set_final.csv"
RETRIEVAL_PATH = "data/american_air_support_pairs.csv"


def extract_customer_message(conversation: str) -> str:
    """
    Golden-set conversations contain literal '\\n'.
    Current evaluation assumption:
    use the final message in the conversation.
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
    df = pd.read_csv(GOLDEN_PATH)

    classifier = HybridIntentClassifier()
    retriever = AmericanAirRetriever(RETRIEVAL_PATH)
    escalation = EscalationDecision()

    true_intents = []
    predicted_intents = []

    true_escalations = []
    predicted_escalations = []

    print(f"Evaluating {len(df)} golden examples...\n")

    for i, row in df.iterrows():

        customer_message = extract_customer_message(
            row["conversation"]
        )

        true_intent = row["intent"]
        true_escalation = row["escalation"]

        predicted_intent = classifier.predict(
            customer_message
        )

        # Retrieval
        if predicted_intent == "NON_SUPPORT":
            cases = pd.DataFrame()
        else:
            cases = retriever.retrieve(
                customer_message,
                intent=predicted_intent,
                top_k=10,
            )

            cases = cases[
                cases["similarity"] >= 0.55
            ].copy()

        # Current escalation logic does not depend
        # on the generated reply text.
        decision = escalation.decide(
            predicted_intent,
            cases,
            "",
        )

        predicted_escalation = decision["decision"]

        true_intents.append(true_intent)
        predicted_intents.append(predicted_intent)

        true_escalations.append(true_escalation)
        predicted_escalations.append(predicted_escalation)

        print(
            f"[{i + 1:03d}/{len(df)}] "
            f"Intent={predicted_intent:<28} "
            f"Expected={true_intent:<28} "
            f"Escalation={predicted_escalation}"
        )

    # --------------------------------------------------
    # INTENT METRICS
    # --------------------------------------------------
    intent_accuracy = accuracy_score(
        true_intents,
        predicted_intents,
    )

    intent_macro_f1 = f1_score(
        true_intents,
        predicted_intents,
        average="macro",
        zero_division=0,
    )

    intent_weighted_f1 = f1_score(
        true_intents,
        predicted_intents,
        average="weighted",
        zero_division=0,
    )

    # --------------------------------------------------
    # ESCALATION METRICS
    # --------------------------------------------------
    escalation_accuracy = accuracy_score(
        true_escalations,
        predicted_escalations,
    )

    escalation_macro_f1 = f1_score(
        true_escalations,
        predicted_escalations,
        average="macro",
        zero_division=0,
    )

    print("\n" + "=" * 70)
    print("FINAL EVALUATION RESULTS")
    print("=" * 70)

    print("\nINTENT")
    print(f"Accuracy     : {intent_accuracy:.4f}")
    print(f"Macro F1     : {intent_macro_f1:.4f}")
    print(f"Weighted F1  : {intent_weighted_f1:.4f}")

    print("\nIntent Classification Report")
    print(
        classification_report(
            true_intents,
            predicted_intents,
            zero_division=0,
        )
    )

    print("\nESCALATION")
    print(f"Accuracy     : {escalation_accuracy:.4f}")
    print(f"Macro F1     : {escalation_macro_f1:.4f}")

    print("\nEscalation Classification Report")
    print(
        classification_report(
            true_escalations,
            predicted_escalations,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()