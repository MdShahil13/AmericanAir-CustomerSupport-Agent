import pandas as pd

from src.classifier.hybrid_intent_classifier import HybridIntentClassifier
from src.retrieval.retriever import AmericanAirRetriever


GOLDEN_PATH = "data/american_air_golden_set_final.csv"
RETRIEVAL_PATH = "data/american_air_support_pairs.csv"
OUTPUT_PATH = "data/evidence_review_50.csv"


def extract_customer_message(conversation: str) -> str:
    """
    Current evaluation assumption:
    the final message in the conversation is the customer message.
    """
    text = str(conversation).replace("\\n", "\n").strip()

    messages = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    return messages[-1] if messages else ""


def main():

    golden = pd.read_csv(GOLDEN_PATH)

    # Fixed sample of 50 for reproducibility
    review_set = golden.sample(
        n=50,
        random_state=42
    ).copy()

    classifier = HybridIntentClassifier()
    retriever = AmericanAirRetriever(RETRIEVAL_PATH)

    rows = []

    print("Creating 50-example evidence review set...\n")

    for _, row in review_set.iterrows():

        customer_message = extract_customer_message(
            row["conversation"]
        )

        predicted_intent = classifier.predict(
            customer_message
        )

        if predicted_intent == "NON_SUPPORT":
            cases = pd.DataFrame()
        else:
            cases = retriever.retrieve(
                customer_message,
                intent=predicted_intent,
                top_k=5
            )

        # Keep the top 5 evidence cases in separate columns
        evidence = []

        if not cases.empty:
            for _, case in cases.iterrows():
                evidence.append({
                    "customer": str(case["customer_text"]),
                    "response": str(case["americanair_response"]),
                    "similarity": float(case["similarity"]),
                })

        # Make exactly 5 slots
        while len(evidence) < 5:
            evidence.append({
                "customer": "",
                "response": "",
                "similarity": ""
            })

        result = {
            "example_id": row["example_id"],
            "gold_intent": row["intent"],
            "predicted_intent": predicted_intent,
            "customer_message": customer_message,

            # Human fills this
            "useful_evidence": "",

            "evidence_1_customer": evidence[0]["customer"],
            "evidence_1_response": evidence[0]["response"],
            "evidence_1_similarity": evidence[0]["similarity"],

            "evidence_2_customer": evidence[1]["customer"],
            "evidence_2_response": evidence[1]["response"],
            "evidence_2_similarity": evidence[1]["similarity"],

            "evidence_3_customer": evidence[2]["customer"],
            "evidence_3_response": evidence[2]["response"],
            "evidence_3_similarity": evidence[2]["similarity"],

            "evidence_4_customer": evidence[3]["customer"],
            "evidence_4_response": evidence[3]["response"],
            "evidence_4_similarity": evidence[3]["similarity"],

            "evidence_5_customer": evidence[4]["customer"],
            "evidence_5_response": evidence[4]["response"],
            "evidence_5_similarity": evidence[4]["similarity"],
        }

        rows.append(result)

    output = pd.DataFrame(rows)

    output.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(f"\nCreated: {OUTPUT_PATH}")
    print("50 examples are ready for human evidence review.")


if __name__ == "__main__":
    main()