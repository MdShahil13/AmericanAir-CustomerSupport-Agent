from src.retrieval.retriever import AmericanAirRetriever


retriever = AmericanAirRetriever(
    "data/american_air_support_pairs.csv"
)

results = retriever.retrieve(
    "My flight was cancelled, what should I do?",
    top_k=3
)

print(
    results[
        [
            "customer_text",
            "americanair_response",
            "predicted_intent",
            "similarity"
        ]
    ].to_string(index=False)
)