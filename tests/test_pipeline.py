from src.agent.pipeline import AmericanAirSupportPipeline


pipeline = AmericanAirSupportPipeline(
    "data/american_air_support_pairs.csv"
)

message = "My flight was cancelled and I need help getting rebooked."

result = pipeline.analyze(message)

print("CUSTOMER:")
print(message)

print("\nPREDICTED INTENT:")
print(result["intent"])

print("\nHISTORICAL CASES:")
print(
    result["cases"][
        [
            "customer_text",
            "americanair_response",
            "predicted_intent",
            "similarity"
        ]
    ].to_string(index=False)
)