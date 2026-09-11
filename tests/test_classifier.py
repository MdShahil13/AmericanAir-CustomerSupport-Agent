from src.classifier.intent_classifier import AmericanAirIntentClassifier


classifier = AmericanAirIntentClassifier()

examples = [
    "My flight was cancelled and I need to rebook",
    "Where is my lost baggage?",
    "How can I get a refund?",
    "Can I select a window seat?",
    "I need wheelchair assistance",
    "Thanks AmericanAir, you were amazing"
]

for text in examples:
    print(f"{text}")
    print(f"Intent: {classifier.predict(text)}")
    print("-" * 60)