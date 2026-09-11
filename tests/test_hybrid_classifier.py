from src.classifier.hybrid_intent_classifier import HybridIntentClassifier


classifier = HybridIntentClassifier()

examples = [
    "My suitcase is lost.",
    "My flight was cancelled and I need to rebook.",
    "Can I get my money back?",
    "I want a window seat.",
    "I need wheelchair assistance.",
    "I just reached Executive Platinum!",
]

for message in examples:
    print(f"\nMessage: {message}")
    print(f"Intent: {classifier.predict(message)}")