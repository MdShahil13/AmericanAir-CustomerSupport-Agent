from src.classifier.llm_intent_classifier import LLMIntentClassifier


classifier = LLMIntentClassifier()

message = "My suitcase is lost and I need help."

print("MESSAGE:")
print(message)

print("\nPREDICTED:")
print(classifier.predict(message))