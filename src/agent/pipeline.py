from src.classifier.hybrid_intent_classifier import HybridIntentClassifier
from src.retrieval.retriever import AmericanAirRetriever


class AmericanAirSupportPipeline:
    def __init__(self, data_path: str):
        self.classifier = HybridIntentClassifier()
        self.retriever = AmericanAirRetriever(data_path)

    def analyze(self, customer_message: str):
        intent = self.classifier.predict(customer_message)

        if intent == "NON_SUPPORT":
            return {
                "intent": intent,
                "cases": []
            }

        cases = self.retriever.retrieve(
            customer_message,
            intent=intent,
            top_k=10
        )

        return {
            "intent": intent,
            "cases": cases
        }