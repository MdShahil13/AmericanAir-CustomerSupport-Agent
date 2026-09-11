from src.classifier.intent_classifier import AmericanAirIntentClassifier
from src.classifier.llm_intent_classifier import LLMIntentClassifier


class HybridIntentClassifier:
    def __init__(self):
        self.rule_classifier = AmericanAirIntentClassifier()
        self.llm_classifier = LLMIntentClassifier()

    def predict(self, customer_message: str) -> str:
        text = customer_message.lower().strip()

        # Very clear intent signals.
        strong_rules = {
            "Baggage": [
                "lost bag",
                "lost baggage",
                "missing bag",
                "missing baggage",
                "luggage is lost",
                "suitcase is lost",
            ],
            "Flight Disruption": [
                "flight cancelled",
                "flight canceled",
                "flight was cancelled",
                "flight was canceled",
                "flight delayed",
                "flight delay",
                "missed flight",
                "need to rebook",
                "rebook my flight",
            ],
            "Refund & Compensation": [
                "need a refund",
                "want a refund",
                "refund my",
                "money back",
                "compensation",
                "reimbursement",
            ],
            "Seats & Upgrades": [
                "window seat",
                "aisle seat",
                "seat selection",
                "upgrade my seat",
                "upgrade to first",
                "business class",
            ],
            "Loyalty / AAdvantage": [
                "aadvantage",
                "executive platinum",
                "elite status",
                "frequent flyer",
                "aadvantage miles",
            ],
            "Accessibility": [
                "wheelchair",
                "wheel chair",
                "mobility assistance",
                "accessible",
            ],
        }

        for intent, keywords in strong_rules.items():
            for keyword in keywords:
                if keyword in text:
                    return intent

        # Fall back to the existing rule classifier.
        rule_result = self.rule_classifier.predict(customer_message)

        # Obvious rule result: don't spend an LLM call.
        if rule_result != "Other / General Support":
            return rule_result

        # Ambiguous case → LLM.
        return self.llm_classifier.predict(customer_message)