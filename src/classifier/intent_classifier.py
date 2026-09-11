import re


class AmericanAirIntentClassifier:
    """
    Lightweight rule-based baseline classifier.

    This is intentionally simple:
    the final agent will later compare this baseline
    against a stronger LLM-based classifier.
    """

    INTENTS = [
        "Flight Disruption",
        "Baggage",
        "Booking & Payment",
        "Check-in & Airport",
        "Seats & Upgrades",
        "Refund & Compensation",
        "Loyalty / AAdvantage",
        "Accessibility",
        "Onboard Services",
        "Other / General Support",
        "NON_SUPPORT",
    ]

    KEYWORDS = {
        "Flight Disruption": [
            "cancel", "cancelled", "canceled", "delay", "delayed",
            "late", "missed flight", "flight status", "diverted",
            "rebook", "rebooked", "disruption"
        ],
        "Baggage": [
            "bag", "bags", "baggage", "luggage", "suitcase",
            "lost bag", "missing bag", "checked bag"
        ],
        "Booking & Payment": [
            "book", "booking", "reservation", "ticket", "payment",
            "charge", "charged", "price", "fare"
        ],
        "Check-in & Airport": [
            "check in", "check-in", "boarding pass", "airport",
            "gate", "security", "boarding"
        ],
        "Seats & Upgrades": [
            "seat", "seating", "upgrade", "first class",
            "business class", "window seat", "aisle seat"
        ],
        "Refund & Compensation": [
            "refund", "compensation", "reimburse", "money back",
            "credit", "voucher"
        ],
        "Loyalty / AAdvantage": [
            "aadvantage", "miles", "points", "reward",
            "frequent flyer"
        ],
        "Accessibility": [
            "wheelchair", "disability", "accessible", "special assistance"
        ],
        "Onboard Services": [
            "wifi", "food", "meal", "drink", "entertainment",
            "movie", "on board", "onboard"
        ],
    }

    def predict(self, text: str) -> str:
        text = text.lower()

        if not text.strip():
            return "Other / General Support"

        # Simple non-support heuristic
        non_support_patterns = [
            r"\bthanks\b",
            r"\bthank you\b",
            r"\bgreat job\b",
            r"\blove american\b",
            r"\byou guys are awesome\b",
        ]

        if any(re.search(pattern, text) for pattern in non_support_patterns):
            return "NON_SUPPORT"

        scores = {}

        for intent, keywords in self.KEYWORDS.items():
            score = sum(
                1 for keyword in keywords
                if keyword in text
            )
            scores[intent] = score

        best_intent = max(scores, key=scores.get)

        if scores[best_intent] == 0:
            return "Other / General Support"

        return best_intent