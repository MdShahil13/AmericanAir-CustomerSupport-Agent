class EscalationDecision:
    def decide(self, intent: str, cases, reply: str):
        # No evidence
        if cases is None or cases.empty:
            return {
                "decision": "ESCALATE",
                "reason": "No sufficiently similar historical AmericanAir evidence was found."
            }

        top_similarity = cases["similarity"].iloc[0]

        # Weak evidence
        if top_similarity < 0.60:
            return {
                "decision": "ESCALATE",
                "reason": f"Historical evidence is weak (top similarity: {top_similarity:.3f})."
            }

        # Intents that commonly need human intervention
        high_risk_intents = {
            "Refund & Compensation",
            "Accessibility",
            "Baggage",
        }

        if intent in high_risk_intents:
            return {
                "decision": "ESCALATE",
                "reason": f"{intent} may require case-specific human assistance."
            }

        return {
            "decision": "AUTO_HANDLE",
            "reason": "Strong historical evidence was found and the request is suitable for automated handling."
        }