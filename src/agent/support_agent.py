from src.agent.llm_client import generate_reply
from src.agent.pipeline import AmericanAirSupportPipeline
from src.escalation.decision import EscalationDecision


class AmericanAirSupportAgent:
    def __init__(self, data_path: str):
        self.pipeline = AmericanAirSupportPipeline(data_path)
        self.escalation_decision = EscalationDecision()

    def respond(self, customer_message: str):
        result = self.pipeline.analyze(customer_message)

        # Non-support messages
        if result["intent"] == "NON_SUPPORT":
            return {
                "intent": "NON_SUPPORT",
                "reply": None,
                "decision": "AUTO_HANDLE",
                "reason": "Message does not require customer support handling.",
                "cases": []
            }

        cases = result["cases"]

        # Keep only reasonably strong evidence
        cases = cases[cases["similarity"] >= 0.55].copy()

        # No useful evidence
        if cases.empty:
            decision = self.escalation_decision.decide(
                result["intent"],
                cases,
                ""
            )

            return {
                "intent": result["intent"],
                "reply": "This case needs human review.",
                "decision": decision["decision"],
                "reason": decision["reason"],
                "cases": cases
            }

        historical_cases = ""

        for i, row in enumerate(cases.itertuples(), start=1):
            historical_cases += f"""
Case {i}:
Customer: {row.customer_text}
AmericanAir: {row.americanair_response}
Similarity: {row.similarity:.3f}
"""

        prompt = f"""
You are an AmericanAir customer support assistant.

Customer message:
{customer_message}

Predicted intent:
{result["intent"]}

Historical AmericanAir cases:
{historical_cases}

Write a short, professional response.

STRICT RULES:
1. Use only information supported by the historical cases.
2. Do not invent phone numbers, policies, refunds, vouchers,
   upgrades, URLs, or procedures.
3. Do not copy a historical response blindly; adapt it to the
   customer's message.
4. If the evidence is insufficient, say that human review is needed.
5. Keep the response concise.
6. Never include URLs, links, @handles, tweet IDs, or social-media
   references from the historical cases.
7. The response MUST be a complete sentence and must not end
   mid-sentence.
8. Prefer the actionable resolution present in the strongest
   historical evidence.
9. If the strongest evidence suggests contacting a team, filing
   a claim, or sending a record locator, you may mention that
   action only when it is explicitly supported by the evidence.
"""

        reply = generate_reply(prompt)
        if not reply or len(reply.strip()) < 20:
         return {
        "intent": result["intent"],
        "reply": "This case needs human review.",
        "decision": "ESCALATE",
        "reason": "The response generator did not produce a complete reply.",
        "cases": cases
    }

        decision = self.escalation_decision.decide(
            result["intent"],
            cases,
            reply
        )

        return {
            "intent": result["intent"],
            "reply": reply,
            "decision": decision["decision"],
            "reason": decision["reason"],
            "cases": cases
        }