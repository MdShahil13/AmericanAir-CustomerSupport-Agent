from src.agent.llm_client import generate_reply


class LLMIntentClassifier:
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

    def predict(self, customer_message: str) -> str:
        prompt = f"""
You classify AmericanAir customer messages.

Choose exactly ONE intent from this list:

- Flight Disruption
- Baggage
- Booking & Payment
- Check-in & Airport
- Seats & Upgrades
- Refund & Compensation
- Loyalty / AAdvantage
- Accessibility
- Onboard Services
- Other / General Support
- NON_SUPPORT

Definitions:

Flight Disruption:
delay, cancellation, missed flight, rebooking, irregular operations.

Baggage:
lost, damaged, delayed, missing, or baggage-related issues.

Booking & Payment:
booking problems, ticket purchase, fare, payment, charges.

Check-in & Airport:
check-in, boarding pass, gate, airport procedures.

Seats & Upgrades:
seat selection, seat problems, upgrades, cabin/class issues.

Refund & Compensation:
refunds, compensation, reimbursement, vouchers.

Loyalty / AAdvantage:
miles, points, AAdvantage, elite status, loyalty benefits.

Accessibility:
wheelchair or accessibility assistance.

Onboard Services:
food, drinks, Wi-Fi, entertainment, cabin amenities.

Other / General Support:
a genuine support request that does not fit the categories above.

NON_SUPPORT:
praise, jokes, casual conversation, complaints with no actionable
support request, or social content that does not require assistance.

Customer message:
{customer_message}

Return ONLY the exact intent name.
"""

        result = generate_reply(
    prompt,
    max_completion_tokens=20
).strip().lower()
        # First try exact match.
        for intent in self.INTENTS:
            if result == intent.lower():
                return intent

        # Handle responses such as:
        # "The correct intent is: Baggage"
        # "Intent: Flight Disruption"
        # "I would classify this as Seats & Upgrades."
        for intent in self.INTENTS:
            if intent.lower() in result:
                return intent

        return "Other / General Support"