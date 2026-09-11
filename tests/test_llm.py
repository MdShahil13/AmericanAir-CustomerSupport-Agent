from src.agent.llm_client import generate_reply


customer_message = """
My flight was cancelled and I need help getting rebooked.
"""

historical_cases = """
Case 1:
Customer: My flight from ALB to PHL via AmericanAir has been canceled and the earliest flight I can rebook is Tuesday.
AmericanAir: Please DM your record locator, and we'll review options with you, Tyra.

Case 2:
Customer: My flight was Friday. It is hard trying to get in contact with you. How can I change my flight?
AmericanAir: We're happy to take a look. Please DM your record locator.

Case 3:
Customer: My flight was cancelled and I need to rebook.
AmericanAir: Please give us a call or send a DM our way with your record locator code.
"""

prompt = f"""
You are an AmericanAir customer support assistant.

Your job is to draft a response using ONLY the historical
AmericanAir responses provided below.

Do NOT invent:
- phone numbers
- policies
- refunds
- vouchers
- upgrades
- website instructions
- facts not present in the evidence

If the evidence does not contain enough information, say that
a human agent should review the case.

CUSTOMER MESSAGE:
{customer_message}

HISTORICAL AMERICANAIR RESPONSES:
{historical_cases}

Write a short, professional customer-support reply.
"""

reply = generate_reply(prompt)

print("Grounded Groq response:")
print(reply)