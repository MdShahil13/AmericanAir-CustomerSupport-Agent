from src.agent.support_agent import AmericanAirSupportAgent


agent = AmericanAirSupportAgent(
    "data/american_air_support_pairs.csv"
)

message = "My flight was cancelled and I need help getting rebooked."

result = agent.respond(message)

print("CUSTOMER:")
print(message)

print("\nINTENT:")
print(result["intent"])

print("\nDECISION:")
print(result["decision"])

print("\nREASON:")
print(result["reason"])

print("\nGENERATED REPLY:")
print(result["reply"])

print("\nEVIDENCE:")
for _, row in result["cases"].iterrows():
    print(
        f"- [{row['similarity']:.3f}] "
        f"{row['customer_text']} -> {row['americanair_response']}"
    )