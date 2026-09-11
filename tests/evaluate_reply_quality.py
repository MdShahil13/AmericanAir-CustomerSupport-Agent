import json
import re

import pandas as pd

from src.agent.support_agent import AmericanAirSupportAgent
from src.agent.llm_client import generate_reply


GOLDEN_PATH = "data/american_air_golden_set_final.csv"
HISTORICAL_PATH = "data/american_air_support_pairs.csv"

OUTPUT_PATH = "data/reply_quality_evaluation_20.csv"


def extract_customer_message(conversation: str) -> str:
    """
    Current evaluation assumption:
    use the final message in the conversation.
    """
    text = str(conversation).replace("\\n", "\n").strip()

    messages = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    return messages[-1] if messages else ""


def clean_json_response(text: str):
    """
    Extract a JSON object even if the model wraps it in markdown.
    """
    text = text.strip()

    # Remove markdown code fences if present.
    text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```\s*", "", text)

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if not match:
        raise ValueError(f"Could not find JSON in judge response:\n{text}")

    return json.loads(match.group(0))


def judge_reply(
    customer_message: str,
    predicted_intent: str,
    generated_reply: str,
    historical_evidence: str,
):
    prompt = f"""
You are evaluating an AI customer-support reply for an AmericanAir
customer-support agent.

Customer message:
{customer_message}

Predicted intent:
{predicted_intent}

Historical evidence:
{historical_evidence}

Generated reply:
{generated_reply}

Evaluate ONLY the generated reply using the historical evidence.

Give a score from 1 to 5 for each:

groundedness:
1 = unsupported / contradicts evidence
3 = partly supported
5 = clearly supported by evidence

relevance:
1 = does not address the customer
3 = partially addresses the issue
5 = directly addresses the issue

completeness:
1 = useless or incomplete
3 = partially useful
5 = gives the useful supported next step

safety:
1 = serious unsupported claim or hallucination
3 = minor unsupported detail
5 = no meaningful hallucination or unsupported claim

Also provide:
overall_score = average of the four scores

Return ONLY valid JSON in this exact structure:

{{
  "groundedness": 1,
  "relevance": 1,
  "completeness": 1,
  "safety": 1,
  "overall_score": 1.0,
  "reason": "brief explanation"
}}
"""

    result = generate_reply(
        prompt,
        max_completion_tokens=300,
    )

    return clean_json_response(result)


def main():

    golden = pd.read_csv(GOLDEN_PATH)

    # Pilot evaluation: 20 fixed examples.
    sample = golden.sample(
        n=20,
        random_state=42,
    ).copy()

    agent = AmericanAirSupportAgent(
        HISTORICAL_PATH
    )

    results = []

    print("Running reply-quality evaluation on 20 examples...\n")

    for position, (_, row) in enumerate(
        sample.iterrows(),
        start=1,
    ):

        customer_message = extract_customer_message(
            row["conversation"]
        )

        print(
            f"[{position:02d}/20] "
            f"Generating reply..."
        )

        result = agent.respond(
            customer_message
        )

        predicted_intent = result["intent"]
        generated_reply = result.get("reply") or ""

        cases = result.get("cases")

        evidence_parts = []

        if cases:
         for evidence_number, case in enumerate(
          cases[:5],
          start=1,
):
                evidence_parts.append(
                    f"""
Evidence {evidence_number}
Similarity: {case["similarity"]:.3f}
Customer: {case["customer_text"]}
AmericanAir: {case["americanair_response"]}
"""
                )

        historical_evidence = "\n".join(
            evidence_parts
        )

        # NON_SUPPORT may intentionally have no reply.
        if predicted_intent == "NON_SUPPORT":

            judge = {
                "groundedness": 5,
                "relevance": 5,
                "completeness": 5,
                "safety": 5,
                "overall_score": 5.0,
                "reason": (
                    "NON_SUPPORT message does not require "
                    "an automated customer-support reply."
                ),
            }

        elif not generated_reply.strip():

            judge = {
                "groundedness": 1,
                "relevance": 1,
                "completeness": 1,
                "safety": 5,
                "overall_score": 2.0,
                "reason": "No reply was generated.",
            }

        else:

            print(
                f"[{position:02d}/20] "
                f"Judging reply..."
            )

            try:

                judge = judge_reply(
                    customer_message=customer_message,
                    predicted_intent=predicted_intent,
                    generated_reply=generated_reply,
                    historical_evidence=historical_evidence,
                )

            except Exception as exc:

                print(
                    f"Judge failed for example "
                    f"{row['example_id']}: {exc}"
                )

                judge = {
                    "groundedness": None,
                    "relevance": None,
                    "completeness": None,
                    "safety": None,
                    "overall_score": None,
                    "reason": f"Judge error: {exc}",
                }

        results.append(
            {
                "example_id": row["example_id"],
                "gold_intent": row["intent"],
                "predicted_intent": predicted_intent,
                "customer_message": customer_message,
                "generated_reply": generated_reply,
                "decision": result["decision"],
                "decision_reason": result["reason"],
                "groundedness": judge["groundedness"],
                "relevance": judge["relevance"],
                "completeness": judge["completeness"],
                "safety": judge["safety"],
                "overall_score": judge["overall_score"],
                "judge_reason": judge["reason"],
            }
        )

    output = pd.DataFrame(results)

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n" + "=" * 65)
    print("REPLY QUALITY EVALUATION")
    print("=" * 65)

    numeric_columns = [
        "groundedness",
        "relevance",
        "completeness",
        "safety",
        "overall_score",
    ]

    for column in numeric_columns:

        values = pd.to_numeric(
            output[column],
            errors="coerce",
        ).dropna()

        if len(values) > 0:

            print(
                f"{column:15s}: "
                f"{values.mean():.2f}"
            )

    overall_values = pd.to_numeric(
        output["overall_score"],
        errors="coerce",
    ).dropna()

    if len(overall_values) > 0:

        strong_reply_rate = (
            overall_values.ge(4.0).mean()
        )

        print(
            f"\nOverall score >= 4: "
            f"{strong_reply_rate * 100:.1f}%"
        )

    print(
        f"\nSaved results to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()