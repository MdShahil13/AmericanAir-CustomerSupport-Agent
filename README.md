# ✈️ AmericanAir AI Support Agent

An AI-powered customer support agent built for **AmericanAir** using the
**Customer Support on Twitter** dataset.

The system combines hybrid intent classification, semantic retrieval of
historical AmericanAir support cases, evidence-grounded LLM response
generation, and escalation decisions.

---

## What the Agent Does

The agent performs three main tasks:

1. **Intent classification**  
   Classifies an incoming customer message into a fixed set of support
   intents.

2. **Evidence-grounded response generation**  
   Retrieves similar historical AmericanAir support cases and uses their
   responses as evidence when drafting a reply.

3. **Escalation decision**  
   Decides whether the request should be `AUTO_HANDLE` or `ESCALATE`, with a
   short reason.

The final agent output has the following structure:

```text
Intent: <predicted intent>
Reply: <customer-facing response>
Action: AUTO_HANDLE OR ESCALATE
Reason: <reason for the decision>
```

---

## Problem Framing

The goal is not to replace a human customer-support team.

The goal is to assist with common AmericanAir customer-support requests by
combining historical support behavior with an LLM.

For this project, a good response should:

- Correctly identify the customer's support intent.
- Be grounded in how AmericanAir historically responded to similar problems.
- Avoid unsupported policies, guarantees, or claims.
- Provide a useful response for common requests.
- Escalate sensitive, ambiguous, unusual, or case-specific requests.

The system therefore focuses on:

- Intent classification
- Historical evidence retrieval
- Evidence-grounded response drafting
- Escalation recommendation

### What I Chose Not to Build

I did not build:

- An autonomous system that performs refunds, cancellations, or account
  changes.
- A system that claims an action has been completed when it has not.
- A fine-tuned customer-support LLM.
- A complete production customer-support platform with authentication,
  monitoring, agent tooling, and persistent customer state.

The result is a **decision-support and response-drafting prototype** rather
than a production customer-service system.

---

## Architecture

```text
                         Customer Message
                                |
                                v
                    +----------------------+
                    | Hybrid Intent        |
                    | Classification       |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | Historical Retrieval |
                    | Sentence Embeddings  |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | Grounded Response    |
                    | Generation           |
                    | GPT-OSS 20B          |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | Escalation Decision  |
                    +----------+-----------+
                               |
                    +----------+----------+
                    |                     |
                    v                     v
              AUTO_HANDLE             ESCALATE
                    |                     |
                    +----------+----------+
                               |
                               v
                         Reason / Reply
```

---

## Dataset

The project uses the **Customer Support on Twitter** dataset from Kaggle.

The dataset contains millions of customer-support tweets and covers multiple
brands.

Only **AmericanAir** interactions are used in this project.

Historical AmericanAir interactions were reconstructed into customer-support
pairs containing:

- Customer message
- Historical AmericanAir response

### Current Development Corpus

The current retrieval corpus contains:

**6,343 AmericanAir support pairs**

These pairs are used for semantic retrieval and response grounding.

### Golden Evaluation Set

A separate evaluation set of:

**200 examples**

was created for model evaluation.

The golden set contains:

- Support and non-support messages
- Multiple customer-support intents
- `AUTO_HANDLE` and `ESCALATE` labels

The labels were created using **machine-assisted annotation followed by human
review and correction**.

They should therefore not be described as a fully independent
multi-annotator human benchmark.

---

## Intent Taxonomy

The system uses a fixed taxonomy of 11 intents:

1. `Flight Disruption`
2. `Baggage`
3. `Booking & Payment`
4. `Check-in & Airport`
5. `Seats & Upgrades`
6. `Refund & Compensation`
7. `Loyalty / AAdvantage`
8. `Accessibility`
9. `Onboard Services`
10. `Other / General Support`
11. `NON_SUPPORT`

A fixed taxonomy was chosen instead of allowing the LLM to generate arbitrary
intent names so that predictions can be evaluated consistently.

---

## Classification

The classifier uses a hybrid strategy.

High-confidence rules are applied first for strong signals such as:

- Lost baggage
- Cancelled flights
- Refund requests
- Seat upgrades
- AAdvantage
- Wheelchair or accessibility requests

The system then uses a rule-based classifier and falls back to an LLM
classifier for ambiguous messages.

This design keeps obvious predictions interpretable while allowing semantic
reasoning for less explicit requests.

---

## Retrieval

Historical support pairs are represented as:

```text
Customer problem
+
Historical AmericanAir response
```

Each pair is embedded using:

```text
all-MiniLM-L6-v2
```

Cosine similarity is used to retrieve semantically similar historical
support cases.

The system retrieves up to the **top 10 historical cases** for an incoming
customer message.

These cases contain both:

```text
Customer message
Historical AmericanAir response
Similarity score
```

The retrieved evidence is then provided to the response generator.

### Embedding Cache

Embeddings are generated once and stored locally:

```text
data/
└── embeddings/
    └── american_air_embeddings.npy
```

This prevents the project from repeatedly recomputing embeddings during
development and evaluation.

---

## Response Generation

The system uses:

```text
openai/gpt-oss-20b
```

for response generation.

The model receives:

- Customer message
- Predicted intent
- Retrieved historical AmericanAir cases

The response-generation prompt instructs the model to:

- Use only information supported by historical evidence.
- Avoid inventing policies or procedures.
- Avoid unsupported phone numbers and URLs.
- Avoid claiming an action has already been completed.
- Adapt historical responses rather than copying them blindly.
- Keep the response concise and professional.
- Escalate when the evidence is insufficient.

### Response Safety

The system contains safeguards for:

- Empty responses
- Incomplete responses
- Weak historical evidence
- Sensitive or higher-risk intents

When response generation fails or evidence is insufficient, the system can
fall back to human review.

---

## Escalation

The agent returns one of two decisions:

```text
AUTO_HANDLE
```

or

```text
ESCALATE
```

The escalation policy considers:

- Whether historical evidence is available
- Strength of retrieved evidence
- The predicted intent
- Higher-risk support categories
- Response-generation failures

Selected higher-risk intents are handled conservatively, including:

- `Baggage`
- `Refund & Compensation`
- `Accessibility`

The goal is to reduce the chance of automatically handling cases that may
require case-specific human assistance.

---

## Streamlit Demo

The project includes a Streamlit interface for interactive testing.

The interface shows:

- Customer message input
- Quick example requests
- Predicted intent
- AUTO_HANDLE / ESCALATE action
- Decision reason
- Generated customer-facing reply
- Historical evidence
- Similarity scores

Run the application:

```powershell
$env:PYTHONPATH="."
streamlit run app/app.py --server.fileWatcherType none
```

Example:

```text
Customer Message:
My suitcase was lost and I need help finding it.

Intent:
Baggage

Action:
ESCALATE

Reason:
Baggage may require case-specific human assistance.

Generated Reply:
Please visit the Baggage office at the airport to file a claim for your lost
suitcase.
```

The generated reply is grounded in a retrieved historical case where
AmericanAir directed a customer to the Baggage office to file a claim.

---

# Evaluation

## Evaluation Methodology

The evaluation focuses on two completed quantitative components:

1. Intent classification
2. Escalation decision

A separate reply-quality evaluation was planned but is not reported as a
final benchmark in the current version.

The evaluation uses a **200-example golden set**.

Because the intent classes are imbalanced, **macro F1** is reported alongside
accuracy.

---

## Intent Classification

The intent classifier was evaluated against the 200-example golden set.

### Baselines

Two simple reference baselines are used:

1. **Majority-class baseline**  
   Always predicts the most frequent intent.

2. **Rule-based baseline**  
   Uses keyword-based intent rules without LLM fallback.

### Results

| Method | Accuracy | Macro F1 |
|---|---:|---:|
| Majority-class baseline | 25.0% | 0.036 |
| Rule-based baseline | 36.5% | 0.35 |
| Hybrid classifier | **38.0%** | **0.37** |

The current hybrid classifier provides a modest improvement over the
rule-based baseline.

### Current Hybrid Results

```text
Accuracy      : 38.0%
Macro F1      : 0.37
Weighted F1   : 0.39
```

Class-level results show stronger performance on several operational intents:

| Intent | F1 |
|---|---:|
| Baggage | 0.59 |
| Flight Disruption | 0.59 |
| Loyalty / AAdvantage | 0.57 |
| Onboard Services | 0.50 |
| Check-in & Airport | 0.42 |

Smaller categories should be interpreted carefully. For example, the
Accessibility class contains only one evaluation example.

---

## Escalation Evaluation

A full 200-example evaluation was performed for the current escalation
policy.

### Results

| Metric | Result |
|---|---:|
| Accuracy | **45.5%** |
| Macro F1 | **0.45** |

The evaluation set contains:

```text
AUTO_HANDLE : 112
ESCALATE    : 88
```

A trivial always-`AUTO_HANDLE` classifier would therefore achieve 56.0%
accuracy on this particular class distribution.

The current escalation policy performs below that trivial baseline and should
therefore be considered an area for further calibration rather than a
production-ready decision policy.

This is an important limitation of the current prototype.

---

# Baseline Comparison

The current results show the following progression:

```text
Majority baseline       25.0%
Rule-based baseline     36.5%
Hybrid classifier       38.0%
```

The hybrid approach provides a small gain over the rule-based system, but the
improvement is not large enough to claim that the classifier is production
ready.

The main value of the current system is therefore the complete
classification + retrieval + grounded generation + escalation workflow, not
a single headline accuracy number.

---

# Failure Analysis

## 1. Confusion with Other / General Support

A large number of ambiguous or dissatisfaction-heavy messages are routed to
`Other / General Support`.

### Example

A customer may complain broadly about AmericanAir without clearly expressing
a specific operational request.

### Hypothesis

The current taxonomy does not always provide enough information to separate
generic dissatisfaction from a specific support intent.

### Improvement

Introduce clearer intent boundaries and hierarchical classification.

---

## 2. NON_SUPPORT Confusion

The classifier sometimes predicts a support intent for messages that are
actually compliments, informational statements, or general conversation.

### Hypothesis

Customer-support tweets are noisy and many messages contain airline-related
words without actually requesting support.

### Improvement

Create a stronger dedicated `NON_SUPPORT` detector using both linguistic
patterns and classifier confidence.

---

## 3. Operational Retrieval Mismatch

Semantic retrieval can return a case that is broadly related but operationally
different.

For example, a cancellation request can retrieve another flight or booking
issue before the most directly relevant cancellation/rebooking case.

### Hypothesis

Sentence embeddings capture broad semantic similarity but do not fully
capture operational distinctions.

### Improvement

Use a reranking stage combining:

- semantic similarity
- intent match
- lexical overlap
- operational keywords

---

## 4. Conservative Escalation

Some valid support requests are escalated because the current policy is
intentionally conservative for selected higher-risk intents.

### Hypothesis

The policy currently uses simple deterministic thresholds rather than a
calibrated confidence model.

### Improvement

Collect more escalation labels and optimize the policy for false
auto-handle risk.

---

## 5. LLM Response Failures

The LLM can occasionally produce an incomplete or empty response.

### Hypothesis

Generation depends on model/provider behavior and available completion
budget.

### Improvement

Continue using completion safeguards and add stronger structured-output
validation before accepting a generated reply.

---

# What Is Misleading About My Headline Number?

The most potentially misleading number is:

**38.0% intent accuracy**

It should not be interpreted as the overall accuracy of the AI support agent.

Several factors limit how the number should be interpreted:

- The evaluation set contains only 200 examples.
- Intent classes are imbalanced.
- The labels were machine-assisted and then human reviewed/corrected.
- The evaluation uses an assumption that the final message in a stored
  conversation represents the customer message being evaluated.
- Intent accuracy does not measure retrieval quality.
- Intent accuracy does not measure reply quality.
- Intent accuracy does not measure escalation safety.
- Retrieval and evaluation originate from the same underlying dataset
  ecosystem, so overlap and near-duplicate risk must be considered.

Therefore, 38.0% is best interpreted as a **prototype benchmark on a
specific golden set**, not as a production accuracy claim.

---

# Limitations

The current prototype has several limitations:

- Small 200-example evaluation set
- Imbalanced intent classes
- Machine-assisted, human-reviewed labels
- Simple escalation policy
- No calibrated confidence model
- Semantic retrieval without a learned reranker
- No statistically strong human reply-quality benchmark yet
- No independent multi-annotator agreement study yet
- Potential retrieval/evaluation source overlap
- No real-time airline operational data
- No production CRM or ticketing integration
- No autonomous customer-service actions

---

# What I Chose Not to Build

This project intentionally does not:

- Execute refunds
- Cancel or modify bookings
- Change customer accounts
- Claim that a support action has been completed
- Access live airline operational systems
- Replace a human support team
- Provide a production customer-service platform

The prototype focuses on **decision support and response drafting**.

---

# What I Would Do With One More Week

## 1. Build a Strictly Held-Out Retrieval Benchmark

Separate evaluation examples from the retrieval corpus before generating
embeddings.

This would make retrieval evaluation much more trustworthy.

## 2. Improve Human Annotation

Create a larger evaluation set with multiple independent reviewers and
measure agreement.

## 3. Improve Retrieval Ranking

Add a reranker using:

- semantic similarity
- intent compatibility
- lexical overlap
- operational keywords

## 4. Calibrate Escalation

Build a larger escalation benchmark and optimize for the most important
error:

**false AUTO_HANDLE on cases that should go to humans.**

## 5. Add Reply-Quality Evaluation

Use a structured rubric for:

- Groundedness
- Relevance
- Completeness
- Professionalism
- Unsupported claims

Then compare LLM-judge results with independent human judgments.

## 6. Evaluate Unseen Customer Messages

Create a genuinely held-out benchmark so that the system cannot retrieve the
same or near-duplicate evaluation cases.

## 7. Add Confidence Monitoring

Log:

- classifier confidence
- retrieval similarity
- escalation reason
- generation failures

This would make the system easier to audit and improve.

---

# Project Structure

```text
AmericanAir-CustomerSupport-Agent/
│
├── data/
│   ├── american_air_golden_set_final.csv
│   ├── american_air_support_pairs.csv
│   └── embeddings/
│       └── american_air_embeddings.npy
│
├── src/
│   ├── classifier/
│   │   ├── __init__.py
│   │   ├── intent_classifier.py
│   │   ├── llm_intent_classifier.py
│   │   └── hybrid_intent_classifier.py
│   │
│   ├── retrieval/
│   │   └── retriever.py
│   │
│   ├── agent/
│   │   ├── pipeline.py
│   │   ├── support_agent.py
│   │   └── llm_client.py
│   │
│   ├── escalation/
│   │   └── decision.py
│   │
│   └── evaluation/
│       └── evaluate_intent.py
│
├── app/
│   └── app.py
│
├── tests/
│   ├── test_retrieval.py
│   ├── test_classifier.py
│   ├── test_pipeline.py
│   ├── test_llm.py
│   ├── test_agent.py
│   ├── test_hybrid_classifier.py
│   └── evaluate_final.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

# Decision Log

The major engineering decisions include:

1. Select AmericanAir as the single target brand.
2. Use a fixed support-intent taxonomy.
3. Add a dedicated `NON_SUPPORT` category.
4. Use a hybrid classifier instead of relying entirely on an LLM.
5. Use historical customer-response pairs as retrieval units.
6. Use sentence embeddings for semantic retrieval.
7. Cache embeddings locally.
8. Retrieve multiple historical cases instead of relying on a single match.
9. Ground response generation in historical evidence.
10. Restrict the LLM from inventing policies, URLs, phone numbers, or actions.
11. Add response-generation safeguards.
12. Use conservative handling for higher-risk intents.
13. Use Streamlit for a lightweight demonstration UI.
14. Keep the golden evaluation set separate from the main application input
    data.
15. Report limitations and negative evaluation findings instead of hiding
    them.

---

# Reproducibility

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create a `.env` file:

```text
GROQ_API_KEY=your_api_key_here
```

Do not commit `.env` or your API key to GitHub.

Run the Streamlit application:

```powershell
$env:PYTHONPATH="."
streamlit run app/app.py --server.fileWatcherType none
```

Run the final evaluation:

```powershell
python tests/evaluate_final.py
```

---

# Conclusion

This project demonstrates an end-to-end AI customer-support workflow for
AmericanAir:

```text
Customer Message
      ↓
Intent Classification
      ↓
Historical Evidence Retrieval
      ↓
Grounded Response Generation
      ↓
Escalation Decision
```

The current prototype achieves **38.0% intent accuracy and 0.37 macro F1**
on a 200-example golden evaluation set.

More importantly, the project demonstrates the engineering trade-offs
required for a support agent: historical grounding, conservative automation,
retrieval quality, and explicit handling of uncertainty.

The current results show that the prototype is functional, but that
classification and escalation calibration still require further work before
production deployment.
