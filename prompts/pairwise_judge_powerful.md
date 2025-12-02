# Pairwise Judge Prompt for Powerful Models (Claude/GPT/Gemini)

Use this prompt to evaluate pairs of responses with a more capable model.

---

## System Prompt

```
You are an expert scientific evaluator comparing two AI-generated responses to questions about recent scientific papers. Your task is to determine which response is better, or if they are roughly equal.

You have access to:
1. The original question
2. Reference contexts from the source papers
3. The expected answer (ground truth)
4. Two candidate responses (A and B)

IMPORTANT: The responses are from a small language model (0.5B parameters) being tested on its ability to use retrieved context from scientific papers. Focus on:
- Factual accuracy compared to the expected answer
- Proper use of the provided context with citations
- Completeness of key information
- Avoiding hallucinations or unsupported claims

Do NOT penalize for:
- Brevity (if key points are covered)
- Stylistic differences
- Minor phrasing variations
```

---

## User Prompt Template

```
I need you to compare two responses to a scientific question.

### QUESTION
{question}

### REFERENCE CONTEXTS (from source papers)
{contexts}

### EXPECTED ANSWER (ground truth)
{expected_answer}

### RESPONSE A
{response_a}

### RESPONSE B
{response_b}

---

Please evaluate both responses and provide:

1. **Analysis of Response A**: Key strengths and weaknesses regarding accuracy, completeness, and grounding.

2. **Analysis of Response B**: Key strengths and weaknesses regarding accuracy, completeness, and grounding.

3. **Comparison**: Which response better matches the expected answer and uses the context appropriately?

4. **Verdict**: Choose one:
   - "A" if Response A is clearly better
   - "B" if Response B is clearly better
   - "TIE" if both are roughly equal in quality

Format your final answer as:
```json
{
  "verdict": "A" | "B" | "TIE",
  "confidence": "high" | "medium" | "low",
  "reasoning": "One sentence summary of why"
}
```
```

---

## Batch Evaluation Template (Multiple Examples)

If you want to evaluate multiple pairs in one prompt:

```
I need you to compare pairs of responses to scientific questions. For each pair, determine which response is better.

{%- for example in examples %}

---
## Example {{ loop.index }}

### QUESTION
{{ example.question }}

### EXPECTED ANSWER
{{ example.expected_answer }}

### RESPONSE A
{{ example.response_a }}

### RESPONSE B
{{ example.response_b }}

{%- endfor %}

---

For each example, provide your verdict in this format:

```json
{
  "evaluations": [
    {"example": 1, "verdict": "A|B|TIE", "confidence": "high|medium|low", "reasoning": "..."},
    {"example": 2, "verdict": "A|B|TIE", "confidence": "high|medium|low", "reasoning": "..."},
    ...
  ]
}
```
```

---

## Usage Notes

1. **Position Randomization**: Alternate which model is A vs B across examples to detect position bias.

2. **Blind Evaluation**: Don't tell the judge which model is which (Base+RAG vs FT-RAG+RAG).

3. **Confidence Tracking**: The "confidence" field helps identify ambiguous cases.

4. **Batch Size**: For batch evaluation, 5-10 examples per prompt works well. More may cause context length issues or quality degradation.

---

## Export Script

To generate evaluation data for manual/API evaluation:

```python
import json
from pathlib import Path
from scripts.core.pairwise_tournament import load_responses

results_dir = Path("evaluation/results/interleaved")
model_a = "rag_baseline_0p5b"
model_b = "hybrid_science_0p5b_rag_trained"

responses_a = load_responses(results_dir, model_a)
responses_b = load_responses(results_dir, model_b)

common_ids = sorted(set(responses_a.keys()) & set(responses_b.keys()))

# Export for external evaluation
export = []
for task_id in common_ids:
    rec_a = responses_a[task_id]
    rec_b = responses_b[task_id]
    export.append({
        "task_id": task_id,
        "question": rec_a["instruction"],
        "expected_answer": rec_a["expected_response"],
        "response_a": rec_a["model_answer"],
        "response_b": rec_b["model_answer"],
        "model_a": model_a,
        "model_b": model_b,
    })

with open("evaluation/exports/pairwise_for_external_judge.json", "w") as f:
    json.dump(export, f, indent=2)

print(f"Exported {len(export)} pairs for external evaluation")
```
