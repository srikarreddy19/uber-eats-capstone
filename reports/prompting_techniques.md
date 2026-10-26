# Prompting Techniques — Uber Eats Intelligence Project

## 1. P.T.C.F. Prompting

P.T.C.F. stands for Persona, Task, Context, and Format.

**Example prompt:**

**Persona:** Act as a data analyst.

**Task:** Identify data quality issues in the orders dataset.

**Context:** The dataset contains order IDs, customer IDs, order amounts, timestamps, and order statuses.

**Format:** Return a table with the issue, affected column, and recommended action.

## 2. Few-Shot Prompting

Few-shot prompting provides examples to guide the expected output.

**Example prompt:**

Classify customer reviews as Positive, Neutral, or Negative.

Examples:
- "Great food and quick delivery!" → Positive
- "The food was okay." → Neutral
- "Food arrived late and cold." → Negative

Now classify: "The delivery took too long, and the food was cold."

Expected output: Negative.

## 3. Role-Based Prompting

Role-based prompting assigns a specific professional role to guide the response.

**Example prompt:**

Act as a senior business intelligence analyst. Analyze delivery times across traffic conditions and weather types. Identify the operational factors associated with longer deliveries and recommend practical improvements.

## 4. Structured Prompting

Structured prompting specifies the required output format.

**Example prompt:**

Analyze customer segments and return the results in this format:

- Segment name
- Typical customer characteristics
- Average order amount
- Order frequency
- Business recommendation

Use clear headings and concise explanations.

## 5. Responsible Use

- Validate generated code before running it.
- Check analytical results against the actual dataset.
- Avoid sharing credentials or confidential data in prompts.
- Treat AI-generated recommendations as suggestions that require verification.
