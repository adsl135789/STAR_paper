T2Q_PROMPT = """
You are an expert at analyzing tables and generating diverse, natural questions that could be answered using the table data.

### Input Table:
{text}

### Your Task:
Generate **5 diverse questions** that cover different query patterns. The questions should be based on the actual content and structure of this table, not just generic column-based templates.

### Question Types to Cover (Generate 1 questions for each type):
- Numerical: "What is the average 'Sales' in 'Region' X?"
- List: "List all 'Products' with 'Price' above 100"
- Count: "How many 'Orders' have 'Status' = shipped?"
- Select: "Which 'Employee' has the highest 'Revenue'?"

### Important Requirements:
- Use **natural, conversational language** - avoid templated patterns like "What is the average..." for every question
- Make questions **specific to the actual content** in the table, not just column names
- Reference real values, names, or entities that appear in the table when possible
- Vary the question structure and wording
- For fact-verification style tables, focus on entity-specific and temporal questions
- For reasoning-oriented tables, include multi-step or conditional questions
- **Language code: {lang}** - Generate all questions in this language

### Output Format (JSON only):
```json
{{
    "questions": ["question1", "question2", "question3", "question4", "question5"]
}}
```

Generate the 5 questions now:
"""

T2Q_PROMPT_KMEANS = """
You are an expert at analyzing tables and generating diverse, natural questions that could be answered using the table data.

### Input Table:
{text}

### Your Task:
Generate **2 diverse questions** that cover different query patterns. The questions should be based on the actual content and structure of this table, not just generic column-based templates.

### Question Types to Cover:
- Numerical: "What is the average 'Sales' in 'Region' X?"
- List: "List all 'Products' with 'Price' above 100"
- Count: "How many 'Orders' have 'Status' = shipped?"
- Select: "Which 'Employee' has the highest 'Revenue'?"

### Important Requirements:
- Use **natural, conversational language** - avoid templated patterns like "What is the average..." for every question
- Make questions **specific to the actual content** in the table, not just column names
- Reference real values, names, or entities that appear in the table when possible
- Vary the question structure and wording
- For fact-verification style tables, focus on entity-specific and temporal questions
- For reasoning-oriented tables, include multi-step or conditional questions
- **Language code: {lang}** - Generate all questions in this language

### Output Format (JSON only):
```json
{{
    "questions": ["question1", "question2"]
}}
```

Generate the 2 questions now:
"""


Q2T_PROMPT = """
Given a natural language question, generate a minimal CSV table (header and one instance row) that can answer the question.

### Rules:
1. The output must include exactly 1 header row and exactly 1 instance row
2. Extract 1-3 column names based on the question intent
3. Use concise but meaningful column headers
4. The instance values must logically satisfy the question
5. The output must be in the same language as the input question

### Output Format (JSON only):
```json
{{
    "header": ["Col1","Col2",...],
    "instances": ["Val1","Val2",...]
}}
```
"""