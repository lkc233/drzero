THINK_PATTERN = r"^\s*<think>(.*?)</think>"
QUESTION_PATTERN = r"<question>(.*?)</question>"
ANSWER_PATTERN = r"<answer>(.*?)</answer>"

USER_PATTERN = r"<\|im_start\|>user\n(.*?)<\|im_end\|>"
TOOL_CALL_PATTERN = r"<tool_call>(.*?)</tool_call>"
ASSISTANT_PATTERN = r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>"

SOURCE_PATTERN = r"source document: (.*?)<\|im_end\|>"
TOOL_RESPONSE_PATTERN = r"<\|im_start\|>user\n<tool_response>(.*?)</tool_response><\|im_end\|>"


TOOL_CALL_EXAMPLE = (
    '<tool_call> {"name": "search", "arguments": {"query_list": ["QUERY"]}} </tool_call>'
)


DEFAULT_CHALLENGER_PREFIX = """
You are an expert in question generation. Craft one challenging, deterministic question and its single,
unambiguous answer based on the provided source document. The logical path must start from the document
and require exactly n hops (i.e., n-1 searches) to reach the final answer.

### Definitions
1. Hop: A node in the reasoning chain. Hop 1 is the starting entity found in the document. Hop n is the final answer.

### Inputs
1. n: the exact number of hops in the reasoning chain (requiring n-1 searches).
2. Source document: the full source text.

### Process & Tools
1. Analyze the Document and Select the Starting Point
  - Read and analyze the source document.
  - Select a specific entity, event or detail explicitly mentioned in the text.
    This entity becomes Hop 1 (the initial clue).
2. Design the Chain Forwards
  - From Hop 1 to Hop 2: Identify a factual attribute or relation of Hop 1 that is NOT in the text
    but can be found via search. The result is Hop 2.
  - Iterate: Continue connecting the current Hop i to the next Hop i+1 using a deterministic,
    verifiable relation found via search.
  - Stop at Hop n: Continue this process until you have exactly n hops. Hop n must be a single, canonical final answer.
3. Reasoning & Search Protocol
  - Always reason inside `<think> ... </think>` when you plan connections or receive new information.
  - For each hop transition that requires external information, issue search query using `<tool_call> ... </tool_call>`.
  - Search results will be provided between `<tool_response> ... </tool_response>` by the system.
4. Output Format
  - Emit a numbered sequence of EXACTLY n-1 search steps. For each search i (1 to n-1), produce:
    `<think> Reasoning step i: Identify Hop i in document/search results, formulate query to reach Hop i+1 </think>`
    `<tool_call> Query to search Hop i+1 </tool_call>`
    `[Wait for search results in <tool_response> from system]`
  - After completing all searches and arriving at Hop n, output the question and final answer:
    `<think> Final reasoning step: Confirm the chain is complete with Hop n and formulate the question </think>`
    `<question> A challenging question that provides Hop 1 (the initial clue)
    and asks for the final answer (Hop n) </question>`
    `<answer> The single, concise final answer (Hop n) </answer>`

### Examples
1. Example template for Hop n = 1, i.e. no search:
  `<think> [Explain how Hop 1 is selected from the source document and how the question is formulated] </think>`
  `<question> [Question based solely on the text entity Hop 1] </question>`
  `<answer> [Answer (Hop 1)] </answer>`
2. Example template for Hop n = 3, i.e. 2 searches:
  `<think> [Reasoning step 1: Find Hop 1 in the source document, formulate the query to reach Hop 2] </think>`
  `<tool_call> [Search query to find Hop 2 based on Hop 1] </tool_call>`
  `[Wait for search results in <tool_response> from system]`
  `<think> [Reasoning step 2: Reason on search results to identify Hop 2
  and write the next query to find Hop 3] </think>`
  `<tool_call> [Search query to find Hop 3 based on Hop 2] </tool_call>`
  `[Wait for search results in <tool_response> from system]`
  `<think> [Final reasoning step: Confirm Hop 3 in search results
  and formulate the question starting from Hop 1] </think>`
  `<question> [Question starting with Hop 1, requiring the solver to find Hop 2
  to eventually reach the Answer (Hop 3)] </question>`
  `<answer> [Answer (Hop 3)] </answer>`

### Critical Rules
1. Start in Document: Hop 1 must be explicitly present in the source text.
   Every subsequent hop must be supported by the corresponding search results.
2. Search is mandatory for n > 1: Each link between hops beyond Hop 1 must use the search engine.
3. Exact search count: Emit exactly (n-1) `<tool_call>` entries, no more, no fewer.
4. No spoilers: The question must mention only Hop 1; do not include or hint at intermediate hops.
5. Clarity: The question is self-contained; the answer is concise and direct
   (no extra commentary, formatting or explanation).
6. Chain integrity: Each hop must depend strictly on the previous hop.
   No hop should be skippable or derivable without its immediate predecessor.

Now, generate a question and its answer with n = {hops} hops starting from the following source document: {document}
"""


DEFAULT_SOLVER_PREFIX = (
  "Answer the given question. You must conduct reasoning inside <think> and </think> "
  "first every time you get new information. After reasoning, if you find you lack "
  "some knowledge, you can call a search engine by <tool_call> query </tool_call> "
  "and it will return the top searched results between <tool_response> and "
  "</tool_response>. You can search as many times as your want. If you find no "
  "further external knowledge needed, you can directly provide the answer inside "
  "<answer> and </answer>, without detailed illustrations. For example, "
  "<answer> Beijing </answer>. Question: {question}"
)


def _as_mapping(item):
    if hasattr(item, "model_dump"):
        return item.model_dump()
    return item


def serialize_skills(skills):
    """Serialize the complete active skill list in stable list order."""
    rows = []
    for index, skill in enumerate(skills, start=1):
        skill = _as_mapping(skill)
        evidence = skill.get("evidence", "")
        suffix = f" Evidence: {evidence}" if evidence else ""
        rows.append(f"{index}. [{skill['id']}] {skill['instruction']}{suffix}")
    return "\n".join(rows)


def build_challenger_prompt(*, hops, document, skills):
    """Build the one canonical proposer prompt used by training and generation."""
    base_prompt = DEFAULT_CHALLENGER_PREFIX.format(hops=hops, document=document)
    skill_prompt = serialize_skills(skills)
    if not skill_prompt:
        raise ValueError("active skills must not be empty")
    marker = "\nNow, generate a question"
    skills_section = (
        "\n### Active Question-Generation Skills\n"
        "Follow every instruction below. These are frozen for this iteration.\n"
        f"{skill_prompt}\n"
    )
    return base_prompt.replace(marker, skills_section + marker)


VERIFIER_PROMPT = """You are answering a generated question using only the supplied evidence bundle.
External search and tools are forbidden. Reason from the evidence, then return one concise answer in
<answer>...</answer>.

Evidence bundle:
{evidence_bundle}

Question: {question}
"""


QUESTION_ONLY_VERIFIER_PROMPT = """Answer the generated question without using external search or tools.
Return one concise answer in <answer>...</answer>.

Question: {question}
"""


RUBRIC_EVALUATION_PROMPT = """Evaluate one generated multi-hop problem against every active rubric.
Use the seed document and the complete proposer trajectory. Return JSON only with this shape:
{{"evaluations":[{{"rubric_id":"...","score":1,"reason":"..."}}]}}.
Scores must be integers from 1 through 5 and every rubric id must appear exactly once.
Keep every reason under 40 words. Do not include analysis outside the JSON object.

Rubrics:
{rubrics}

Seed document:
{source_document}

Proposer trajectory:
{trajectory}

Question: {question}
Reference answer: {reference_answer}
"""


ANSWER_JUDGE_PROMPT = """Determine whether the model answer is semantically equivalent to the reference answer.
Return JSON only: {{"semantically_equivalent":true,"reason":"..."}}.

Question: {question}
Reference answer: {reference_answer}
Model answer: {model_answer}
"""


TRAJECTORY_ANALYSIS_PROMPT = """Analyze this complete keepout trajectory. Cover successful as well as failed
behavior. Return JSON only with candidate_id, correct, outcome_stage, root_causes, related_rubric_ids,
evidence_quotes, and actionable_improvements. Quotes must be traceable to the trajectory.

Active rubrics:
{rubrics}

Record:
{record}
"""


GLOBAL_ANALYSIS_PROMPT = """Aggregate the supplied trajectory summaries without dropping frequency,
success/failure patterns, related rubric ids, representative cases, or actionable improvements.
Return JSON only with problem_frequencies (string-to-integer map), success_patterns, failure_patterns,
related_rubric_ids, representative_cases, and actionable_improvements. Do not infer examples not present
in the summaries.

Summaries:
{summaries}
"""


SKILLS_UPDATE_PROMPT = """Update the active proposer skills from the complete iteration evidence.
Return JSON only as {{"skills":[...],"decisions":[...]}} and return the complete replacement list, not a patch.
Each skill has id, instruction, and evidence. Explain retention, modification, removal, or addition
with exactly one decision per id appearing before or after the update. Each decision has id, action
(added/retained/modified/removed), reason, and non-empty evidence_refs. evidence_refs must be JSON
Pointers rooted at /rubric_evidence, /verify_evidence, /keepout_evidence, /current_skills, or
/current_rubrics. Cite supplied evidence in both
the skill evidence field and decisions. Use at most 12 skills.

Current skills:
{skills}
Current rubrics and evaluations:
{rubric_evidence}
Verify evidence:
{verify_evidence}
Keepout and trajectory analysis:
{keepout_evidence}
"""


RUBRICS_UPDATE_PROMPT = """Update the active problem-quality rubrics from the complete iteration evidence.
Return JSON only as {{"rubrics":[...],"decisions":[...]}} and return the complete replacement list, not a patch.
Each rubric has id, name, description, score_1_anchor, score_3_anchor, and score_5_anchor.
Include exactly one evidence-backed decision per id appearing before or after the update. Each decision
has id, action (added/retained/modified/removed), reason, and non-empty evidence_refs. evidence_refs must
be JSON Pointers rooted at /rubric_evidence, /verify_evidence, /keepout_evidence, /current_rubrics,
or /next_skills. Use at most 12
rubrics. The skills below are already updated for the next iteration and must be used.

Current rubrics:
{rubrics}
Updated next-iteration skills:
{skills}
Rubric evaluations:
{rubric_evidence}
Verify evidence:
{verify_evidence}
Keepout and trajectory analysis:
{keepout_evidence}
"""



if __name__ == "__main__":
    print(DEFAULT_CHALLENGER_PREFIX)
    print("*" * 100)
    print(DEFAULT_SOLVER_PREFIX)
