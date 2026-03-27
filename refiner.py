from langchain_core.prompts import PromptTemplate
from llm_factory import get_refiner_llm

# Keywords that indicate the input actually needs refining
REFINE_TRIGGERS = (
    "create", "write", "make", "build", "fix", "edit",
    "update", "generate", "refactor", "implement", "add"
)

def refine_prompt(user_input: str) -> str:
    """
    Refines user prompt only when it's worth the token cost.
    Short inputs, greetings, and questions are returned as-is.
    Only action-oriented prompts (create, build, fix...) get refined.
    """
    text = user_input.strip()

    # Skip refining if too short
    if len(text) < 50:
        return text

    # Skip refining if no action keywords — it's probably a question
    if not any(trigger in text.lower() for trigger in REFINE_TRIGGERS):
        return text

    llm = get_refiner_llm()

    template = """Rewrite this coding task as a clear 1-2 sentence instruction for an AI agent.
No bullet points. No headers. No extra context. Just the core instruction.

Task: {user_input}
Instruction:"""

    prompt = PromptTemplate(template=template, input_variables=["user_input"])
    chain = prompt | llm

    try:
        response = chain.invoke({"user_input": text})
        refined = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        # Safety — if refiner bloats the prompt, discard and use original
        if len(refined) > len(text) * 2 or len(refined) < 10:
            return text
        return refined
    except Exception as e:
        print(f"Refiner error (using original): {e}")
        return text