from langchain_core.prompts import PromptTemplate
from llm_factory import get_refiner_llm

def refine_prompt(user_input: str) -> str:
    """Refines a messy user prompt into a clear statement of intent."""
    if len(user_input.strip()) < 10:
        return user_input
        
    llm = get_refiner_llm()
    
    template = """You are a prompt refiner for an advanced local coding agent named Codi. 
The user might type messy, typo-filled, or overly brief instructions.
Rewrite their input into a highly detailed, clear, and actionable instruction for the coding agent.
Ensure you capture the complete intent of the user. Do exactly what is asked. 
Embellish the prompt so the coding agent fully implements every feature without cutting any corners or skipping logic.
Do NOT answer the prompt or add pleasantries. Just return the refined prompt.

Original prompt: {user_input}
Refined prompt:"""
    
    prompt = PromptTemplate(template=template, input_variables=["user_input"])
    chain = prompt | llm
    
    try:
        response = chain.invoke({"user_input": user_input})
        return response.content.strip() if hasattr(response, 'content') else str(response).strip()
    except Exception as e:
        print(f"Error refining prompt: {e}")
        return user_input
