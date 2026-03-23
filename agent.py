from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from tools import get_all_tools
from llm_factory import get_coder_llm

REACT_TEMPLATE = """You are a fully local CLI-based coding agent.
You have access to the following code and CLI tools, which you must use to accomplish the user's task:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

CRITICAL RULE: DO NOT combine an Action and a Final Answer in the same response! If you use an Action, STOP WRITING and do not write a Final Answer. Wait for the Observation!

RULES YOU MUST FOLLOW:
- Once you have successfully written a file, DO NOT write it again unless explicitly asked
- After completing all required actions, immediately say: Final Answer: Done.
- Never repeat the same Action + Action Input twice

Conversation History:
{history}

Question: {input}
Thought:{agent_scratchpad}"""

def create_agent():
    tools = get_all_tools()
    
    # Placeholder for MCP tool ingestion
    # Example:
    # mcp_tools = load_mcp_servers(os.getenv("MCP_CONFIG_PATH"))
    # tools.extend(mcp_tools)

    llm = get_coder_llm()
    
    prompt = PromptTemplate(
        template=REACT_TEMPLATE,
        input_variables=["tools", "tool_names", "history", "input", "agent_scratchpad"]
    )
    
    agent = create_react_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors="Check your output format. Output ONLY one of: a Thought/Action/Action Input block OR a Final Answer. Never both at once.",
        max_iterations=6
    )
    
    return agent_executor
