from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat

from agents.deep_research.tools.bing_search import bing_search_tool
from agents.deep_research.tools.fetch_webpage import fetch_webpage_tool
from config import get_model_client
from datetime import datetime

model_client = get_model_client()

MAX_MESSAGES  = 50

PROMPT_RESERACH = """You are a research assistant focused on finding accurate information.
The **TIME NOW** is {{time_now}}
Use the bing_search tool to find relevant information.
Break down complex queries into specific search terms.
Always verify information across multiple sources when possible.
When you find relevant information, explain why it's relevant and how it connects to the query. When you get feedback from the a verifier agent, use your tools to act on the feedback and make progress.
""".replace("{{time_now}}", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

PROMPT_VERIFIER = """You are a research verification specialist.
Your role is to:
1. Verify that search queries are effective and suggest improvements if needed
2. Explore drill downs where needed e.g, if the answer is likely in a link in the returned search results, suggest clicking on the link
3. Suggest additional angles or perspectives to explore. Be judicious in suggesting new paths to avoid scope creep or wasting resources, if the task appears to be addressed and we can provide a report, do this and respond with "TERMINATE".
4. Track progress toward answering the original question
5. When the research is complete, provide a detailed summary in markdown format. For incomplete research, end your message with "CONTINUE RESEARCH". For complete research, end your message with APPROVED.

Your responses should be structured as:
- Progress Assessment
- Gaps/Issues (if any)
- Suggestions (if needed)
- Next Steps (if research is incomplete), or Final Summary (if research is complete)
- CONTINUE RESEARCH or APPROVED
"""

PROMPT_SUMMARY = """You are a summary agent. Your role is to provide a detailed markdown summary of the research as a report to the user. Your report should have a reasonable title that matches the research question and should summarize the key details in the results found in natural an actionable manner. The main results/answer should be in the first paragraph. Where reasonable, your report should have clear comparison tables that drive critical insights. Most importantly, you should have a reference section and cite the key sources (where available) for facts obtained INSIDE THE MAIN REPORT. Also, where appropriate, you may add images if available that illustrate concepts needed for the summary.
Your report should end with the word "TERMINATE" to signal the end of the conversation.
"""


PROMPT_SELECTOR = """
You are coordinating a research team by selecting the team member to speak/act next. The following team member roles are available:
{roles}.
The research_assistant performs searches and analyzes information.
The verifier evaluates progress and ensures completeness.
The summary_agent provides a detailed markdown summary of the research as a report to the user, only when research result is APPROVED.

Given the current context, select the most appropriate next speaker.
The research_assistant should search and analyze.
The verifier should evaluate progress and guide the research (select this role is there is a need to verify/evaluate progress). 
You should ONLY select the summary_agent role if the research result is APPROVED by verifier.

Base your selection on:
1. Current stage of research
2. Last speaker's findings or suggestions
3. Need for verification vs need for new information
Read the following conversation. Then select the next role from {participants} to play. Only return the role.

{history}

Read the above conversation. Then select the next role from {participants} to play. ONLY RETURN THE ROLE.
"""

text_mention_termination = TextMentionTermination("TERMINATE")
max_messages_termination = MaxMessageTermination(max_messages=MAX_MESSAGES)
termination = text_mention_termination | max_messages_termination

def create_team()->SelectorGroupChat:
    research_assistant = AssistantAgent(
        "research_assistant",
        description="An agent that provides assistance with tool use.",
        model_client=model_client,
        model_client_stream=True,
        system_message=PROMPT_RESERACH,
        tools=[fetch_webpage_tool, bing_search_tool])

    verifier = AssistantAgent(
        "verifier",
        description="An agent that provides assistance with tool use.",
        model_client=model_client,
        model_client_stream=True,
        system_message=PROMPT_VERIFIER)

    summary_agent = AssistantAgent(
        name="summary_agent",
        description="Generate a report based on planning and data analysis and code execution results.",
        model_client=model_client,
        model_client_stream=True,
        system_message=PROMPT_SUMMARY)
    
    return SelectorGroupChat(
        [research_assistant, verifier, summary_agent],
        termination_condition=termination,
        model_client=model_client,
        selector_prompt=PROMPT_SELECTOR,
        allow_repeated_speaker=True)
