import os
import time
from datetime import datetime

import chainlit as cl
from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import ModelClientStreamingChunkEvent, TextMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_core import CancellationToken

from agents.deep_research.main import create_team
from config import DEEP_RESEARCH_AGENT


@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name=DEEP_RESEARCH_AGENT,
            markdown_description="Thorough exploration to gain deep understanding.",
            icon="public/icons/deep_research.png",
            starters=[
                cl.Starter(
                    label="DeepSeek-R1 training methodology",
                    message="DeepSeek-R1 training methodology",
                ),
                cl.Starter(
                    label="Electric vehicle market",
                    message="Analysis of China's electric vehicle market in 2024",
                ),
            ],
        ),
    ]

@cl.on_chat_start
async def on_chat_start():
    deep_research_team = create_team()
    cl.user_session.set(DEEP_RESEARCH_AGENT, deep_research_team)

@cl.on_message  # type: ignore
async def chat(message: cl.Message) -> None:
    deep_research_team = cl.user_session.get(DEEP_RESEARCH_AGENT)
    await run_stream_team(
        deep_research_team,
        message,
    )

async def run_stream_team(team=SelectorGroupChat, message: cl.Message | None = None):
    executing = False

    async with cl.Step(name="Executing") as executing_step:
        start = time.time()

        final_answer = cl.Message(content="")

        async for msg in team.run_stream(
            task=[TextMessage(content=message.content, source="user")],
            cancellation_token=CancellationToken(),
        ):
            if isinstance(msg, ModelClientStreamingChunkEvent):
                if msg.source != "summary_agent":
                    executing = True
                else:
                    executing = False
                    executed_for = round(time.time() - start)
                    executing_step.name = f"Executed for {executed_for}s"
                    await executing_step.update()

                if executing:
                    await executing_step.stream_token(msg.content)
                else:
                    await final_answer.stream_token(msg.content)
            elif executing_step is not None:
                await executing_step.send()
            elif isinstance(msg, TaskResult):
                pass
            else:
                pass
    file = md_to_pdf(final_answer.content)
    await final_answer.stream_token(f"\n\nPDF: [{os.path.basename(file)}]({file})")
    await final_answer.send()


def md_to_pdf(md: str) -> bytes:
    from markdown_pdf import MarkdownPdf, Section

    os.makedirs("public/pdfs", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"public/pdfs/deepresearch_{timestamp}.pdf"

    pdf = MarkdownPdf()
    pdf.meta["title"] = "Title"
    pdf.add_section(
        Section(
            md.rsplit("TERMINATE", 1)[0].rstrip(),
            toc=False,
        )
    )
    pdf.save(filename)

    return filename
