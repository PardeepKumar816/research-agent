from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict, List, Annotated
import operator
import json

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

tavily = TavilySearch(max_results=3)
llm_fast = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
llm_smart = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


class ResearchState(TypedDict):
    topic: str
    messages: Annotated[List[BaseMessage], operator.add]
    search_queries: List[str]
    search_results: Annotated[List[str], operator.add]
    new_queries: List[str]
    iteration: int
    report: str
    is_complete: bool
    status_updates: Annotated[List[str], operator.add]


def plan_node(state: ResearchState) -> dict:
    print("NODE: plan")
    response = llm_fast.invoke([HumanMessage(content=f"""
Research topic: {state['topic']}
Generate 3 specific, focused search queries targeting different aspects.
Return ONLY a JSON array: ["query1", "query2", "query3"]""")])

    try:
        queries = json.loads(response.content.strip())
        if not isinstance(queries, list) or len(queries) == 0:
            raise ValueError
    except:
        queries = [
            state['topic'],
            f"{state['topic']} latest developments",
            f"{state['topic']} examples and use cases"
        ]

    print(f"NODE: plan done — {queries}")
    return {
        "search_queries": queries,
        "messages": [response],
        "status_updates": [f"📋 Generated {len(queries)} search queries"]
    }


def search_node(state: ResearchState) -> dict:
    print("NODE: search")
    queries = state.get("new_queries") or state["search_queries"]
    results = []

    for query in queries:
        try:
            raw = tavily.invoke(query)
            items = raw.get("results", raw) if isinstance(raw, dict) else raw
            for r in (items or []):
                if isinstance(r, dict):
                    results.append(
                        f"Source: {r.get('url','')}\n"
                        f"Title: {r.get('title','')}\n"
                        f"Content: {r.get('content','')}\n---"
                    )
        except Exception as e:
            results.append(f"Search failed for '{query}': {e}")

    print(f"NODE: search done — {len(results)} results")
    return {
        "search_results": results,
        "iteration": state["iteration"] + 1,
        "new_queries": [],
        "status_updates": [
            f"🔍 Iteration {state['iteration']+1}: searched {len(queries)} queries, got {len(results)} results"
        ]
    }


def reflect_node(state: ResearchState) -> dict:
    print("NODE: reflect")
    results_text = "\n".join(state["search_results"][-12:])

    response = llm_fast.invoke([HumanMessage(content=f"""
Topic: {state['topic']} | Iteration: {state['iteration']}

Research so far:
{results_text[:3000]}

Be PRACTICAL. Good enough = covers main aspects with recent info.

Return ONLY JSON:
{{
  "is_complete": true or false,
  "reason": "one sentence",
  "follow_up_queries": ["q1", "q2"]
}}

If is_complete is true, follow_up_queries can be [].
follow_up_queries must be DIFFERENT from previous searches.""")])

    try:
        ev = json.loads(response.content.strip())
        is_complete = ev.get("is_complete", False)
        follow_ups = ev.get("follow_up_queries", [])
        reason = ev.get("reason", "")
    except:
        is_complete = state["iteration"] >= 2
        follow_ups = []
        reason = "Defaulted after parse error"

    print(f"NODE: reflect done — is_complete={is_complete}")
    return {
        "is_complete": is_complete,
        "new_queries": follow_ups,
        "messages": [response],
        "status_updates": [f"🤔 Reflection: {'sufficient' if is_complete else 'need more research'} — {reason}"]
    }


def write_report_node(state: ResearchState) -> dict:
    print("NODE: write_report")
    results_text = "\n".join(state["search_results"])

    response = llm_smart.invoke([HumanMessage(content=f"""
Write a comprehensive research report.

Topic: {state['topic']}

Research data:
{results_text[:6000]}

Format exactly like this:
# {state['topic']}

## Executive Summary
(3-4 sentences)

## Key Findings
- finding 1
- finding 2
- finding 3
- finding 4
- finding 5

## Detailed Analysis
(3 paragraphs, factual and specific)

## Current Trends & Outlook
(what's happening now and where it's heading)

## Sources
(list all URLs found in the research)
""")])

    print("NODE: write_report done")
    return {
        "report": response.content,
        "messages": [response],
        "status_updates": ["✅ Report written successfully"]
    }


def should_continue(state: ResearchState) -> str:
    if state["is_complete"] or state["iteration"] >= 2:
        return "write"
    return "search_more"


def build_graph(interrupt_before_write: bool = False):
    print(f"\n>>> build_graph called: interrupt_before_write={interrupt_before_write}\n")

    b = StateGraph(ResearchState)
    b.add_node("plan", plan_node)
    b.add_node("search", search_node)
    b.add_node("reflect", reflect_node)
    b.add_node("write_report", write_report_node)

    b.set_entry_point("plan")
    b.add_edge("plan", "search")
    b.add_edge("search", "reflect")
    b.add_conditional_edges(
        "reflect",
        should_continue,
        {"write": "write_report", "search_more": "search"}
    )
    b.add_edge("write_report", END)

    checkpointer = MemorySaver()

    if interrupt_before_write:
        print(">>> Compiling WITH interrupt_before=['write_report']")
        compiled = b.compile(
            checkpointer=checkpointer,
            interrupt_before=["write_report"]
        )
    else:
        print(">>> Compiling WITHOUT interrupt")
        compiled = b.compile(checkpointer=checkpointer)

    print(f">>> Graph compiled: {type(compiled)}")
    return compiled


def get_initial_state(topic: str) -> dict:
    return {
        "topic": topic,
        "messages": [],
        "search_queries": [],
        "search_results": [],
        "new_queries": [],
        "iteration": 0,
        "report": "",
        "is_complete": False,
        "status_updates": []
    }