"""LangGraph orchestration; all scientific calculations use deterministic Python tools."""
from typing import Any, TypedDict

from .agent import (critique_and_refine, execute_experiment, inspect_and_plan,
                    produce_report, run_investigation)


class ResearchState(TypedDict, total=False):
    question: str
    data: Any
    plan: dict[str, Any]
    quality: dict[str, Any]
    result: dict[str, Any]
    critique: dict[str, Any]
    history: list[dict[str, Any]]
    error: str


def build_graph():
    try:
        from langgraph.graph import END, StateGraph
    except ImportError:
        return None
    graph = StateGraph(ResearchState)

    graph.add_node("inspect_and_plan", lambda state: inspect_and_plan(state["question"]))
    graph.add_node("execute_experiment", execute_experiment)
    graph.add_node("critique_and_refine", critique_and_refine)
    graph.add_node("produce_report", lambda state: {"result": produce_report(state)})
    graph.set_entry_point("inspect_and_plan")
    graph.add_edge("inspect_and_plan", "execute_experiment")
    graph.add_edge("execute_experiment", "critique_and_refine")
    graph.add_edge("critique_and_refine", "produce_report")
    graph.add_edge("produce_report", END)
    return graph.compile()


def invoke(question: str) -> dict:
    graph = build_graph()
    if graph is None:
        return {"result": run_investigation(question)}
    try:
        return graph.invoke({"question": question})
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
