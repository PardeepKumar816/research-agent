import streamlit as st
import uuid
from dotenv import load_dotenv
load_dotenv()

from agent import build_graph, get_initial_state

st.set_page_config(
    page_title="ResearchAgent",
    page_icon="🔬",
   # layout="wide"
)

# st.markdown("""<style>
# #MainMenu {visibility: hidden;}
# footer {visibility: hidden;}
# header {visibility: hidden;}
# </style>""", unsafe_allow_html=True)

st.markdown("""
<style>
    [data-testid="stHeader"] { visibility: hidden; }
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.graph = None
    st.session_state.report = ""
    st.session_state.status_log = []
    st.session_state.phase = "idle"
    st.session_state.research_summary = {}
    st.session_state.topic = ""
    st.session_state.hitl_mode = True

with st.sidebar:
    st.title("🔬 ResearchAgent")
    st.caption("Autonomous AI researcher powered by LangGraph + Groq")
    st.divider()

    if st.session_state.phase == "idle":
        new_hitl = st.toggle(
            "Human approval before report",
            value=st.session_state.hitl_mode,
            help="When ON, agent pauses before writing and asks you to approve."
        )
        if new_hitl != st.session_state.hitl_mode:
            st.session_state.hitl_mode = new_hitl
    else:
        label = "✅ Human approval ON" if st.session_state.hitl_mode else "⚡ Autonomous mode"
        st.info(label)

    st.divider()
    st.caption("**How it works:**")
    st.caption("1. Breaks topic into search queries")
    st.caption("2. Searches the web via Tavily")
    st.caption("3. Reflects — enough info?")
    st.caption("4. Repeats if needed (max 2x)")
    st.caption("5. Writes structured report")
    st.divider()

    if st.button("🔄 New Research", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.graph = None
        st.session_state.report = ""
        st.session_state.status_log = []
        st.session_state.phase = "idle"
        st.session_state.research_summary = {}
        st.session_state.topic = ""
        st.rerun()

    st.divider()
    st.caption("Built with LangGraph + Groq + Tavily")

st.title("🔬 ResearchAgent")
st.caption("Give it a topic. It searches the web, reflects, and writes a comprehensive report.")

if st.session_state.phase == "idle":
    st.markdown("### What do you want to research?")

    topic = st.text_input(
        "Research topic",
        placeholder="e.g. LangGraph for building production AI agents",
        label_visibility="collapsed"
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption("Examples: `Agentic AI 2025` · `LangGraph vs AutoGen` · `RAG vs fine-tuning`")
    with col2:
        start = st.button("🚀 Research", use_container_width=True, type="primary")

    if start:
        if not topic.strip():
            st.warning("Please enter a research topic.")
        else:
            st.session_state.topic = topic.strip()
            st.session_state.graph = build_graph(
                interrupt_before_write=st.session_state.hitl_mode
            )
            st.session_state.status_log = []
            st.session_state.phase = "researching"
            st.rerun()

elif st.session_state.phase == "researching":
    st.markdown(f"### Researching: *{st.session_state.topic}*")

    graph = st.session_state.graph
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    if graph is None:
        st.error("Graph is None. Click New Research.")
        st.session_state.phase = "idle"
        st.rerun()

    with st.status("🔍 Agent is researching...", expanded=True) as status:
        try:
            for update in graph.stream(
                get_initial_state(st.session_state.topic),
                config=config,
                stream_mode="updates"
            ):
                node_name = list(update.keys())[0]

                if node_name == "__interrupt__":
                    continue

                node_data = update[node_name]

                for msg in node_data.get("status_updates", []):
                    st.write(msg)
                    st.session_state.status_log.append(msg)

                if node_name == "plan":
                    queries = node_data.get("search_queries", [])
                    if queries:
                        st.write(f"**Queries:** {', '.join(queries)}")

                elif node_name == "reflect":
                    new_q = node_data.get("new_queries", [])
                    is_done = node_data.get("is_complete", False)
                    if new_q and not is_done:
                        st.write(f"**Follow-up:** {', '.join(new_q)}")

            snapshot = graph.get_state(config)
            next_nodes = snapshot.next
            values = snapshot.values

            if next_nodes and "write_report" in next_nodes:
                status.update(
                    label="⏸️ Research done — awaiting your approval",
                    state="complete"
                )
                st.session_state.research_summary = {
                    "iterations": values.get("iteration", 0),
                    "results_count": len(values.get("search_results", [])),
                    "queries": values.get("search_queries", []),
                    "sample_results": values.get("search_results", [])[:3]
                }
                st.session_state.phase = "awaiting_approval"
            else:
                status.update(label="✅ Research complete!", state="complete")
                st.session_state.report = values.get("report", "")
                st.session_state.phase = "done"

        except Exception as e:
            status.update(label=f"❌ Error: {e}", state="error")
            st.error(f"Agent error: {str(e)}")
            st.session_state.phase = "idle"

    st.rerun()

elif st.session_state.phase == "awaiting_approval":
    st.markdown(f"### Research complete: *{st.session_state.topic}*")
    st.info("Review the research below, then approve to generate the report.")

    summary = st.session_state.research_summary
    graph = st.session_state.graph
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Iterations", summary.get("iterations", 0))
    with col2:
        st.metric("Results collected", summary.get("results_count", 0))

    with st.expander("📄 Preview research", expanded=True):
        for i, r in enumerate(summary.get("sample_results", [])):
            st.caption(f"Result {i+1}")
            st.text(r[:400])
            st.divider()

    with st.expander("🔍 Queries used"):
        for q in summary.get("queries", []):
            st.write(f"• {q}")

    st.markdown("### Ready to write the report?")

    approve = st.button(
        "✅ Approve — Write Report",
        type="primary",
        use_container_width=True
    )
    restart = st.button("🔄 Research More", use_container_width=True)

    if approve:
        with st.spinner("✍️ Writing report..."):
            try:
                final_result = None
                for update in graph.stream(
                    None,
                    config=config,
                    stream_mode="updates"
                ):
                    node_name = list(update.keys())[0]
                    node_data = update[node_name]
                    if node_name == "write_report":
                        final_result = node_data

                if final_result and final_result.get("report"):
                    st.session_state.report = final_result["report"]
                else:
                    final_state = graph.get_state(config)
                    st.session_state.report = final_state.values.get("report", "No report generated")

                st.session_state.phase = "done"
                st.rerun()

            except Exception as e:
                st.error(f"Error writing report: {str(e)}")

    if restart:
        st.session_state.graph = build_graph(interrupt_before_write=True)
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.phase = "researching"
        st.rerun()

elif st.session_state.phase == "done":
    st.markdown(f"### Report: *{st.session_state.topic}*")

    st.download_button(
        label="⬇️ Download (.md)",
        data=st.session_state.report,
        file_name=f"research_{st.session_state.topic[:30].replace(' ','_')}.md",
        mime="text/markdown"
    )

    st.divider()
    st.markdown(st.session_state.report)
    st.divider()

    with st.expander("🔎 Agent activity log"):
        for entry in st.session_state.status_log:
            st.write(entry)

    if st.button("🔬 Research another topic", type="primary"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.graph = None
        st.session_state.report = ""
        st.session_state.status_log = []
        st.session_state.phase = "idle"
        st.session_state.research_summary = {}
        st.session_state.topic = ""
        st.rerun()