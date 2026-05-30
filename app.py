import streamlit as st
import uuid
from dotenv import load_dotenv
load_dotenv()

from agent import build_graph, get_initial_state

st.set_page_config(
    page_title="ResearchAgent",
    page_icon="🔬",
    layout="wide"
)

st.markdown("""<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>""", unsafe_allow_html=True)


# ── ONE-TIME session state init ───────────────────────────────────────────────
# "initialized" key acts as a sentinel — this block only runs once per session

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

print(f"APP RERUN: phase={st.session_state.phase} | hitl={st.session_state.hitl_mode}")


# ── Sidebar ───────────────────────────────────────────────────────────────────

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
        # only update if changed — prevents unnecessary reruns
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


# ── Main ──────────────────────────────────────────────────────────────────────

st.title("🔬 ResearchAgent")
st.caption("Give it a topic. It searches the web, reflects, and writes a comprehensive report.")


# ── PHASE: IDLE ───────────────────────────────────────────────────────────────

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
            print(f"\n>>> START clicked: topic={topic.strip()} | hitl={st.session_state.hitl_mode}\n")
            st.session_state.topic = topic.strip()
            st.session_state.graph = build_graph(
                interrupt_before_write=st.session_state.hitl_mode
            )
            st.session_state.status_log = []
            st.session_state.phase = "researching"
            print(f">>> Phase set to: {st.session_state.phase}")
            st.rerun()


# ── PHASE: RESEARCHING ────────────────────────────────────────────────────────

elif st.session_state.phase == "researching":
    print(f"\n>>> RESEARCHING phase entered")
    st.markdown(f"### Researching: *{st.session_state.topic}*")

    graph = st.session_state.graph
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    print(f">>> graph={graph}")
    print(f">>> config={config}")

    if graph is None:
        st.error("Graph is None — something went wrong. Click New Research.")
        st.session_state.phase = "idle"
        st.rerun()

    with st.status("🔍 Agent is researching...", expanded=True) as status:
        try:
            print(">>> Starting graph.stream()")

            for update in graph.stream(
                get_initial_state(st.session_state.topic),
                config=config,
                stream_mode="updates"
            ):
                node_name = list(update.keys())[0]
                 # __interrupt__ is LangGraph's signal that the graph paused
                # at an interrupt_before checkpoint. It's a tuple, not a dict.
                # Skip it — we handle the pause via get_state() after the loop.
                if node_name == "__interrupt__":
                    print(">>> Interrupt signal received — graph paused")
                    continue
                node_data = update[node_name]

                print(f">>> Stream update: node={node_name}")

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

            print(">>> graph.stream() finished")

            snapshot = graph.get_state(config)
            next_nodes = snapshot.next
            values = snapshot.values

            print(f">>> next_nodes={next_nodes}")
            print(f">>> report empty={values.get('report','') == ''}")
            print(f">>> hitl={st.session_state.hitl_mode}")

            st.write(f"DEBUG: next_nodes={next_nodes} | hitl={st.session_state.hitl_mode}")

            if next_nodes and "write_report" in next_nodes:
                print(">>> → phase = awaiting_approval")
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
                print(f">>> → phase = done (next_nodes was {next_nodes})")
                status.update(label="✅ Research complete!", state="complete")
                st.session_state.report = values.get("report", "")
                st.session_state.phase = "done"

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f">>> EXCEPTION: {e}")
            print(tb)
            status.update(label=f"❌ Error: {e}", state="error")
            st.error(f"Agent error: {str(e)}")
            st.code(tb)
            st.session_state.phase = "idle"

    st.rerun()


# ── PHASE: AWAITING APPROVAL ──────────────────────────────────────────────────

elif st.session_state.phase == "awaiting_approval":
    print(">>> AWAITING_APPROVAL phase entered")
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
        print(">>> APPROVE clicked")
        with st.spinner("✍️ Writing report..."):
            try:
                print(">>> Resuming graph with stream(None)")
                final_result = None

                for update in graph.stream(
                    None,
                    config=config,
                    stream_mode="updates"
                ):
                    node_name = list(update.keys())[0]
                    node_data = update[node_name]
                    print(f">>> Resume stream: node={node_name}")
                    if node_name == "write_report":
                        final_result = node_data

                print(f">>> final_result={final_result is not None}")

                if final_result and final_result.get("report"):
                    st.session_state.report = final_result["report"]
                    print(">>> Report from stream")
                else:
                    final_state = graph.get_state(config)
                    st.session_state.report = final_state.values.get("report", "No report generated")
                    print(f">>> Report from get_state: len={len(st.session_state.report)}")

                st.session_state.phase = "done"
                st.rerun()

            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print(f">>> APPROVE ERROR: {e}")
                print(tb)
                st.error(f"Error writing report: {str(e)}")
                st.code(tb)

    if restart:
        st.session_state.graph = build_graph(interrupt_before_write=True)
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.phase = "researching"
        st.rerun()


# ── PHASE: DONE ───────────────────────────────────────────────────────────────

elif st.session_state.phase == "done":
    print(">>> DONE phase entered")
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