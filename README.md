# ResearchAgent 🔬
### Autonomous AI research agent with human-in-the-loop — powered by LangGraph + Groq

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://research-agent-free.streamlit.app/)


## What it does
Give it any research topic. The agent autonomously:
1. Breaks the topic into focused search queries
2. Searches the web via Tavily
3. Reflects — do I have enough information?
4. Searches again with better queries if needed
5. Pauses for human approval (optional)
6. Writes a structured research report with sources

## Architecture

```
[plan] → [search] → [reflect] → [write_report] → END
                        ↑____________↓
                    (if needs more research)
                    
                    ⏸️ interrupt_before=[write_report]
                    (human-in-the-loop mode)
```

## Tech Stack
| Layer | Technology |
|---|---|
| Agent framework | LangGraph (stateful graph + checkpointing) |
| LLM | Llama 3.3 70B + Llama 3.1 8B via Groq (free) |
| Web search | Tavily Search API (free tier) |
| Observability | LangSmith |
| UI | Streamlit with live status streaming |

## Key features
- 🔁 ReAct loop — Reason, Act, Observe, repeat
- ⏸️ Human-in-the-loop — agent pauses before writing, waits for approval
- 💾 Checkpointing — full state preserved at pause point via MemorySaver
- 📡 Live streaming — watch each agent step as it happens
- ⬇️ Download report as Markdown
- 🆓 100% free to run

## Run locally

```bash
git clone https://github.com/PardeepKumar816/research-agent.git
cd research-agent

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Add your API keys to .env

streamlit run app.py
```

## Environment variables

```
GROQ_API_KEY=your_key          # console.groq.com (free)
LANGCHAIN_API_KEY=your_key     # smith.langchain.com (free)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=research-agent
TAVILY_API_KEY=your_key        # tavily.com (free tier)
```

## What I learned
The hardest bug: LangGraph emits a special `__interrupt__` event into the stream when hitting a checkpoint. Most tutorials don't mention this. When `interrupt_before=["write_report"]` is set, you must explicitly skip this event in your stream loop or it crashes. Debugging this through terminal traces taught me more about LangGraph internals than any documentation.

## Built by
[Pardeep Kumar](https://linkedin.com/in/pardeep-kumar-a257221a1) — Flutter & AI Engineer  
[GitHub](https://github.com/PardeepKumar816) | [LinkedIn](https://linkedin.com/in/pardeep-kumar-a257221a1)
