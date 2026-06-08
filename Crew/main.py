"""
=============================================================
  RESERVOIR WELL PRODUCTION ANALYZER — CrewAI Project
  Course: Agentic AI Design
  Domain: Petroleum / Reservoir Engineering
=============================================================

OVERVIEW:
  Three AI agents collaborate to analyze a well's production data:
    1. Data Scout      → Reads well data and summarizes reservoir context
    2. Reservoir Analyst → Calculates PI, Recovery Factor, Decline Rate
    3. Report Writer   → Produces a concise engineering recommendation report

HOW TO RUN:
  1. pip install -r requirements.txt
  2. Copy .env.example to .env and fill in your API keys
  3. python main.py
"""

import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import FileReadTool, SerperDevTool
from langchain_openai import ChatOpenAI

# ─── Load Environment Variables ───────────────────────────────────────────────
load_dotenv()

OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY    = os.getenv("SERPER_API_KEY")    # for web search
LANGFUSE_PUB_KEY  = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SEC_KEY  = os.getenv("LANGFUSE_SECRET_KEY")

# ─── Langfuse Observability (optional but recommended) ─────────────────────────
callbacks = []
try:
    if LANGFUSE_PUB_KEY and LANGFUSE_SEC_KEY:
        from langfuse.callback import CallbackHandler
        langfuse_handler = CallbackHandler(
            public_key=LANGFUSE_PUB_KEY,
            secret_key=LANGFUSE_SEC_KEY
        )
        callbacks = [langfuse_handler]
        print("[✓] Langfuse monitoring active — visit https://cloud.langfuse.com")
    else:
        print("[!] Langfuse keys not set. Running without observability tracing.")
except ImportError:
    print("[!] langfuse not installed. pip install langfuse to enable monitoring.")

# ─── LLM Configuration ────────────────────────────────────────────────────────
# Primary LLM
primary_llm = ChatOpenAI(
    model="gpt-4o-mini",        # cheap, fast, good for structured tasks
    temperature=0.2,
    callbacks=callbacks
)

# Fallback LLM (used if primary fails or gives bad output)
fallback_llm = ChatOpenAI(
    model="gpt-3.5-turbo",      # even cheaper fallback
    temperature=0.1,
    callbacks=callbacks
)

# ─── Tools ────────────────────────────────────────────────────────────────────

# Tool 1: File reader — reads the well data text file (Pre-built)
file_reader = FileReadTool(file_path="data/well_data.txt")

# Tool 2: Web search — looks up production benchmarks or field info (Pre-built)
web_search = SerperDevTool()

# Tool 3: Custom reservoir calculator (Custom-built)
import sys
sys.path.insert(0, "tools")
from reservoir_calculator import ReservoirCalculatorTool
reservoir_calc = ReservoirCalculatorTool()

# ─── Agents ───────────────────────────────────────────────────────────────────

# AGENT 1: Data Scout
data_scout = Agent(
    role="Petroleum Data Scout",
    goal=(
        "Read the well data file and extract all key production parameters: "
        "flow rate, pressure, OOIP, cumulative production, and decline history."
    ),
    backstory=(
        "You are a field data engineer with 10 years reading well test reports "
        "and production logs. You extract numbers with precision and flag any gaps "
        "or anomalies in the data before passing it forward."
    ),
    tools=[file_reader, web_search],
    llm=primary_llm,
    verbose=True,
    max_iter=3,                 # limits retries to avoid infinite loops
)

# AGENT 2: Reservoir Analyst
reservoir_analyst = Agent(
    role="Reservoir Engineer",
    goal=(
        "Use the extracted well parameters to calculate the Productivity Index (PI), "
        "Recovery Factor (RF), and annual decline rate (D). "
        "Interpret what these numbers mean for the well's health."
    ),
    backstory=(
        "You are a senior reservoir engineer trained on Craft & Hawkins and "
        "Dake's Fundamentals. You trust numbers over intuition and always show "
        "your calculation inputs before reporting results. "
        "You use the reservoir_calculator tool for every calculation."
    ),
    tools=[reservoir_calc],
    llm=primary_llm,
    verbose=True,
    max_iter=3,
)

# AGENT 3: Report Writer
report_writer = Agent(
    role="Engineering Report Writer",
    goal=(
        "Compile a concise, professional well performance report based on the "
        "data summary and calculated metrics. Include an actionable recommendation "
        "(e.g., artificial lift, workover, infill drilling, or abandonment)."
    ),
    backstory=(
        "You write reports for reservoir engineers and asset managers. "
        "You are clear, factual, and use standard petroleum engineering terminology. "
        "Your reports are always under 600 words and end with a ranked recommendation table."
    ),
    tools=[],                   # writer uses no tools — works on prior agent outputs
    llm=primary_llm,
    verbose=True,
    max_iter=2,
)

# ─── Tasks ────────────────────────────────────────────────────────────────────

# TASK 1: Read and summarize well data
task_read_data = Task(
    description=(
        "Read the file at data/well_data.txt using the file reader tool. "
        "Extract and list the following in a structured format:\n"
        "  - Well name and field\n"
        "  - Reservoir pressure and flowing BHP (calculate drawdown)\n"
        "  - Current flow rate (STB/day)\n"
        "  - Initial flow rate and years since first production\n"
        "  - OOIP (MMSTB) and cumulative production (MMSTB)\n"
        "  - Drive mechanism and GOR\n"
        "If any value is missing, state 'Not available' rather than guessing."
    ),
    expected_output=(
        "A structured bullet-point summary of all extracted well parameters, "
        "with units clearly stated for each value."
    ),
    agent=data_scout,
)

# TASK 2: Calculate reservoir metrics
task_calculate = Task(
    description=(
        "Using the parameters from Task 1, call the reservoir_calculator tool "
        "with the following input string format:\n"
        "  'flow_rate=X, drawdown=Y, OOIP=Z, cumulative=W, qi=A, qf=B, time=C'\n\n"
        "Where:\n"
        "  flow_rate = current flow rate in STB/day\n"
        "  drawdown  = reservoir pressure minus FBHP in psi\n"
        "  OOIP      = original oil in place in MMSTB\n"
        "  cumulative= cumulative production in MMSTB\n"
        "  qi        = initial rate at first production in STB/day\n"
        "  qf        = current rate in STB/day\n"
        "  time      = years since first production\n\n"
        "Report the three calculated values (PI, RF, Decline Rate) and explain "
        "what each result means for this specific well.\n\n"
        "FALLBACK: If the tool fails, estimate using industry rules of thumb: "
        "PI ~ 0.5–2.0 STB/day/psi for similar reservoirs, RF ~ 20–35% for solution gas drive."
    ),
    expected_output=(
        "Three calculated metrics (PI, RF, Decline Rate) with numeric values, units, "
        "and a brief interpretation of whether each result is good, average, or poor "
        "relative to typical field benchmarks."
    ),
    agent=reservoir_analyst,
    context=[task_read_data],   # receives output from Task 1
)

# TASK 3: Write engineering report
task_write_report = Task(
    description=(
        "Using the data summary (Task 1) and calculated metrics (Task 2), "
        "write a professional well performance report with these sections:\n"
        "  1. Well Overview (2–3 sentences)\n"
        "  2. Production Performance Summary (bullet points)\n"
        "  3. Calculated Metrics Table (PI, RF, Decline Rate)\n"
        "  4. Key Observations (2–3 bullet points)\n"
        "  5. Recommendations (ranked table with Action, Priority, Rationale)\n\n"
        "Keep the total report under 600 words. Use standard petroleum engineering "
        "terminology. The report should be actionable for a field engineer."
    ),
    expected_output=(
        "A complete well performance report in markdown format, saved to "
        "outputs/well_report.md, with all five sections filled in and "
        "a recommendations table at the end."
    ),
    agent=report_writer,
    context=[task_read_data, task_calculate],  # receives both prior outputs
    output_file="outputs/well_report.md",      # saves report to file automatically
)

# ─── Crew Assembly ────────────────────────────────────────────────────────────

crew = Crew(
    agents=[data_scout, reservoir_analyst, report_writer],
    tasks=[task_read_data, task_calculate, task_write_report],
    process=Process.sequential,   # tasks run one after another in order
    verbose=True,
)

# ─── Fallback Wrapper & Main Execution ────────────────────────────────────────

def run_with_fallback():
    """
    Runs the crew. If the primary LLM fails, switches all agents to fallback_llm and retries.
    This demonstrates the fallback mechanism required by the assessment.
    """
    print("\n" + "="*60)
    print("  RESERVOIR WELL PRODUCTION ANALYZER")
    print("  Powered by CrewAI + GPT-4o-mini")
    print("="*60 + "\n")

    try:
        result = crew.kickoff()
        print("\n[✓] Analysis complete. Report saved to outputs/well_report.md")
        return result

    except Exception as primary_error:
        print(f"\n[!] Primary LLM failed: {primary_error}")
        print("[→] Activating fallback: switching to GPT-3.5-turbo and retrying...\n")

        # Switch all agents to fallback LLM
        for agent in crew.agents:
            agent.llm = fallback_llm

        try:
            result = crew.kickoff()
            print("\n[✓] Fallback succeeded. Report saved to outputs/well_report.md")
            return result
        except Exception as fallback_error:
            print(f"\n[✗] Fallback also failed: {fallback_error}")
            print("[✗] Please check your API keys in .env and try again.")
            return None


if __name__ == "__main__":
    run_with_fallback()
