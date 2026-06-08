reservoir_crew/
│
├── main.py                     ← Run this! All agents, tasks, and crew are here
│
├── tools/
│   ├── reservoir_calculator.py ← Custom tool: calculates PI, RF, Decline Rate
│   ├── fallback_handler.py     ← Handles tool/LLM failures gracefully
│   └── langfuse_config.py      ← Observability setup
│
├── data/
│   └── well_data.txt           ← Input: well production parameters
│
├── outputs/
│   └── well_report.md          ← Output: generated engineering report
│
├── requirements.txt
├── .env.example                ← Copy to .env and add your API keys
└── README.md
