# Autonomous AI Developer Agent

This repository contains a complete, autonomous AI developer system built using CrewAI and the BMAD methodology. It can take a high-level project description and execute a full development lifecycle, from planning and architecture to test-driven coding and version control.

## Architecture

This system implements an agentic workflow:

1.  **Configuration (`project_config.yaml`):** You define the project specifics (name, description, tech stack) here. The agent prompts are generic templates.
2.  **Orchestration (`run.py`):** The main script that manages the entire process. It renders prompts and orchestrates two main phases.
3.  **Phase 1: The Planning Crew:**
    *   **Analyst, PM, Architect** agents collaborate to produce a `PRD.md` and `ARCHITECTURE.md`.
    *   This phase includes optional, user-driven steps for market research and brainstorming using search tools.
    *   **Human-in-the-Loop:** The script pauses for your approval of the planning documents before proceeding.
4.  **Phase 2: The Development Crew (Orchestrated Loop):**
    *   The **Scrum Master** agent breaks the approved plan into detailed story files (`.md`).
    *   For each story, a **Test-Driven Development (TDD)** loop begins:
        1.  The **QA Engineer** agent writes a `pytest` test that initially fails.
        2.  The **Dev Agent** writes the application code to make the test pass.
        3.  The **Self-Correction Loop:** If tests fail, the captured error is fed back to the Dev Agent to fix the code. This repeats up to 3 times.
    *   Once tests pass, the code is committed to a local Git repository for that project.
5.  **Sandboxing:** The entire process is designed to run inside a **Docker** container to isolate file I/O and command execution, ensuring safety.

## Setup Instructions

### 0. Copy / Clone this project

Each new project can start as a copy. Then edit the `project_config.yaml` as needed

### 1. Prerequisites
- Docker installed and running.
- Python 3.9+ installed on your host machine (for local testing if needed).
- Your local LLM (e.g., via llama.cpp) must be running and accessible at the network address specified in `run.py`.

### 2. Build the Docker Image

From the root directory of this project (`/ai_developer_workspace/`), run:

```bash
docker build -t aidev .
```

This will build a container with all necessary dependencies and tools. 


## How to Run 

Configure Your Project: 

  - Open and edit project_config.yaml.
  - Define your project_name, one_liner_description, technical_stack, etc. This is the only file you need to change for a new project.

Bring up docker container with docker compose:

```
docker-compose up --build

docker-compose exec main bash

python run.py
```

If you need to jump ahead to a different phase:

```
python run.py --skip-planning
```


 
You may also run the Docker container directly:
          
```
docker run --rm -it --env-file ./.env -v "$(pwd):/app" aidev
```

The -it flag runs the container in interactive mode, which is necessary for the input() prompts (e.g., for brainstorming and plan approval).
     
The agent will now begin the process. You will be prompted in your terminal to approve steps along the way. 