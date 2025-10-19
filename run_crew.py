import os
import yaml
import json
import logging
import sys
from jinja2 import Environment, FileSystemLoader
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from crewai_tools import FileReadTool
from tools.shell_tool import ShellTool
import litellm

litellm._turn_on_debug()
os.environ['OTEL_SDK_DISABLED'] = 'true'
os.environ['CREWAI_DISABLE_TELEMETRY'] = 'true'
os.environ['CREWAI_TRACING_ENABLED'] = 'false'

# --- LiteLLM Callback for Detailed Logging ---
class LiteLLMLoggingCallback:
    def litellm_pre_call(self, kwargs):
        # Log the model and messages before the call
        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])
        logger.info("LLM Call Pre-Call", extra={
            "model": model,
            "messages": messages
        })
    
    def litellm_post_call(self, kwargs, response):
        # Log the raw response from the LLM
        model = kwargs.get("model", "unknown")
        if response is None:
            response_content = "None response"
        else:
            response_content = getattr(response, 'content', str(response))
        logger.info("LLM Call Post-Call", extra={
            "model": model,
            "response": response_content
        })
    
    def litellm_failure_callback(self, kwargs, response):
        # Log any failures
        model = kwargs.get("model", "unknown")
        logger.error("LLM Call Failed", extra={
            "model": model,
            "error": str(response)
        })

# --- Configuration & Constants ---
PROJECT_CONFIG_FILE = 'project_config.yaml'
PROMPT_TEMPLATE_FILE = 'templates/prompts.yaml'
PLANNING_DIR = './output/planning'
STORIES_DIR = './output/stories'

# --- Structured Logging Setup ---
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        
        # Add any extra fields from the log record
        if hasattr(record, 'extra'):
            log_record.update(record.extra)
        
        # Add details if they exist
        if record.args:
            log_record["details"] = [str(arg) for arg in record.args]
            
        return json.dumps(log_record)

log_handler = logging.FileHandler('./output/logs/run.log', mode='w')
log_handler.setFormatter(JsonFormatter())
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)
logger.addHandler(logging.StreamHandler(sys.stdout)) # Also print to console

# --- LLM and Tool Configuration ---
# NOTE: Update LLM parameters based on your model and hardware.
# n_gpu_layers=-1 will try to offload all layers to GPU.
# For CPU only, set n_gpu_layers=0.
# For remote model usage, specify the model URL in model_kwargs
try:
    # Use ChatOpenAI to connect to an OpenAI-compatible API endpoint
    llm = ChatOpenAI(
        # The base_url should point to the v1 endpoint of your server
        base_url=os.getenv("OPENAI_API_BASE", "http://192.168.2.77:8080/v1"),
        
        # The API key is required, but can be a dummy value if your server doesn't use one
        api_key=os.getenv("OPENAI_API_KEY", "not-needed"), 
        
        # Specify the model name explicitly
        # model="qwen3coder30b",
        
        temperature=0.1,
        
        # You can pass other model parameters here
        # model_kwargs might not be needed, as params are often top-level
        max_tokens=2048, 
    )

    print("SUCCESS: ChatOpenAI client initialized successfully.")

    # Register the LiteLLM callback for detailed logging
    litellm.callbacks = [LiteLLMLoggingCallback()]

    # Example of how to use it
    # response = llm.invoke("Explain what a Docker container is in a single sentence.")
    # print(response.content)

except Exception as e:
    print(f"ERROR: Failed to initialize ChatOpenAI client: {e}")
    import traceback
    traceback.print_exc()

# --- Helper Functions ---
def render_crew_definitions(project_config):
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template(PROMPT_TEMPLATE_FILE)
    rendered_yaml_str = template.render(project_config)
    return yaml.safe_load(rendered_yaml_str)

def apply_code_changes(code_json_str, codebase_dir):
    try:
        code_map = json.loads(code_json_str)
        for filename, content in code_map.items():
            filepath = os.path.join(codebase_dir, filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                f.write(content)
            logger.info(f"Applied changes to {filename}")
        return True
    except (json.JSONDecodeError, TypeError) as e:
        logger.error("Failed to parse or apply LLM code output.", extra={"error": str(e), "output": code_json_str})
        return False

# --- Main Orchestration ---
def main():
    import os
    import sys
    
    # Redirect stdin to /dev/null to prevent hanging on interactive prompts
    if not sys.stdin.isatty():
        sys.stdin = open('/dev/null', 'r')
    
    logger.info("--- Starting Autonomous AI Developer ---")
    
    # Load project config
    with open(PROJECT_CONFIG_FILE, 'r') as f:
        project_config = yaml.safe_load(f)
    
    project_dir = project_config['project_name'].lower().replace(' ', '_')

    # Render crew definitions with project context
    crew_defs = render_crew_definitions(project_config)

    # Instantiate agents
    agents = {name: Agent(llm=llm, **props) for name, props in crew_defs['agents'].items()}

    # --- PHASE 1: PLANNING ---
    logger.info("--- Phase 1: Planning ---")
    planning_tasks = [
        Task(agent=agents['analyst'], **crew_defs['tasks']['create_brief']),
        Task(agent=agents['project_manager'], context=[Task(agent=agents['analyst'], **crew_defs['tasks']['create_brief'])], **crew_defs['tasks']['create_prd']),
        Task(agent=agents['architect'], context=[Task(agent=agents['project_manager'], **crew_defs['tasks']['create_prd'])], **crew_defs['tasks']['create_architecture'])
    ]
    planning_crew = Crew(agents=[agents['analyst'], agents['project_manager'], agents['architect']], tasks=planning_tasks, process=Process.sequential, share_crew=False, human_input=False, verbose=False, telemetry=False, tracing=False)
    planning_result = planning_crew.kickoff()

    sys.stdout.flush()
    
    # Save planning documents
    # The final task's output is the result for a sequential crew
    architecture_doc = planning_result.raw
    prd_doc = planning_crew.tasks[1].output.raw
    with open(os.path.join(PLANNING_DIR, 'PRD.md'), 'w') as f: f.write(prd_doc)
    with open(os.path.join(PLANNING_DIR, 'ARCHITECTURE.md'), 'w') as f: f.write(architecture_doc)

    logger.info(f"Planning complete. Documents saved in '{PLANNING_DIR}'.")

if __name__ == "__main__":
    main()
