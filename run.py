import os
import yaml
import json
import subprocess
import logging
import sys
import argparse
from jinja2 import Environment, FileSystemLoader
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from crewai_tools import FileReadTool, FileWriterTool
from tools.shell_tool import ShellTool
import litellm

#litellm.set_verbose = True
# os.environ['LITELLM_LOG'] = 'DEBUG'
litellm._turn_on_debug()

# See also setting in .env file
# This will cause problems in subprocesses if enabled
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
OUTPUT_DIR = './output'
PLANNING_DIR = f'{OUTPUT_DIR}/planning'
STORIES_DIR = f'{OUTPUT_DIR}/stories'
SRC_DIR = f'{OUTPUT_DIR}/src'
TESTS_DIR = f'{OUTPUT_DIR}/tests'
LOGS_DIR = f'{OUTPUT_DIR}/logs'
MAX_FIX_ATTEMPTS = 3

# --- Structured Logging Setup ---
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        
        # Add extra fields from the log record, excluding standard ones
        standard_attrs = {"args", "asctime", "created", "exc_info", "exc_text", "filename", "funcName", "levelname", "levelno", "lineno", "message", "module", "msecs", "msg", "name", "pathname", "process", "processName", "relativeCreated", "stack_info", "thread", "threadName"}
        for key, value in record.__dict__.items():
            if key not in standard_attrs:
                log_record[key] = value
        
        # Add details if they exist
        if record.args:
            log_record["details"] = [str(arg) for arg in record.args]
            
        return json.dumps(log_record)

log_handler = logging.FileHandler(f'{LOGS_DIR}/run.log', mode='w')
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
shell_tool = ShellTool()

# --- Helper Functions ---
def render_crew_definitions(project_config):
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template(PROMPT_TEMPLATE_FILE)
    rendered_yaml_str = template.render(project_config)
    return yaml.safe_load(rendered_yaml_str)

def create_project_directories(project_dir):
    os.makedirs(PLANNING_DIR, exist_ok=True)
    os.makedirs(STORIES_DIR, exist_ok=True)
    os.makedirs(SRC_DIR, exist_ok=True)
    os.makedirs(TESTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    # Create project directory within src
    project_src_dir = os.path.join(SRC_DIR, project_dir)
    os.makedirs(project_src_dir, exist_ok=True)
    # Create tests directory within tests
    project_tests_dir = os.path.join(TESTS_DIR, project_dir)
    os.makedirs(project_tests_dir, exist_ok=True)

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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the AI developer workflow.')
    parser.add_argument('-y', '--yes', action='store_true', help='Automatically approve the planning phase')
    parser.add_argument('--skip-planning', action='store_true', help='Skip the planning phase and proceed directly to development')
    args = parser.parse_args()
    
    logger.info("--- Starting Autonomous AI Developer ---")
    
    # Load project config
    with open(PROJECT_CONFIG_FILE, 'r') as f:
        project_config = yaml.safe_load(f)
    
    project_dir = project_config['project_name'].lower().replace(' ', '_')
    test_file_name = f"tests/test_{project_dir}.py"
    create_project_directories(project_dir)
    # Set project directory to be within src
    project_dir = os.path.join(SRC_DIR, project_dir)

    # Render crew definitions with project context
    crew_defs = render_crew_definitions(project_config)

    # Instantiate agents
    agents = {name: Agent(llm=llm, **props) for name, props in crew_defs['agents'].items()}

    # Create FileWriterTool for writing stories
    story_write_tool = FileWriterTool(output_dir=STORIES_DIR)

    # --- PHASE 1: PLANNING ---
    # Run the planning phase in a separate process to avoid input conflicts
    if not args.skip_planning:
        logger.info("Starting planning phase...")
        result = subprocess.run([sys.executable, 'run_crew.py'], 
                              capture_output=True, text=True, cwd=os.getcwd(), stdin=subprocess.DEVNULL)
        
        if result.returncode != 0:
            logger.error("Planning phase failed", extra={
                "error": result.stderr,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "command": [sys.executable, 'run_crew.py']
            })
            return
        
        logger.info("Planning phase completed successfully.")
        
        # Check if planning documents were created
        prd_path = os.path.join(PLANNING_DIR, 'PRD.md')
        architecture_path = os.path.join(PLANNING_DIR, 'ARCHITECTURE.md')
        
        if not os.path.exists(prd_path) or not os.path.exists(architecture_path):
            logger.error("Planning documents were not created properly.")
            return

        logger.info(f"Planning complete. Documents saved in '{PLANNING_DIR}'.")
        sys.stdout.flush()
        
        # Check if auto-approval is enabled
        if args.yes:
            logger.info("Auto-approving the planning phase.")
        else:
            approval = input("Please review planning documents and type 'approve' to continue: ").lower()
            if approval != 'approve':
                logger.info("Project not approved. Exiting.")
                return
    else:
        logger.info("Skipping planning phase as requested.")

    # --- PHASE 2: DEVELOPMENT ---
    logger.info("--- Phase 2: Development ---")
    
    # Step 1: Scrum Master generates stories
    # Create a FileReadTool instance with the correct file path
    prd_file_path = os.path.join(PLANNING_DIR, 'PRD.md')
    architecture_file_path = os.path.join(PLANNING_DIR, 'ARCHITECTURE.md')
    file_read_tool = FileReadTool(file_path=prd_file_path)
    story_gen_task = Task(
        agent=agents['scrum_master'],
        tools=[file_read_tool, story_write_tool],
        **crew_defs['tasks']['generate_stories']
    )
    scrum_crew = Crew(agents=[agents['scrum_master']], tasks=[story_gen_task], share_crew=False, human_input=False, verbose=False, telemetry=False, tracing=False)
    scrum_crew.kickoff()
    logger.info(f"Scrum Master generated stories in '{STORIES_DIR}'.")

    # Step 2: TDD Loop for each story
    story_files = sorted([f for f in os.listdir(STORIES_DIR) if f.endswith('.md')])

    for story_file in story_files:
        logger.info(f"--- Processing Story: {story_file} ---")
        with open(os.path.join(STORIES_DIR, story_file), 'r') as f:
            story_content = f.read()

        # Step 2a: QA Agent writes tests
        qa_task = Task(
            agent=agents['qa_engineer'],
            description=crew_defs['tasks']['write_tests']['description'].format(story_content=story_content),
            expected_output=crew_defs['tasks']['write_tests']['expected_output']
        )
        test_code = Crew(agents=[agents['qa_engineer']], tasks=[qa_task]).kickoff()
        test_file_path = os.path.join(TESTS_DIR, test_file_name)
        with open(test_file_path, 'w') as f: f.write(test_code)
        logger.info(f"QA Engineer wrote tests to '{test_file_path}'.")

        # Step 2b: Dev Agent writes code (with self-correction loop)
        for attempt in range(MAX_FIX_ATTEMPTS):
            logger.info(f"Dev Agent coding attempt {attempt + 1}/{MAX_FIX_ATTEMPTS}")
            
            if attempt == 0:
                # First attempt
                dev_task_desc = crew_defs['tasks']['write_code']['description'].format(story_content=story_content, test_file_content=test_code)
                dev_task_expected_out = crew_defs['tasks']['write_code']['expected_output']
            else:
                # Subsequent attempts (fixing)
                dev_task_desc = crew_defs['tasks']['fix_code']['description'].format(story_content=story_content, test_file_content=test_code, error_log=test_output)
                dev_task_expected_out = crew_defs['tasks']['fix_code']['expected_output']

            dev_task = Task(agent=agents['dev_agent'], description=dev_task_desc, expected_output=dev_task_expected_out)
            generated_code_json = Crew(agents=[agents['dev_agent']], tasks=[dev_task]).kickoff()
            
            if not apply_code_changes(generated_code_json, SRC_DIR):
                test_output = "Code generation failed, could not parse or apply changes."
                continue 

            # Run tests
            test_output = shell_tool.run(f"cd {OUTPUT_DIR} && pytest")
            logger.info("Test Run Output:", extra={"output": test_output})
            
            if "Return Code: 0" in test_output and "failed" not in test_output:
                logger.info("Tests Passed! Story complete.")
                subprocess.run(['git', 'add', '.'], cwd=OUTPUT_DIR)
                subprocess.run(['git', 'commit', '-m', f'feat: Implement story {story_file}'], cwd=OUTPUT_DIR)
                break
            else:
                logger.warning(f"Attempt {attempt + 1} failed. Retrying...")
        else:
            logger.error(f"Failed to complete story {story_file} after {MAX_FIX_ATTEMPTS} attempts. Stopping.")
            return

    logger.info("--- All stories implemented. Project complete. ---")

if __name__ == "__main__":
    main()
