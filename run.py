# --- IMPORTS ---
import os
import yaml
import json
import subprocess
import logging
import re
import sys
import argparse
from typing import Dict, Any

from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, RootModel, Field

from crewai import Agent, Task, Crew, Process
from crewai.tasks import TaskOutput
from langchain_openai import ChatOpenAI
from crewai_tools import FileReadTool, FileWriterTool

import litellm
from tools.shell_tool import ShellTool

# --- ENVIRONMENT & TELEMETRY ---
# litellm.set_verbose = True
# os.environ['LITELLM_LOG'] = 'DEBUG'
litellm._turn_on_debug()

# See also setting in .env file
# This will cause problems in subprocesses if enabled
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["CREWAI_TRACING_ENABLED"] = "false"

# --- CONFIGURATION & CONSTANTS ---
PROJECT_CONFIG_FILE = "project_config.yaml"
PROMPT_TEMPLATE_FILE = "templates/prompts.yaml"
OUTPUT_DIR = "./output"
PLANNING_DIR = f"{OUTPUT_DIR}/planning"
STORIES_DIR = f"{OUTPUT_DIR}/stories"
SRC_DIR = f"{OUTPUT_DIR}/src"
TESTS_DIR = f"{OUTPUT_DIR}/tests"
DOCS_DIR = f"{OUTPUT_DIR}/docs"
LOGS_DIR = f"{OUTPUT_DIR}/logs"
MAX_FIX_ATTEMPTS = 3


# --- STRUCTURED LOGGING SETUP ---
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
        }

        standard_attrs = {
            "args",
            "asctime",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "message",
            "module",
            "msecs",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "thread",
            "threadName",
        }
        for key, value in record.__dict__.items():
            if key not in standard_attrs:
                log_record[key] = value

        if record.args:
            log_record["details"] = [str(arg) for arg in record.args]

        return json.dumps(log_record)


logger = logging.getLogger()
logger.setLevel(logging.INFO)
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setFormatter(JsonFormatter())
logger.addHandler(stdout_handler)


def setup_logger(log_file_name, append=False):
    """Configures the root logger to output to a specific file."""
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close()

    file_mode = "a" if append else "w"
    file_handler = logging.FileHandler(
        os.path.join(LOGS_DIR, log_file_name), mode=file_mode
    )
    file_handler.setFormatter(JsonFormatter())
    logger.addHandler(file_handler)
    logger.info(f"Logging reconfigured to '{log_file_name}' (mode: {file_mode})")


# Initial logger setup
setup_logger("run.log", append=False)


# --- LITELLM CALLBACK FOR DETAILED LOGGING ---
class LiteLLMLoggingCallback:
    def litellm_pre_call(self, kwargs):
        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])
        logger.info("LLM Call Pre-Call", extra={"model": model, "messages": messages})

    def litellm_post_call(self, kwargs, response):
        model = kwargs.get("model", "unknown")
        if response is None:
            response_content = "None response"
        else:
            response_content = getattr(response, "content", str(response))
        logger.info(
            "LLM Call Post-Call", extra={"model": model, "response": response_content}
        )

    def litellm_failure_callback(self, kwargs, response):
        model = kwargs.get("model", "unknown")
        logger.error("LLM Call Failed", extra={"model": model, "error": str(response)})


# --- PYDANTIC MODELS ---
class Stories(RootModel[Dict[str, str]]):
    """Stories is a pydantic class for CrewAI to parse stories into from json output."""

    pass


class ScaffoldOutput(RootModel[Dict[str, str]]):
    """ScaffoldOutput is a pydantic class for CrewAI to parse scaffolding results into from json output."""

    root: Dict[str, str] = Field(
        ...,
        example={
            "src/my_project/main.py": "created",
            "tests/my_project/test_main.py": "created",
            "README.md": "created",
        },
    )


# --- LLM AND TOOL CONFIGURATION ---
try:
    llm = ChatOpenAI(
        base_url=os.getenv("OPENAI_API_BASE", "http://192.168.2.77:8080/v1"),
        api_key=os.getenv("OPENAI_API_KEY", "not-needed"),
        temperature=0.1,
        max_tokens=32000,
    )
    logger.info("SUCCESS: ChatOpenAI client initialized successfully.")
    litellm.callbacks = [LiteLLMLoggingCallback()]
except Exception as e:
    logger.error(f"ERROR: Failed to initialize ChatOpenAI client: {e}", exc_info=True)
    sys.exit(1)

shell_tool = ShellTool()


# --- CALLBACK & HELPER FUNCTIONS ---
def _parse_and_validate_output(
    output: TaskOutput, pydantic_model: BaseModel, callback_name: str
):
    """
    Helper to parse raw task output and validate with a Pydantic model.
    Handles CrewAI's output variations (json_dict, pydantic, or raw).
    """
    logger.info(f"DEBUG: {callback_name} called.")
    logger.debug(
        f"DEBUG: output.raw type: {type(output.raw)}, content (first 500 chars): {str(output.raw)[:500]}"
    )
    logger.debug(
        f"DEBUG: output.json_dict type: {type(output.json_dict)}, content: {output.json_dict}"
    )
    logger.debug(
        f"DEBUG: output.pydantic type: {type(output.pydantic)}, content: {output.pydantic}"
    )

    if output.pydantic and isinstance(output.pydantic, pydantic_model):
        logger.info(f"DEBUG: Using output.pydantic for {callback_name}.")
        return output.pydantic.root

    try:
        raw_json_data = json.loads(output.raw)
        validated_data = pydantic_model.model_validate(raw_json_data).root
        logger.info(
            f"DEBUG: Using Pydantic.model_validate(raw_json_data).root for {callback_name}."
        )
        return validated_data
    except json.JSONDecodeError as e:
        logger.error(
            f"JSON decoding error in {callback_name}: {e}",
            exc_info=True,
            extra={"raw_output": output.raw},
        )
        raise
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during Pydantic validation in {callback_name}: {e}",
            exc_info=True,
            extra={"raw_output": output.raw},
        )
        raise


def _save_stories_callback(output: TaskOutput):
    """
    Callback function to save generated stories from the Scrum Master.
    """
    try:
        stories_map = _parse_and_validate_output(
            output, Stories, "_save_stories_callback"
        )

        for filename, content in stories_map.items():
            filepath = os.path.join(STORIES_DIR, filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            if content is not None:  # Ensure content is not None before writing
                with open(filepath, "w") as f:
                    f.write(content)
                logger.info(f"Story saved: {filepath}")
            else:
                logger.warning(
                    f"Skipping saving story {filename} due to empty content."
                )
        logger.info(f"All stories processed in '{STORIES_DIR}'.")
    except Exception as e:
        logger.error(f"Error in _save_stories_callback: {e}", exc_info=True)
        raise  # Re-raise to ensure the task fails if an issue occurs


def _save_tests_callback(output: TaskOutput, test_file_path: str):
    """
    Callback function to save generated tests from the QA Engineer.
    It extracts clean Python code by stripping markdown fences and writes it to the specified test file.
    """
    try:
        # Check if the output is a CrewAI TaskOutput object
        if isinstance(output, TaskOutput):
            raw_output = output.raw
        else:
            raw_output = str(output)

        # Remove markdown code fences if present
        # This regex looks for:
        # ``` (optional language, e.g., python) newline
        # (captured content) newline
        # ```
        match = re.search(r"```(?:\w+)?\n(.*?)\n```", raw_output, re.DOTALL)
        if match:
            clean_content = match.group(1).strip()
        else:
            # If no markdown fences, assume the whole output is the code
            clean_content = raw_output.strip()

        logger.debug(f"Writing cleaned test content to: {test_file_path}")
        os.makedirs(os.path.dirname(test_file_path), exist_ok=True)
        with open(test_file_path, "w") as f:
            f.write(clean_content)
        logger.info(f"Test file saved: {test_file_path}")
    except Exception as e:
        logger.error(f"Error in _save_tests_callback: {e}", exc_info=True)
        raise  # Re-raise to ensure the task fails if an issue occurs


def _extract_code_map_from_output(
    output: TaskOutput, callback_name: str
) -> Dict[str, str]:
    """
    Helper function to extract the code map (filename to content) from TaskOutput.
    Handles CrewAI's output variations (json_dict, pydantic, or raw markdown).
    """
    logger.debug(f"DEBUG: _extract_code_map_from_output called for {callback_name}.")
    logger.debug(
        f"DEBUG: output.raw type: {type(output.raw)}, content (first 500 chars): {str(output.raw)[:500]}"
    )
    logger.debug(
        f"DEBUG: output.json_dict type: {type(output.json_dict)}, content: {output.json_dict}"
    )
    logger.debug(
        f"DEBUG: output.pydantic type: {type(output.pydantic)}, content: {output.pydantic}"
    )

    code_map = {}

    # Attempt 1: Direct Pydantic or json_dict if available and correctly typed
    if isinstance(output.pydantic, dict):
        code_map = output.pydantic
    elif isinstance(output.json_dict, dict):
        code_map = output.json_dict
    elif isinstance(output.raw, str):
        # Attempt 2: Try to parse raw content as JSON
        try:
            parsed_raw = json.loads(output.raw)
            if isinstance(parsed_raw, dict):
                code_map = parsed_raw
            else:
                raise ValueError("JSON in raw output is not a dictionary.")
        except json.JSONDecodeError:
            # Attempt 3: Try to extract code from markdown
            markdown_code_blocks = re.findall(
                r"```(?:python)?\n(.*?)\n```", output.raw, re.DOTALL
            )
            if markdown_code_blocks:
                # If there's only one block, assume it's the main application file.
                # This is a heuristic and might need refinement.
                if len(markdown_code_blocks) == 1:
                    logger.warning(
                        f"Extracted single markdown code block. Assuming main app file for {callback_name}."
                    )
                    # This name needs to come from context, for now we will hardcode to task_manager.py
                    code_map = {"task_manager.py": markdown_code_blocks[0]}
                else:
                    logger.error(
                        f"Multiple markdown code blocks found in {callback_name}. Cannot determine filenames."
                    )
                    raise ValueError(
                        "Multiple markdown code blocks found, unable to map to filenames."
                    )
            else:
                logger.warning(
                    f"No JSON or markdown code blocks found in {callback_name}. Treating raw output as single file content."
                )
                # Fallback: Treat the entire raw output as content for a single, default file.
                # This is a very weak heuristic but provides some resilience.
                code_map = {
                    "task_manager.py": output.raw
                }  # Again a heuristic, needs improvement through better prompting

    if not isinstance(code_map, dict) or not code_map:
        logger.error(
            f"Failed to extract a valid code map from TaskOutput in {callback_name}.",
            extra={"raw_output": output.raw},
        )
        raise ValueError("Could not extract valid code map from agent output.")

    return code_map


def _apply_code_callback(output: TaskOutput, codebase_dir: str):
    """
    Callback function to apply generated code changes from the Dev Agent.
    It extracts code from the TaskOutput and writes it to the specified codebase directory.
    """
    try:
        code_map = _extract_code_map_from_output(output, "_apply_code_callback")

        for filename, content in code_map.items():
            filepath = os.path.join(codebase_dir, filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "w") as f:
                f.write(content)
            logger.info(f"Applied changes to {filename}")
        return True
    except Exception as e:
        logger.error(
            f"An unexpected error occurred in _apply_code_callback: {e}",
            exc_info=True,
            extra={"raw_output": output.raw},
        )
        return False


def _scaffold_callback(output: TaskOutput, src_dir: str, tests_dir: str):
    """
    Callback function to apply generated scaffolding from the Scaffolder Agent.
    """
    try:
        scaffold_map = _parse_and_validate_output(
            output, ScaffoldOutput, "_scaffold_callback"
        )

        os.makedirs(src_dir, exist_ok=True)
        os.makedirs(tests_dir, exist_ok=True)

        for filepath_relative, status in scaffold_map.items():
            if status == "created":
                full_path = ""
                # Determine if the file belongs in src or tests
                if filepath_relative.startswith("src/"):
                    full_path = os.path.join(OUTPUT_DIR, filepath_relative)
                elif filepath_relative.startswith("tests/"):
                    full_path = os.path.join(OUTPUT_DIR, filepath_relative)
                else:
                    logger.warning(
                        f"File '{filepath_relative}' does not start with 'src/' or 'tests/' -- creating in {OUTPUT_DIR}"
                    )
                    full_path = os.path.join(OUTPUT_DIR, filepath_relative)

                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(
                    full_path, "a"
                ):  # Create empty file if it doesn't exist, do not overwrite
                    pass
                logger.info(f"Scaffolded empty file: {full_path}")
            else:
                logger.warning(
                    f"Skipping scaffolding for {filepath_relative} with unexpected status: {status}"
                )
        logger.info(f"All scaffolding processed.")
        return True
    except Exception as e:
        logger.error(f"Error in _scaffold_callback: {e}", exc_info=True)
        return False  # Return False to indicate scaffolding failure


def render_crew_definitions(
    project_config,
    prd_content="",
    architecture_content="",
    developer_instructions="",
    dev_context=None,
):
    env = Environment(loader=FileSystemLoader("."))
    template = env.get_template(PROMPT_TEMPLATE_FILE)
    context = {
        **project_config,
        "prd_content": prd_content,
        "architecture_content": architecture_content,
        "developer_instructions": developer_instructions,
    }
    if dev_context:
        # Only add story_content to the context if it's explicitly provided
        if "story_content" in dev_context:
            context["story_content"] = dev_context["story_content"]
        if "test_file_content" in dev_context:
            context["test_file_content"] = dev_context["test_file_content"]
        if "error_log" in dev_context:
            context["error_log"] = dev_context["error_log"]
    rendered_yaml_str = template.render(context)
    return yaml.safe_load(rendered_yaml_str)


def create_project_directories(project_dir):
    os.makedirs(PLANNING_DIR, exist_ok=True)
    os.makedirs(STORIES_DIR, exist_ok=True)
    os.makedirs(SRC_DIR, exist_ok=True)
    os.makedirs(TESTS_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    project_src_dir = os.path.join(SRC_DIR, project_dir)
    os.makedirs(project_src_dir, exist_ok=True)
    project_tests_dir = os.path.join(TESTS_DIR, project_dir)
    os.makedirs(project_tests_dir, exist_ok=True)


# --- PHASE-SPECIFIC FUNCTIONS ---
def run_planning_crew_tasks(project_config):
    """Encapsulates the tasks for the planning crew."""
    crew_defs = render_crew_definitions(project_config)
    agents = {
        name: Agent(llm=llm, **props) for name, props in crew_defs["agents"].items()
    }

    brief_task = Task(
        agent=agents["analyst"],
        name="Generate Project Brief",
        **crew_defs["tasks"]["create_brief"],
    )

    prd_file_path = os.path.join(PLANNING_DIR, "PRD.md")
    prd_task = Task(
        agent=agents["project_manager"],
        name="Create Product Requirements Document (PRD)",
        context=[brief_task],
        output_file=prd_file_path,
        **crew_defs["tasks"]["create_prd"],
    )

    architecture_file_path = os.path.join(PLANNING_DIR, "ARCHITECTURE.md")
    architecture_task = Task(
        agent=agents["architect"],
        name="Design System Architecture",
        context=[prd_task],
        output_file=architecture_file_path,
        **crew_defs["tasks"]["create_architecture"],
    )

    planning_crew = Crew(
        agents=[agents["analyst"], agents["project_manager"], agents["architect"]],
        tasks=[brief_task, prd_task, architecture_task],
        process=Process.sequential,
        verbose=True,
        share_crew=False,
        human_input=False,
        telemetry=False,
        tracing=False,
        output_log_file=os.path.join(LOGS_DIR, "planning_crew.log"),
    )
    planning_crew.kickoff()
    logger.info(f"Planning complete. Documents saved in '{PLANNING_DIR}'.")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the AI developer workflow.")
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Automatically approve the planning phase",
    )
    parser.add_argument(
        "--skip-planning",
        action="store_true",
        help="Skip the planning phase and proceed directly to development",
    )
    return parser.parse_args()


def load_project_config():
    with open(PROJECT_CONFIG_FILE, "r") as f:
        return yaml.safe_load(f)


def setup_project_environment(project_config):
    project_dir_name = project_config["project_name"].lower().replace(" ", "_")
    create_project_directories(project_dir_name)


# --- PHASE-SPECIFIC FUNCTIONS ---
def handle_planning_phase(project_config, args):
    logger.info("--- PHASE 1: PLANNING ---")
    prd_content = ""
    architecture_content = ""

    if not args.skip_planning:
        logger.info("Starting planning phase...")
        run_planning_crew_tasks(project_config)

        prd_path = os.path.join(PLANNING_DIR, "PRD.md")
        architecture_path = os.path.join(PLANNING_DIR, "ARCHITECTURE.md")

        if not os.path.exists(prd_path) or not os.path.exists(architecture_path):
            logger.error("Planning documents were not created properly. Exiting.")
            sys.exit(1)

        logger.info(f"Planning complete. Documents saved in '{PLANNING_DIR}'.")

        if args.yes:
            logger.info("Auto-approving the planning phase.")
        else:
            approval = input(
                "Please review planning documents and type 'approve' to continue: "
            ).lower()
            if approval != "approve":
                logger.info("Project not approved. Exiting.")
                sys.exit(0)
    else:
        logger.info("Skipping planning phase as requested.")

    prd_path = os.path.join(PLANNING_DIR, "PRD.md")
    architecture_path = os.path.join(PLANNING_DIR, "ARCHITECTURE.md")

    if os.path.exists(prd_path):
        with open(prd_path, "r") as f:
            prd_content = f.read()
    if os.path.exists(architecture_path):
        with open(architecture_path, "r") as f:
            architecture_content = f.read()

    return prd_content, architecture_content


def handle_scaffolding_phase(project_config, prd_content, architecture_content):
    logger.info("--- PHASE 2: SCAFFOLDING ---")
    setup_logger("scaffolding.log")
    logger.info("Starting Scaffolding phase...")

    crew_defs = render_crew_definitions(
        project_config,
        prd_content=prd_content,
        architecture_content=architecture_content,
    )
    agents = {
        name: Agent(llm=llm, **props) for name, props in crew_defs["agents"].items()
    }
    architecture_path = os.path.join(PLANNING_DIR, "ARCHITECTURE.md")
    file_read_tool = FileReadTool(file_path=architecture_path)
    scaffold_write_tool = FileWriterTool(root_dir=OUTPUT_DIR)

    scaffold_task = Task(
        agent=agents["scaffolder"],
        name="Create Project Scaffold",
        description=crew_defs["tasks"]["create_scaffold"]["description"],
        expected_output=crew_defs["tasks"]["create_scaffold"]["expected_output"],
        tools=[file_read_tool, scaffold_write_tool],
        callback=lambda output: _scaffold_callback(output, SRC_DIR, TESTS_DIR),
        output_json=ScaffoldOutput,
    )

    scaffolding_crew = Crew(
        agents=[agents["scaffolder"]],
        tasks=[scaffold_task],
        share_crew=False,
        human_input=False,
        verbose=False,
        telemetry=False,
        tracing=False,
    )
    successful_scaffolding = scaffolding_crew.kickoff()

    if not successful_scaffolding:
        logger.error("Scaffolding phase failed. Exiting.")
        sys.exit(1)

    logger.info("Scaffolding phase completed successfully.")
    setup_logger("run.log", append=True)


def handle_developer_instructions_phase(
    project_config, prd_content, architecture_content
):
    logger.info("--- PHASE 3: DEVELOPER INSTRUCTIONS GENERATION ---")
    setup_logger("dev_instructions.log")

    crew_defs = render_crew_definitions(
        project_config,
        prd_content=prd_content,
        architecture_content=architecture_content,
    )
    agents = {
        name: Agent(llm=llm, **props) for name, props in crew_defs["agents"].items()
    }
    instructions_file_path = os.path.join(DOCS_DIR, "DEVELOPER_INSTRUCTIONS.md")
    instructions_write_tool = FileWriterTool(root_dir=DOCS_DIR)

    developer_instructions_task = Task(
        agent=agents["documentation_writer"],
        name="Generate Developer Instructions",
        description=crew_defs["tasks"]["generate_developer_instructions"][
            "description"
        ],
        expected_output=crew_defs["tasks"]["generate_developer_instructions"][
            "expected_output"
        ],
        tools=[instructions_write_tool],
        output_file=instructions_file_path,
    )

    dev_instructions_crew = Crew(
        agents=[agents["documentation_writer"]],
        tasks=[developer_instructions_task],
        share_crew=False,
        human_input=False,
        verbose=False,
        telemetry=False,
        tracing=False,
    )
    dev_instructions_crew.kickoff()

    developer_instructions_content = ""
    # No need to check os.path.exists for instructions_file_path here, as the task output_file handles writing.
    # We directly read it back if the task was successful.
    with open(instructions_file_path, "r") as f:
        developer_instructions_content = f.read()
    logger.info(
        f"Developer instructions generated and saved to {instructions_file_path}"
    )

    setup_logger("run.log", append=True)
    return developer_instructions_content


def handle_development_phase(
    project_config, prd_content, architecture_content, developer_instructions_content
):
    logger.info("--- PHASE 4: DEVELOPMENT ---")

    crew_defs = render_crew_definitions(
        project_config,
        prd_content=prd_content,
        architecture_content=architecture_content,
        developer_instructions=developer_instructions_content,
    )
    agents = {
        name: Agent(llm=llm, **props) for name, props in crew_defs["agents"].items()
    }

    # Step 1: Scrum Master generates stories
    setup_logger("scrum_master.log")
    logger.info("Starting Scrum Master story generation...")

    story_gen_task = Task(
        agent=agents["scrum_master"],
        name="Generate User Stories",
        description=crew_defs["tasks"]["generate_stories"]["description"],
        expected_output=crew_defs["tasks"]["generate_stories"]["expected_output"],
        callback=_save_stories_callback,
        output_json=Stories,
    )
    scrum_crew = Crew(
        agents=[agents["scrum_master"]],
        tasks=[story_gen_task],
        share_crew=False,
        human_input=False,
        verbose=False,
        telemetry=False,
        tracing=False,
    )

    scrum_crew.kickoff()
    logger.info("Scrum Master generated stories.")
    setup_logger("run.log", append=True)

    # Step 2: TDD Loop for each story
    story_files = sorted([f for f in os.listdir(STORIES_DIR) if f.endswith(".md")])
    if not story_files:
        logger.warning("No stories found to develop. Exiting development phase.")
        return

    project_dir_name = project_config["project_name"].lower().replace(" ", "_")
    test_file_path_for_qa = os.path.join(TESTS_DIR, f"test_{project_dir_name}.py")

    for story_file in story_files:
        _story_name = story_file.replace(".md", "")
        logger.info(f"--- Processing Story: {story_file} ---")
        with open(os.path.join(STORIES_DIR, story_file), "r") as f:
            story_content = f.read()

        # Step 2a: QA Agent writes tests
        setup_logger(f"qa_{_story_name}.log")
        logger.info(f"Starting QA for story: {_story_name}")

        qa_crew_defs = render_crew_definitions(
            project_config,
            prd_content=prd_content,
            architecture_content=architecture_content,
            developer_instructions=developer_instructions_content,
            dev_context={"story_content": story_content},
        )

        qa_task = Task(
            agent=agents["qa_engineer"],
            name=f"Write tests for {story_file}",
            description=qa_crew_defs["tasks"]["write_tests"]["description"],
            expected_output=qa_crew_defs["tasks"]["write_tests"]["expected_output"],
            callback=lambda output: _save_tests_callback(output, test_file_path_for_qa),
        )
        test_code_output = Crew(
            agents=[agents["qa_engineer"]],
            tasks=[qa_task],
            verbose=True,
            share_crew=False,
            human_input=False,
            telemetry=False,
            tracing=False,
        ).kickoff()
        logger.info(f"QA Engineer wrote tests to '{test_file_path_for_qa}'.")
        setup_logger("run.log", append=True)

        # Step 2b: Dev Agent writes code (with self-correction loop)
        for attempt in range(MAX_FIX_ATTEMPTS):
            setup_logger(f"dev_{_story_name}_attempt_{attempt+1}.log")
            logger.info(f"Dev Agent coding attempt {attempt + 1}/{MAX_FIX_ATTEMPTS}")

            if attempt == 0:
                dev_crew_defs = render_crew_definitions(
                    project_config,
                    prd_content=prd_content,
                    architecture_content=architecture_content,
                    developer_instructions=developer_instructions_content,
                    dev_context={
                        "story_content": story_content,
                        "test_file_content": test_code_output.raw,
                    },
                )
                dev_task_desc = dev_crew_defs["tasks"]["write_code"]["description"]
                dev_task_expected_out = dev_crew_defs["tasks"]["write_code"][
                    "expected_output"
                ]
            else:
                dev_crew_defs = render_crew_definitions(
                    project_config,
                    prd_content=prd_content,
                    architecture_content=architecture_content,
                    developer_instructions=developer_instructions_content,
                    dev_context={
                        "story_content": story_content,
                        "test_file_content": test_code_output.raw,
                        "error_log": test_output,
                    },
                )
                dev_task_desc = dev_crew_defs["tasks"]["fix_code"]["description"]
                dev_task_expected_out = dev_crew_defs["tasks"]["fix_code"][
                    "expected_output"
                ]

            dev_task = Task(
                agent=agents["dev_agent"],
                name=(
                    f"Develop code for {story_file}"
                    if attempt == 0
                    else f"Fix code for {story_file}"
                ),
                description=dev_task_desc,
                expected_output=dev_task_expected_out,
                context=[qa_task],
                callback=lambda output: _apply_code_callback(output, SRC_DIR),
            )
            Crew(
                agents=[agents["dev_agent"]],
                tasks=[dev_task],
                share_crew=False,
                human_input=False,
                telemetry=False,
                tracing=False,
            ).kickoff()

            test_raw_output = shell_tool.run(f"cd {OUTPUT_DIR} && pytest")
            test_output = _parse_pytest_output(test_raw_output)
            logger.info("Test Run Output:", extra={"output": test_output})

            if "Return Code: 0" in test_output and "failed" not in test_output:
                logger.info("Tests Passed! Story complete.")
                subprocess.run(["git", "add", "."], cwd=OUTPUT_DIR)
                subprocess.run(
                    ["git", "commit", "-m", f"feat: Implement story {_story_name}"],
                    cwd=OUTPUT_DIR,
                )
                break
            else:
                logger.warning(f"Attempt {attempt + 1} failed. Retrying...")
        else:
            logger.error(
                f"Failed to complete story {story_file} after {MAX_FIX_ATTEMPTS} attempts. Stopping."
            )
            sys.exit(1)
        setup_logger("run.log", append=True)


def _parse_pytest_output(raw_output: str) -> str:
    """
    Parses the raw pytest output to extract and highlight relevant error messages.
    """
    lines = raw_output.split("\n")
    parsed_output = []

    in_error_section = False
    error_header_pattern = re.compile(r"____.*____")
    error_detail_pattern = re.compile(r"E\s+.*")

    for line in lines:
        if "=== ERRORS ===" in line:
            in_error_section = True
            parsed_output.append(line)
            continue

        if in_error_section:
            if error_header_pattern.match(line):
                # This is a test method error header, include it
                parsed_output.append(line)
            elif error_detail_pattern.match(line):
                # This is an error detail line, include it
                parsed_output.append(line)
            elif line.strip().startswith("E   "):  # General Python Error lines
                parsed_output.append(line)
            elif (
                "--- Captured stdout call ---" in line
                or "--- Captured stderr call ---" in line
            ):
                # Often verbose, skip these for brevity in the parsed output
                pass
            elif "==============" in line and "errors" in line:
                # End of error section summary
                parsed_output.append(line)
                in_error_section = False
            elif line.strip() == "":
                # Keep blank lines around errors for readability
                parsed_output.append(line)

        if (
            "Return Code:" in line
            or "collected" in line
            or "passed in" in line
            or "failed in" in line
        ):
            parsed_output.append(line)

    if (
        not parsed_output and raw_output
    ):  # If no specific errors found, but there was output, return the whole thing
        return raw_output

    return "\n".join(parsed_output)


# --- MAIN ORCHESTRATION ---
def main():
    args = parse_arguments()
    logger.info("--- Starting Autonomous AI Developer ---")
    project_config = load_project_config()
    setup_project_environment(project_config)

    prd_content, architecture_content = handle_planning_phase(project_config, args)
    handle_scaffolding_phase(project_config, prd_content, architecture_content)
    developer_instructions_content = handle_developer_instructions_phase(
        project_config, prd_content, architecture_content
    )
    handle_development_phase(
        project_config,
        prd_content,
        architecture_content,
        developer_instructions_content,
    )

    logger.info("--- All stories implemented. Project complete. ---")


if __name__ == "__main__":
    main()
