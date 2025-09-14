import os
import dotenv
import asyncio
from typing import Dict, List, Any
from github import Github, GithubException

from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import FunctionAgent
from llama_index.core.agent.workflow import AgentWorkflow, AgentOutput, ToolCall, ToolCallResult
from llama_index.core.workflow import Context
from llama_index.core.prompts import RichPromptTemplate

# -----------------------------
# Load Environment Variables
# -----------------------------
dotenv.load_dotenv()

# -----------------------------
# GitHub & LLM Configuration
# -----------------------------
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
# REPOSITORY is in the format "username/repo-name" from GitHub Actions
REPOSITORY = os.getenv("REPOSITORY")  # This will be something like "andreifrolovmd/recipes-api"
PR_NUMBER_STR = os.getenv("PR_NUMBER")

# Initialize GitHub client
git = Github(GITHUB_TOKEN) if GITHUB_TOKEN else None

# Get repository object
repo = None
if git and REPOSITORY:
    try:
        # Use the repository name directly (not URL)
        repo = git.get_repo(REPOSITORY)
        print(f"✅ Successfully connected to repository: {REPOSITORY}")
    except GithubException as e:
        print(f"Error: Could not access repository '{REPOSITORY}'. Please check the name and your token permissions. Details: {e}")
else:
    if not REPOSITORY:
        print("Error: REPOSITORY environment variable not set.")
    if not GITHUB_TOKEN:
        print("Error: GITHUB_TOKEN environment variable not set.")

# Initialize LLM
llm = OpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL", "https://litellm.aks-hs-prod.int.hyperskill.org")
)

# -----------------------------
# GitHub Tool Functions
# -----------------------------

def get_pr_details(pr_number: int) -> Dict[str, Any]:
    """
    Get details about a pull request given its number.
    Returns author, title, body, diff_url, state, and commit SHAs.
    """
    if repo is None:
        return {"error": "GitHub repository not initialized or accessible."}
    try:
        pull_request = repo.get_pull(pr_number)
        commit_SHAs = [c.sha for c in pull_request.get_commits()]
        return {
            "author": pull_request.user.login,
            "title": pull_request.title,
            "body": pull_request.body,
            "diff_url": pull_request.diff_url,
            "state": pull_request.state,
            "commit_SHAs": commit_SHAs,
            "head_sha": pull_request.head.sha
        }
    except Exception as e:
        return {"error": f"Failed to fetch PR details for PR #{pr_number}: {str(e)}"}

def get_file_contents(file_path: str) -> str:
    """
    Fetch the contents of a file from the repository's default branch given its path.
    """
    if repo is None:
        return "Error: GitHub repository not initialized or accessible."
    try:
        file_content = repo.get_contents(file_path)
        return file_content.decoded_content.decode('utf-8')
    except Exception as e:
        return f"Error fetching file '{file_path}': {str(e)}"

def get_pr_commit_details(commit_sha: str) -> List[Dict[str, Any]]:
    """
    Get details about a specific commit including changed files and their diffs (patches).
    """
    if repo is None:
        return [{"error": "GitHub repository not initialized or accessible."}]
    try:
        commit = repo.get_commit(commit_sha)
        return [{
            "filename": f.filename,
            "status": f.status,
            "additions": f.additions,
            "deletions": f.deletions,
            "changes": f.changes,
            "patch": f.patch
        } for f in commit.files]
    except Exception as e:
        return [{"error": f"Failed to fetch commit details for SHA {commit_sha}: {str(e)}"}]

def post_review_to_github(pr_number: int, comment: str) -> str:
    """
    Post a review comment to a GitHub pull request.
    """
    if repo is None:
        return "Error: GitHub repository not initialized or accessible."
    try:
        pull_request = repo.get_pull(pr_number)
        pull_request.create_issue_comment(body=comment)
        return f"Comment posted successfully to PR #{pr_number}."
    except Exception as e:
        return f"Error posting review to PR #{pr_number}: {str(e)}"

# -----------------------------
# State Management Functions
# -----------------------------

async def add_context_to_state(ctx: Context, context: str) -> str:
    """Add gathered context information to the workflow state."""
    current_state = await ctx.store.get("state", default={})
    current_state["gathered_contexts"] = context
    await ctx.store.set("state", current_state)
    return "Context added to state."

async def add_comment_to_state(ctx: Context, draft_comment: str) -> str:
    """Add the draft review comment to the workflow state."""
    current_state = await ctx.store.get("state", default={})
    current_state["review_comment"] = draft_comment
    await ctx.store.set("state", current_state)
    return "Draft comment added to state."

async def add_final_review_to_state(ctx: Context, final_review: str) -> str:
    """Add the final review comment to the workflow state."""
    current_state = await ctx.store.get("state", default={})
    current_state["final_review_comment"] = final_review
    await ctx.store.set("state", current_state)
    return "Final review added to state."

# -----------------------------
# FunctionTool Conversions
# -----------------------------
tools = [
    FunctionTool.from_defaults(fn) for fn in [
        get_pr_details, get_file_contents, get_pr_commit_details,
        post_review_to_github, add_context_to_state, add_comment_to_state,
        add_final_review_to_state
    ]
]

# -----------------------------
# Agent Definitions
# -----------------------------
context_agent = FunctionAgent(
    llm=llm,
    name="ContextAgent",
    description="Gathers context for PR review including details, diffs, and files.",
    tools=[get_pr_details, get_file_contents, get_pr_commit_details, add_context_to_state],
    system_prompt="You are a context-gathering agent. You MUST gather PR details, changed files, and any other requested files. Once gathered, use `add_context_to_state` to save it and then hand control to the CommentorAgent.",
    can_handoff_to=["CommentorAgent"]
)

commentor_agent = FunctionAgent(
    llm=llm,
    name="CommentorAgent",
    description="Uses gathered context to draft a thorough PR review comment.",
    tools=[add_comment_to_state],
    system_prompt="You are a commentor agent. Write a ~200-300 word PR review in markdown. Address the author directly. Cover good points, rule adherence, tests, and documentation. Quote lines that need improvement and offer suggestions. Save your review using `add_comment_to_state` and then you MUST hand off to the ReviewAndPostingAgent. If you need more info, hand off to ContextAgent.",
    can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"]
)

review_and_posting_agent = FunctionAgent(
    llm=llm,
    name="ReviewAndPostingAgent",
    description="Reviews the draft comment for quality and posts the final review to GitHub.",
    tools=[add_final_review_to_state, post_review_to_github],
    system_prompt="You are the final reviewer and posting agent. Check the draft comment for quality and completeness. If it's inadequate, send it back to the CommentorAgent for a rewrite. If it's good, save it using `add_final_review_to_state`, post it to GitHub using `post_review_to_github`, and then conclude the workflow. Always extract the PR number from the initial user request to post the review.",
    can_handoff_to=["CommentorAgent"]
)

# -----------------------------
# Workflow Definition
# -----------------------------
workflow = AgentWorkflow(
    agents=[context_agent, commentor_agent, review_and_posting_agent],
    root_agent=context_agent.name,  # Start with ContextAgent to gather info first
    initial_state={"gathered_contexts": "", "review_comment": "", "final_review_comment": ""}
)

# -----------------------------
# Main Execution Function
# -----------------------------
async def main():
    """Main function to run the agent workflow non-interactively."""
    if not PR_NUMBER_STR:
        print("Error: PR_NUMBER environment variable is not set. Exiting.")
        return

    try:
        pr_number = int(PR_NUMBER_STR)
    except ValueError:
        print(f"Error: Invalid PR_NUMBER '{PR_NUMBER_STR}'. Must be an integer. Exiting.")
        return

    print(f"🔧 Environment Variables:")
    print(f"  - REPOSITORY: {REPOSITORY}")
    print(f"  - PR_NUMBER: {pr_number}")
    print(f"  - GITHUB_TOKEN: {'✅ Set' if GITHUB_TOKEN else '❌ Not set'}")
    print(f"  - OPENAI_API_KEY: {'✅ Set' if os.getenv('OPENAI_API_KEY') else '❌ Not set'}")

    # The initial query that starts the entire process
    query = f"Write and post a review for PR number {pr_number} in the repository {REPOSITORY}."
    print(f"🚀 Starting agent workflow with query: '{query}'")

    try:
        response = await workflow.arun(input=query)
        print("\n✅ Workflow finished.")
        print("Final response:", response)
    except Exception as e:
        print(f"\n❌ Workflow failed with error: {str(e)}")
        raise

# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    if repo:
        asyncio.run(main())
    else:
        print("Could not initialize repository. Agent will not run.")
    if git:
        git.close()
