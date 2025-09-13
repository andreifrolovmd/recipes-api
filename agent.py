import os
import dotenv
import asyncio
from typing import Dict, List, Any
from github import Github

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

# Initialize GitHub client
git = Github(os.getenv("GITHUB_TOKEN")) if os.getenv("GITHUB_TOKEN") else None

# Repository configuration - get from environment variables
repository = os.getenv("REPOSITORY")  # Format: username/repo-name
pr_number = os.getenv("PR_NUMBER")

if not repository:
    raise ValueError("REPOSITORY environment variable is required")
if not pr_number:
    raise ValueError("PR_NUMBER environment variable is required")

# Get repository object if GitHub token is available
repo = None
if git is not None:
    repo = git.get_repo(repository)

# -----------------------------
# Setup LLM
# -----------------------------
llm = OpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL")
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
        return {"error": "GitHub repository not initialized"}

    try:
        # Get the pull request
        pull_request = repo.get_pull(pr_number)

        # Get commit SHAs
        commit_SHAs = []
        commits = pull_request.get_commits()

        for c in commits:
            commit_SHAs.append(c.sha)

        # Get the head SHA (last commit)
        head_sha = pull_request.head.sha if pull_request.head else None

        # Create PR details dictionary
        pr_details = {
            "user": pull_request.user.login,
            "author": pull_request.user.login,  # Include both for compatibility
            "title": pull_request.title,
            "body": pull_request.body,
            "diff_url": pull_request.diff_url,
            "state": pull_request.state,
            "commit_SHAs": commit_SHAs,
            "head_sha": head_sha
        }

        return pr_details

    except Exception as e:
        return {"error": f"Failed to fetch PR details: {str(e)}"}

def get_file_contents(file_path: str) -> str:
    """
    Fetch the contents of a file from the repository given its path.
    """
    if repo is None:
        return "Error: GitHub repository not initialized"

    try:
        # Get file contents from the default branch
        file_content = repo.get_contents(file_path)

        # Decode and return the content
        if file_content.encoding == "base64":
            return file_content.decoded_content.decode('utf-8')
        else:
            return file_content.content

    except Exception as e:
        return f"Error fetching file: {str(e)}"

def get_pr_commit_details(commit_sha: str) -> List[Dict[str, Any]]:
    """
    Get details about a specific commit including changed files and their diffs.
    Returns a list of changed files directly.
    """
    if repo is None:
        return [{"error": "GitHub repository not initialized"}]

    try:
        # Get the commit
        commit = repo.get_commit(commit_sha)

        # Get changed files information
        changed_files: List[Dict[str, Any]] = []
        for f in commit.files:
            changed_files.append({
                "filename": f.filename,
                "status": f.status,
                "additions": f.additions,
                "deletions": f.deletions,
                "changes": f.changes,
                "patch": f.patch
            })

        return changed_files

    except Exception as e:
        return [{"error": f"Failed to fetch commit details: {str(e)}"}]

def post_review_to_github(pr_number: int, comment: str) -> str:
    """
    Post a review comment to a GitHub pull request.
    """
    if repo is None:
        return "Error: GitHub repository not initialized"

    try:
        # Get the pull request
        pull_request = repo.get_pull(pr_number)

        # Try both methods to ensure the comment is posted
        try:
            # First try creating a review
            review = pull_request.create_review(body=comment)
            return f"Review posted successfully to PR #{pr_number}. Review ID: {review.id}"
        except Exception as review_error:
            # If review fails, try issue comment
            comment_obj = pull_request.create_issue_comment(body=comment)
            return f"Comment posted successfully to PR #{pr_number}. Comment ID: {comment_obj.id}"

    except Exception as e:
        return f"Error posting review: {str(e)}"

# -----------------------------
# State Management Functions
# -----------------------------

async def add_context_to_state(ctx: Context, context: str) -> str:
    """
    Add gathered context information to the state.
    """
    try:
        current_state = await ctx.store.get("state", default={})
        current_state["gathered_contexts"] = context
        await ctx.store.set("state", current_state)
        return "Context added to state successfully."
    except Exception as e:
        return f"Error adding context to state: {str(e)}"

async def add_comment_to_state(ctx: Context, draft_comment: str) -> str:
    """
    Add draft comment to the state.
    """
    try:
        current_state = await ctx.store.get("state", default={})
        current_state["review_comment"] = draft_comment
        await ctx.store.set("state", current_state)
        return "Draft comment added to state successfully."
    except Exception as e:
        return f"Error adding comment to state: {str(e)}"

async def add_final_review_to_state(ctx: Context, final_review: str) -> str:
    """
    Add final review to the state.
    """
    try:
        current_state = await ctx.store.get("state", default={})
        current_state["final_review_comment"] = final_review
        await ctx.store.set("state", current_state)
        return "Final review added to state successfully."
    except Exception as e:
        return f"Error adding final review to state: {str(e)}"

# -----------------------------
# Convert functions to tools
# -----------------------------
pr_details_tool = FunctionTool.from_defaults(
    get_pr_details,
    name="get_pr_details"
)

file_contents_tool = FunctionTool.from_defaults(
    get_file_contents,
    name="get_file_contents"
)

pr_commit_details_tool = FunctionTool.from_defaults(
    get_pr_commit_details,
    name="get_pr_commit_details"
)

add_context_to_state_tool = FunctionTool.from_defaults(
    add_context_to_state,
    name="add_context_to_state"
)

add_comment_to_state_tool = FunctionTool.from_defaults(
    add_comment_to_state,
    name="add_comment_to_state"
)

add_final_review_to_state_tool = FunctionTool.from_defaults(
    add_final_review_to_state,
    name="add_final_review_to_state"
)

post_review_to_github_tool = FunctionTool.from_defaults(
    post_review_to_github,
    name="post_review_to_github"
)

# -----------------------------
# Create the ContextAgent with FunctionAgent
# -----------------------------
context_system_prompt = """You are the context gathering agent. When gathering context, you MUST gather:
- The details: author, title, body, diff_url, state, and head_sha;
- Changed files;
- Any requested for files;
Once you gather the requested info, use add_context_to_state to save it, then you MUST hand control back to the CommentorAgent."""

context_agent = FunctionAgent(
    llm=llm,
    name="ContextAgent",
    description="Gathers all needed context for PR review including details, diffs, and files.",
    tools=[pr_details_tool, file_contents_tool, pr_commit_details_tool, add_context_to_state_tool],
    system_prompt=context_system_prompt,
    can_handoff_to=["CommentorAgent"]
)

# -----------------------------
# Create the CommentorAgent
# -----------------------------
commentor_system_prompt = """You are the commentor agent that writes review comments for pull requests as a human reviewer would.
Ensure to do the following for a thorough review:
 - Request for the PR details, changed files, and any other repo files you may need from the ContextAgent.
 - Once you have asked for all the needed information, write a good ~200-300 word review in markdown format detailing:
    - What is good about the PR?
    - Did the author follow ALL contribution rules? What is missing?
    - Are there tests for new functionality? If there are new models, are there migrations for them? - use the diff to determine this.
    - Are new endpoints documented? - use the diff to determine this.
    - Which lines could be improved upon? Quote these lines and offer suggestions the author could implement.
 - Use add_comment_to_state to save your review.
 - **Once you have successfully saved the review, you MUST hand off to the ReviewAndPostingAgent to finalize and post the review.**
 - If you need any additional details, you must hand off to the ContextAgent.
 - You should directly address the author. So your comments should sound like:
 "Thanks for fixing this. I think all places where we call quote should be fixed. Can you roll this fix out everywhere?" """

commentor_agent = FunctionAgent(
    llm=llm,
    name="CommentorAgent",
    description="Uses the context gathered by the context agent to draft a pull review comment.",
    tools=[add_comment_to_state_tool],
    system_prompt=commentor_system_prompt,
    can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"]
)

# -----------------------------
# Create the ReviewAndPostingAgent
# -----------------------------
review_and_posting_system_prompt = """You are the Review and Posting agent. You coordinate the entire review process and ensure reviews are posted to GitHub.

Your responsibilities:
1. If no review comment exists in the state, request the CommentorAgent to create one.
2. Once a review is generated, run a final check to ensure it meets these criteria:
   - Be a ~200-300 word review in markdown format
   - Specify what is good about the PR
   - Check if the author followed ALL contribution rules and note what is missing
   - Include notes on test availability for new functionality
   - Include notes on whether new endpoints are documented  
   - Include suggestions on which lines could be improved with quoted examples

3. If the review does not meet these criteria, ask the CommentorAgent to rewrite and address the concerns.
4. When satisfied with the review, use add_final_review_to_state to save it, then post it to GitHub using post_review_to_github.
5. Always extract the PR number from the user's request to post the review."""

review_and_posting_agent = FunctionAgent(
    llm=llm,
    name="ReviewAndPostingAgent",
    description="Reviews the draft comment and posts the final review to GitHub.",
    tools=[add_final_review_to_state_tool, post_review_to_github_tool],
    system_prompt=review_and_posting_system_prompt,
    can_handoff_to=["CommentorAgent"]
)

# -----------------------------
# Create the Workflow
# -----------------------------
workflow_agent = AgentWorkflow(
    agents=[context_agent, commentor_agent, review_and_posting_agent],
    root_agent=review_and_posting_agent.name,
    initial_state={
        "gathered_contexts": "",
        "review_comment": "",
        "final_review_comment": ""
    },
)

# -----------------------------
# Main async function for running the workflow
# -----------------------------
async def main():
    # Get PR number from environment and construct query
    query = f"Write a review for PR number {pr_number}"
    prompt = RichPromptTemplate(query)

    handler = workflow_agent.run(prompt.format())

    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("\n\nFinal response:", event.response.content)
            if event.tool_calls:
                print("Selected tools:", [call.tool_name for call in event.tool_calls])
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")

# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Error running workflow: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if git:
            git.close()
