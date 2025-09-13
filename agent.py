# -----------------------------
# GitHub Tool Functions
# -----------------------------
import os
import sys

# Initialize GitHub client
git = Github(os.getenv("GITHUB_TOKEN")) if os.getenv("GITHUB_TOKEN") else None

# Repository and PR configuration from environment variables
full_repo_name = os.getenv("REPOSITORY")
pr_number = os.getenv("PR_NUMBER")

# Get repository object if GitHub token is available
repo = None
if git is not None:
    try:
        repo = git.get_repo(full_repo_name)
    except Exception as e:
        print(f"Error getting repository: {e}", file=sys.stderr)
        sys.exit(1)

# ... (rest of the file content)

# -----------------------------
# Main async function for running the workflow
# -----------------------------
async def main():
    if not pr_number:
        print("PR_NUMBER environment variable not set. Exiting.")
        return

    query = f"Write a review for PR number {pr_number}."
    prompt = RichPromptTemplate(query)

    handler = workflow_agent.run(prompt.format())
    # ... (rest of the streaming logic)

# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    asyncio.run(main())
    if git:
        git.close()
