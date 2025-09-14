import os
import sys
import asyncio
from typing import Dict, List, Any
from github import Github
import openai

# -----------------------------
# Load Environment Variables and Command Line Arguments
# -----------------------------
try:
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    pass

# Handle both environment variables and command line arguments
if len(sys.argv) >= 6:
    # Command line arguments provided (for compatibility)
    github_token = sys.argv[1]
    repository = sys.argv[2]
    pr_number = sys.argv[3]
    openai_api_key = sys.argv[4]
    openai_base_url = sys.argv[5]
else:
    # Get configuration from environment variables (preferred method)
    github_token = os.getenv("GITHUB_TOKEN")
    repository = os.getenv("REPOSITORY")
    pr_number = os.getenv("PR_NUMBER")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_base_url = os.getenv("OPENAI_BASE_URL")

# Validate required environment variables
if not github_token:
    print("Error: GITHUB_TOKEN environment variable not set.")
    sys.exit(1)

if not repository:
    print("Error: REPOSITORY environment variable not set.")
    sys.exit(1)

if not openai_api_key:
    print("Error: OPENAI_API_KEY environment variable not set.")
    sys.exit(1)

# Extract repository parts
username, repo_name = repository.split('/')
full_repo_name = repository

# Initialize GitHub client
try:
    git = Github(github_token)
    repo = git.get_repo(full_repo_name)
    print(f"Successfully connected to repository: {full_repo_name}")
except Exception as e:
    print(f"Error: Could not get repository '{full_repo_name}'. Details: {e}")
    sys.exit(1)

# Validate PR number
if not pr_number or not str(pr_number).isdigit():
    print("Error: Pull request number not provided or invalid.")
    sys.exit(1)

pr_number = int(pr_number)

# Setup OpenAI client
openai.api_key = openai_api_key
if openai_base_url:
    openai.api_base = openai_base_url

# -----------------------------
# Simple PR Review Functions
# -----------------------------

def get_pr_details(pr_num: int) -> Dict[str, Any]:
    """Get details about a pull request."""
    try:
        pull_request = repo.get_pull(pr_num)
        return {
            "author": pull_request.user.login,
            "title": pull_request.title,
            "body": pull_request.body or "No description provided",
            "state": pull_request.state,
        }
    except Exception as e:
        return {"error": f"Failed to fetch PR details: {str(e)}"}

def get_pr_files(pr_num: int) -> List[Dict[str, Any]]:
    """Get changed files in a pull request."""
    try:
        pull_request = repo.get_pull(pr_num)
        files = []
        for file in pull_request.get_files():
            files.append({
                "filename": file.filename,
                "status": file.status,
                "additions": file.additions,
                "deletions": file.deletions,
                "patch": file.patch[:1000] if file.patch else "No patch available"  # Limit patch size
            })
        return files
    except Exception as e:
        return [{"error": f"Failed to fetch files: {str(e)}"}]

def generate_review_comment(pr_details: Dict, files: List[Dict]) -> str:
    """Generate a review comment using OpenAI."""
    try:
        # Prepare context for the review
        context = f"""
PR Title: {pr_details.get('title', 'N/A')}
Author: {pr_details.get('author', 'N/A')}
Description: {pr_details.get('body', 'No description')}

Changed Files:
"""
        for file in files[:5]:  # Limit to first 5 files
            if "error" not in file:
                context += f"- {file['filename']} ({file['status']}, +{file['additions']} -{file['deletions']})\n"

        prompt = f"""Please write a concise code review comment (200-300 words) for this pull request:

{context}

Your review should include:
1. What's good about the PR
2. Any potential issues or improvements
3. Whether tests are needed for new functionality
4. Documentation suggestions if applicable

Write the review in a helpful, constructive tone addressing the author directly."""

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful code reviewer."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )

        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating review: {str(e)}"

def post_review_comment(pr_num: int, comment: str) -> str:
    """Post a review comment to GitHub."""
    try:
        pull_request = repo.get_pull(pr_num)
        comment_obj = pull_request.create_issue_comment(body=comment)
        return f"Comment posted successfully to PR #{pr_num}. Comment ID: {comment_obj.id}"
    except Exception as e:
        return f"Error posting comment: {str(e)}"

# -----------------------------
# Main Function
# -----------------------------
def main():
    print(f"Environment Variables:")
    print(f"  - REPOSITORY: {repository}")
    print(f"  - PR_NUMBER: {pr_number}")
    print(f"  - GITHUB_TOKEN: {'Set' if github_token else 'Not set'}")
    print(f"  - OPENAI_API_KEY: {'Set' if openai_api_key else 'Not set'}")

    print(f"Starting PR review for PR #{pr_number} in {full_repo_name}")

    # Get PR details
    print("Fetching PR details...")
    pr_details = get_pr_details(pr_number)
    if "error" in pr_details:
        print(f"Error: {pr_details['error']}")
        return

    # Get changed files
    print("Fetching changed files...")
    files = get_pr_files(pr_number)

    # Generate review comment
    print("Generating review comment...")
    comment = generate_review_comment(pr_details, files)

    # Post the comment
    print("Posting review comment...")
    result = post_review_comment(pr_number, comment)
    print(f"Result: {result}")

    print("PR review completed successfully!")

if __name__ == "__main__":
    main()
    if 'git' in locals():
        git.close()