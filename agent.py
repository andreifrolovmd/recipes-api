import dotenv

dotenv.load_dotenv()

llm = OpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
)
github_client = Github(os.getenv("GITHUB_TOKEN")) if os.getenv("GITHUB_TOKEN") else Github()
repository = os.getenv("REPOSITORY")
pr_number = os.getenv("PR_NUMBER")
query = "Write a review for PR: " + pr_number
