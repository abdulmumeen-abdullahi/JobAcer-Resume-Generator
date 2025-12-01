from dotenv import load_dotenv
import os
import json
import time
import requests
from typing import List
from google.adk.tools import FunctionTool
from urllib.parse import urlparse
import asyncio
from jobparser_agent import run_job_parser
import re

# ----------------------------- Configuration -----------------------------
load_dotenv()
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise RuntimeError("GITHUB_TOKEN not set in environment")
print("GITHUB_TOKEN loaded successfully.")

CACHE_DIR = "./readme_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


# Extract github username
def extract_github_username(profile_input: str) -> str:
    """
    Accepts either a GitHub username or full profile URL and returns the username.
    """
    if profile_input.startswith("http://") or profile_input.startswith("https://"):
        parsed_url = urlparse(profile_input)
        path_parts = parsed_url.path.strip("/").split("/")
        if len(path_parts) >= 1 and path_parts[0]:
            return path_parts[0]
        else:
            raise ValueError(f"Invalid GitHub URL: {profile_input}")
    else:
        return profile_input.strip()

# ----------------------------- RepoFetcherAgent -----------------------------
class RepoFetcherAgent:
    def __init__(self, github_token=None, retry_attempts=5, backoff_factor=2):
        self.session = requests.Session()
        self.headers = {"Authorization": f"token {github_token}"} if github_token else {}
        self.cache_dir = CACHE_DIR
        self.retry_attempts = retry_attempts
        self.backoff_factor = backoff_factor

    # Cache helpers
    def _cache_file(self, owner: str, repo: str) -> str:
        return os.path.join(self.cache_dir, f"{owner.replace('/', '_')}_{repo.replace('/', '_')}.json")

    def _load_cache(self, owner: str, repo: str):
        path = self._cache_file(owner, repo)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _save_cache(self, owner: str, repo: str, content: str, etag: str):
        path = self._cache_file(owner, repo)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"content": content, "etag": etag}, f)

    # HTTP helper with retry
    def _get(self, url, headers=None):
        attempt = 0
        while attempt < self.retry_attempts:
            resp = self.session.get(url, headers=headers)
            if resp.status_code in [429, 500, 502, 503, 504]:
                time.sleep(self.backoff_factor ** attempt)
                attempt += 1
                continue
            resp.raise_for_status()
            return resp
        raise RuntimeError(f"Failed to GET {url} after {self.retry_attempts} attempts.")
    
    def extract_readme_title(self, md_text: str) -> str:
        """
        Extracts the first top-level README title (# Heading).
        Returns None if not found.
        """
        match = re.search(r'^\s*#\s+(.+)', md_text, flags=re.MULTILINE)
        if match:
            return match.group(1).strip()
        return None

    # GitHub API
    def list_repos(self, username):
        url = f"https://api.github.com/users/{username}/repos?per_page=100"
        return self._get(url, headers=self.headers).json()

    def get_readme(self, owner, repo):
        url = f"https://api.github.com/repos/{owner}/{repo}/readme"
        headers = self.headers.copy()
        headers["Accept"] = "application/vnd.github.raw+json"

        cached = self._load_cache(owner, repo)
        if cached and "etag" in cached:
            headers["If-None-Match"] = cached["etag"]

        resp = self.session.get(url, headers=headers)
        if resp.status_code == 304 and cached:
            return cached["content"]
        elif resp.status_code == 200:
            content = resp.text
            etag = resp.headers.get("ETag", "")
            if etag:
                self._save_cache(owner, repo, content, etag)
            return content
        return ""
    # Clean README content
    def clean_readme(self, md_text):
        # Remove image markdown ![alt](url)
        md_text = re.sub(r'!\[.*?\]\(.*?\)', '', md_text)
        # Remove badges like [![badge](url)](link)
        md_text = re.sub(r'\[!\[.*?\]\(.*?\)\]\(.*?\)', '', md_text)
        return md_text.strip()

    # Fetch repos using combined keywords/skills from parsed_job
    def fetch_repos(self, profile_input, parsed_job: dict, max_repos: int = 10, max_words: int = 300):
        username = extract_github_username(profile_input)
        keywords = [k.lower() for k in parsed_job.get("keywords", [])]
        skills = [s.lower() for s in parsed_job.get("skills", [])]
        search_terms = list(set(keywords + skills))

        repos = self.list_repos(username)
        results = []

        for repo in repos[:max_repos]:  # <-- limit number of repos
            owner = repo["owner"]["login"]
            repo_name = repo["name"]
            readme = self.get_readme(owner, repo_name) or ""
            combined_text = f"{repo_name} {readme}".lower()

            if search_terms and not any(term in combined_text for term in search_terms):
                continue

            cleaned_readme = self.clean_readme(readme)
            title = self.extract_readme_title(cleaned_readme) or repo_name

            # Limit the number of words per README
            limited_readme = " ".join(cleaned_readme.split()[:max_words])

            results.append({
                "name": title,
                "url": repo["html_url"],
                "readme": limited_readme
            })
            time.sleep(0.2)

            if len(results) >= max_repos:
                break

        return results


# ----------------------------- ADK FunctionTool -----------------------------
repo_fetcher = RepoFetcherAgent(GITHUB_TOKEN)

fetch_repos_tool = FunctionTool(
    func=lambda username, parsed_job={}: repo_fetcher.fetch_repos(username, parsed_job)
)



if __name__ == "__main__":

    profile_input = "https://github.com/abdulmumeen-abdullahi"
    job_description = """
    We are hiring a Junior Machine Learning Engineer.
    Responsibilities: data pipelines, model training, deployment, LLMs.
    Required: Python, TensorFlow or PyTorch, teamwork.
    """

    async def main():
        parsed_job = await run_job_parser(job_description)
        output = fetch_repos_tool.func(profile_input, parsed_job=parsed_job)
        print(json.dumps(output, indent=2))

    asyncio.run(main())
