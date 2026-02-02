"""
GitHub API Client
Handles all GitHub operations: branches, issues, commits, PRs
Supports corporate proxy configuration
"""

import os
import base64
import requests
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class GitHubConfig:
    """GitHub configuration"""
    token: str
    owner: str
    repo: str
    base_url: str = "https://api.github.com"
    proxy: Optional[str] = None  # e.g., "http://proxy.lilly.com:9000"
    verify_ssl: bool = False  # Disabled for corporate SSL inspection
    
    @classmethod
    def from_env(cls, repo_full_name: Optional[str] = None) -> "GitHubConfig":
        """Create config from environment variables"""
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GITHUB_PAT")
        if not token:
            raise ValueError("GITHUB_TOKEN or GITHUB_PAT environment variable required")
        
        repo_name = repo_full_name or os.environ.get("GITHUB_REPO", "")
        if "/" in repo_name:
            owner, repo = repo_name.split("/", 1)
        else:
            owner = os.environ.get("GITHUB_OWNER", "")
            repo = repo_name
        
        # Check for proxy settings
        proxy = (
            os.environ.get("HTTPS_PROXY") or 
            os.environ.get("https_proxy") or
            os.environ.get("HTTP_PROXY") or
            os.environ.get("http_proxy")
        )
        
        return cls(token=token, owner=owner, repo=repo, proxy=proxy)


class GitHubClient:
    """GitHub API client for agent operations"""
    
    def __init__(self, config: GitHubConfig):
        self.config = config
        self.headers = {
            "Authorization": f"Bearer {config.token}",
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        self.base_url = config.base_url
        self.repo_path = f"/repos/{config.owner}/{config.repo}"
        
        # Setup session with proxy if configured
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.session.verify = config.verify_ssl
        
        if config.proxy:
            self.session.proxies = {
                "http": config.proxy,
                "https": config.proxy
            }
    
    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make an API request with retry logic"""
        url = f"{self.base_url}{endpoint}"
        
        # Retry configuration
        max_retries = 3
        retry_delay = 1
        
        last_error = None
        for attempt in range(max_retries):
            try:
                response = self.session.request(method, url, timeout=30, **kwargs)
                
                if not response.ok:
                    error_msg = f"GitHub API error: {response.status_code}"
                    try:
                        error_data = response.json()
                        error_msg += f" - {error_data.get('message', '')}"
                    except:
                        error_msg += f" - {response.text}"
                    raise Exception(error_msg)
                
                if response.status_code == 204:
                    return {}
                return response.json()
                
            except requests.exceptions.ConnectionError as e:
                last_error = e
                if attempt < max_retries - 1:
                    import time
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                continue
            except requests.exceptions.Timeout as e:
                last_error = e
                if attempt < max_retries - 1:
                    import time
                    time.sleep(retry_delay)
                    continue
        
        # If we get here, all retries failed
        raise Exception(f"Connection failed after {max_retries} attempts: {last_error}")
    
    # =========================================================================
    # Repository Info
    # =========================================================================
    
    def get_repo_info(self) -> Dict[str, Any]:
        """Get repository information"""
        return self._request("GET", self.repo_path)
    
    def get_default_branch(self) -> str:
        """Get the default branch name"""
        repo = self.get_repo_info()
        return repo.get("default_branch", "main")
    
    # =========================================================================
    # Branches
    # =========================================================================
    
    def get_branch(self, branch_name: str) -> Dict[str, Any]:
        """Get branch information"""
        return self._request("GET", f"{self.repo_path}/branches/{branch_name}")
    
    def get_branch_sha(self, branch_name: str) -> str:
        """Get the SHA of a branch's HEAD"""
        branch = self.get_branch(branch_name)
        return branch["commit"]["sha"]
    
    def create_branch(self, branch_name: str, from_branch: Optional[str] = None) -> Dict[str, Any]:
        """Create a new branch"""
        if from_branch is None:
            from_branch = self.get_default_branch()
        
        sha = self.get_branch_sha(from_branch)
        
        return self._request(
            "POST",
            f"{self.repo_path}/git/refs",
            json={
                "ref": f"refs/heads/{branch_name}",
                "sha": sha
            }
        )
    
    def branch_exists(self, branch_name: str) -> bool:
        """Check if a branch exists"""
        try:
            self.get_branch(branch_name)
            return True
        except:
            return False
    
    def delete_branch(self, branch_name: str) -> None:
        """Delete a branch"""
        self._request("DELETE", f"{self.repo_path}/git/refs/heads/{branch_name}")
    
    # =========================================================================
    # Issues
    # =========================================================================
    
    def create_issue(
        self,
        title: str,
        body: str,
        labels: Optional[List[str]] = None,
        assignees: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Create a new issue"""
        data = {"title": title, "body": body}
        if labels:
            data["labels"] = labels
        if assignees:
            data["assignees"] = assignees
        
        return self._request("POST", f"{self.repo_path}/issues", json=data)
    
    def get_issue(self, issue_number: int) -> Dict[str, Any]:
        """Get an issue by number"""
        return self._request("GET", f"{self.repo_path}/issues/{issue_number}")
    
    def close_issue(self, issue_number: int) -> Dict[str, Any]:
        """Close an issue"""
        return self._request(
            "PATCH",
            f"{self.repo_path}/issues/{issue_number}",
            json={"state": "closed"}
        )
    
    def add_issue_comment(self, issue_number: int, body: str) -> Dict[str, Any]:
        """Add a comment to an issue"""
        return self._request(
            "POST",
            f"{self.repo_path}/issues/{issue_number}/comments",
            json={"body": body}
        )
    
    # =========================================================================
    # Files & Commits
    # =========================================================================
    
    def get_file_content(self, path: str, branch: Optional[str] = None) -> Dict[str, Any]:
        """Get file content from repository"""
        params = {}
        if branch:
            params["ref"] = branch
        return self._request("GET", f"{self.repo_path}/contents/{path}", params=params)
    
    def create_or_update_file(
        self,
        path: str,
        content: str,
        message: str,
        branch: str,
        sha: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create or update a file in the repository"""
        encoded_content = base64.b64encode(content.encode()).decode()
        
        data = {
            "message": message,
            "content": encoded_content,
            "branch": branch
        }
        
        if sha:
            data["sha"] = sha
        else:
            try:
                existing = self.get_file_content(path, branch)
                data["sha"] = existing["sha"]
            except:
                pass
        
        return self._request("PUT", f"{self.repo_path}/contents/{path}", json=data)
    
    def create_files_batch(
        self,
        files: Dict[str, str],
        message: str,
        branch: str
    ) -> Dict[str, Any]:
        """Create multiple files in a single commit using Git Data API"""
        branch_sha = self.get_branch_sha(branch)
        commit = self._request("GET", f"{self.repo_path}/git/commits/{branch_sha}")
        base_tree_sha = commit["tree"]["sha"]
        
        tree_items = []
        for path, content in files.items():
            blob = self._request(
                "POST",
                f"{self.repo_path}/git/blobs",
                json={"content": content, "encoding": "utf-8"}
            )
            tree_items.append({
                "path": path,
                "mode": "100644",
                "type": "blob",
                "sha": blob["sha"]
            })
        
        new_tree = self._request(
            "POST",
            f"{self.repo_path}/git/trees",
            json={"base_tree": base_tree_sha, "tree": tree_items}
        )
        
        new_commit = self._request(
            "POST",
            f"{self.repo_path}/git/commits",
            json={
                "message": message,
                "tree": new_tree["sha"],
                "parents": [branch_sha]
            }
        )
        
        self._request(
            "PATCH",
            f"{self.repo_path}/git/refs/heads/{branch}",
            json={"sha": new_commit["sha"]}
        )
        
        return new_commit
    
    # =========================================================================
    # Pull Requests
    # =========================================================================
    
    def create_pull_request(
        self,
        title: str,
        body: str,
        head_branch: str,
        base_branch: Optional[str] = None,
        draft: bool = False
    ) -> Dict[str, Any]:
        """Create a pull request"""
        if base_branch is None:
            base_branch = self.get_default_branch()
        
        return self._request(
            "POST",
            f"{self.repo_path}/pulls",
            json={
                "title": title,
                "body": body,
                "head": head_branch,
                "base": base_branch,
                "draft": draft
            }
        )
    
    def get_pull_request(self, pr_number: int) -> Dict[str, Any]:
        """Get a pull request by number"""
        return self._request("GET", f"{self.repo_path}/pulls/{pr_number}")
    
    def list_pull_requests(
        self,
        state: str = "open",
        head: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List pull requests"""
        params = {"state": state}
        if head:
            params["head"] = f"{self.config.owner}:{head}"
        return self._request("GET", f"{self.repo_path}/pulls", params=params)
    
    def add_pr_comment(self, pr_number: int, body: str) -> Dict[str, Any]:
        """Add a comment to a pull request"""
        return self.add_issue_comment(pr_number, body)


# =========================================================================
# Helper function for quick setup
# =========================================================================

def create_github_client(repo: Optional[str] = None, proxy: Optional[str] = None) -> GitHubClient:
    """Create a GitHub client from environment"""
    config = GitHubConfig.from_env(repo)
    if proxy:
        config.proxy = proxy
    return GitHubClient(config)


# =========================================================================
# Connection Test
# =========================================================================

def test_connection():
    """Test GitHub connection with detailed diagnostics"""
    
    # Suppress SSL warnings for corporate environments
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    print("\n" + "="*60)
    print("🔍 GitHub Connection Diagnostics")
    print("="*60)
    
    # Check environment
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GITHUB_PAT")
    repo = os.environ.get("GITHUB_REPO", "")
    proxy = (
        os.environ.get("HTTPS_PROXY") or 
        os.environ.get("https_proxy") or
        os.environ.get("HTTP_PROXY") or
        os.environ.get("http_proxy")
    )
    
    print("\n📋 Environment Variables:")
    print(f"   GITHUB_TOKEN: {'✅ Set' if token else '❌ Not set'}")
    print(f"   GITHUB_REPO: {repo if repo else '❌ Not set'}")
    print(f"   Proxy: {proxy if proxy else '⚠️  Not set (may need for corporate network)'}")
    
    if not token:
        print("\n❌ Cannot test without GITHUB_TOKEN")
        return False
    
    # Test basic connectivity first (with SSL verification disabled for corporate)
    print("\n🌐 Testing basic internet connectivity...")
    print("   (SSL verification disabled for corporate network)")
    try:
        test_resp = requests.get("https://www.google.com", timeout=10, verify=False)
        print("   ✅ Internet connection OK")
    except Exception as e:
        print(f"   ❌ No internet: {e}")
        print("\n💡 You may need to set proxy:")
        print("   $env:HTTPS_PROXY = 'http://your-proxy:port'")
        return False
    
    # Test GitHub API
    print("\n🐙 Testing GitHub API connectivity...")
    
    # Try without proxy first (SSL verification disabled)
    try:
        resp = requests.get(
            "https://api.github.com",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
            verify=False
        )
        if resp.ok:
            print("   ✅ GitHub API accessible")
        else:
            print(f"   ⚠️  GitHub responded with: {resp.status_code}")
    except requests.exceptions.ConnectionError as e:
        print(f"   ❌ Connection failed: {type(e).__name__}")
        print("\n💡 This usually means corporate firewall is blocking.")
        print("   Try setting proxy:")
        print("   $env:HTTPS_PROXY = 'http://proxy.lilly.com:9000'")
        
        # Try with common Lilly proxy
        print("\n🔄 Trying with common corporate proxy...")
        common_proxies = [
            "http://proxy.lilly.com:9000",
            "http://webproxy.lilly.com:9000",
            "http://proxy.elililly.com:9000",
        ]
        
        for test_proxy in common_proxies:
            try:
                print(f"   Trying {test_proxy}...")
                resp = requests.get(
                    "https://api.github.com",
                    headers={"Authorization": f"Bearer {token}"},
                    proxies={"https": test_proxy},
                    timeout=10,
                    verify=False
                )
                if resp.ok:
                    print(f"   ✅ Works with proxy: {test_proxy}")
                    print(f"\n💡 Set this proxy:")
                    print(f"   $env:HTTPS_PROXY = '{test_proxy}'")
                    return True
            except:
                continue
        
        print("   ❌ Common proxies didn't work")
        print("\n💡 Ask your IT team for the correct proxy URL")
        return False
    
    # Test repo access if configured
    if repo and "/" in repo:
        print(f"\n📦 Testing repository access: {repo}")
        try:
            owner, repo_name = repo.split("/")
            config = GitHubConfig(token=token, owner=owner, repo=repo_name, proxy=proxy)
            client = GitHubClient(config)
            info = client.get_repo_info()
            print(f"   ✅ Repository accessible: {info['full_name']}")
            print(f"   Default branch: {info['default_branch']}")
            return True
        except Exception as e:
            print(f"   ❌ Repository error: {e}")
            return False
    
    return True


if __name__ == "__main__":
    test_connection()