import git
import os
from uuid import UUID

class GitManager:
    def __init__(self):
        self.base_path = os.getenv("GIT_BASE_PATH", "/repos")
        
    def init_repo(self, project_id: UUID):
        repo_path = f"{self.base_path}/{project_id}"
        repo = git.Repo.init(repo_path)
        
        # Create basic .gitignore
        with open(f"{repo_path}/.gitignore", "w") as f:
            f.write("__pycache__/\n*.pyc\n.env\n")
            
        repo.index.add([".gitignore"])
        repo.index.commit("Initial commit")
        return repo_path

    def commit_version(self, project_id: UUID, message: str):
        repo = git.Repo(f"{self.base_path}/{project_id}")
        repo.git.add(A=True)
        repo.index.commit(message)
        return repo.head.commit.hexsha

