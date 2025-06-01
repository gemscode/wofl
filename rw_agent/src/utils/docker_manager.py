import os
from jinja2 import Environment, FileSystemLoader
from uuid import UUID

class DockerManager:
    def __init__(self):
        self.template_env = Environment(
            loader=FileSystemLoader("src/templates/docker")
        )
        
    def generate_dockerfiles(self, project_id: UUID, agents: list):
        context = {
            'project_id': str(project_id),
            'agents': agents,
            'python_version': os.getenv("PYTHON_VERSION", "3.9")
        }
        
        # Generate Dockerfile
        template = self.template_env.get_template("Dockerfile.j2")
        dockerfile = template.render(context)
        
        # Write to project directory
        with open(f"projects/{project_id}/Dockerfile", "w") as f:
            f.write(dockerfile)
            
        return dockerfile

