from crewai import Agent, Task, Crew
from typing import List
from src.utils import CassandraManager, GitManager, DockerManager

class ProjectOrchestrator:
    def __init__(self, project_spec: dict):
        self.project_spec = project_spec
        self.cassandra = CassandraManager()
        self.git = GitManager()
        self.docker = DockerManager()
        
        self.agents = [
            Agent(
                role="System Architect",
                goal="Ensure project structure and requirements",
                backstory="Expert in building maintainable system architectures",
                verbose=True
            ),
            Agent(
                role="Security Engineer",
                goal="Implement security best practices",
                backstory="Security specialist with focus on secure coding",
                verbose=True
            )
        ]
        
    def _create_project_task(self):
        return Task(
            description=f"Create project {self.project_spec['name']}",
            expected_output="Full project structure with core files",
            agent=self.agents[0],
            async_execution=False,
            human_input=False
        )
        
    def _security_setup_task(self):
        return Task(
            description="Configure security settings",
            expected_output="Security configuration files",
            agent=self.agents[1],
            context=[self._create_project_task()],
            async_execution=False
        )
        
    def orchestrate(self):
        crew = Crew(
            agents=self.agents,
            tasks=[self._create_project_task(), self._security_setup_task()],
            verbose=2
        )
        
        result = crew.kickoff()
        
        # Post-processing
        project_id = self.cassandra.create_project(
            self.project_spec['user_id'],
            self.project_spec['name'],
            self.project_spec['agents']
        )
        
        self.git.init_repo(project_id)
        self.docker.generate_dockerfiles(project_id, self.project_spec['agents'])
        
        return {
            "project_id": str(project_id),
            "status": "created",
            "agents": self.project_spec['agents']
        }

