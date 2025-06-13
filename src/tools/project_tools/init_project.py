import argparse
from uuid import UUID, uuid4
from src.crew.orchestrator import ProjectOrchestrator
from src.utils.cassandra_manager import CassandraManager

def main():
    parser = argparse.ArgumentParser(description="Initialize new project")
    parser.add_argument("--name", required=True, help="Project name")
    parser.add_argument("--user", required=True, help="User ID")
    parser.add_argument("--agents", nargs="+", default=[], help="Selected agents")
    
    args = parser.parse_args()
    
    # Enforce required agents
    required_agents = {'agent_core', 'agent_security'}
    selected_agents = set(args.agents) | required_agents
    
    # Init orchestrator
    orchestrator = ProjectOrchestrator({
        "name": args.name,
        "user_id": UUID(args.user),
        "agents": list(selected_agents)
    })
    
    result = orchestrator.orchestrate()
    
    print(f"""
    Project initialized successfully!
    ID: {result['project_id']}
    Agents: {', '.join(result['agents'])}
    """)

if __name__ == "__main__":
    main()

