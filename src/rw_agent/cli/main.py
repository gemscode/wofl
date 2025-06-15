# src/agents/agent_core/cli/main.py
import click
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
from rw_agent.cli.commands import init, deploy, config, integrity

def get_version():
    """Get package version safely"""
    try:
        return version('rwagent')
    except PackageNotFoundError:
        return "0.1.0-dev"

@click.group()
@click.version_option(get_version())
@click.option('--verbose', is_flag=True, help='Enable verbose output')
@click.option('--config-file', type=click.Path(), help='Specify config file path')
@click.pass_context
def cli(ctx, verbose, config_file):
    """🚀 RW Agent Framework CLI
    
    A powerful framework for building AI-driven applications with
    modular agents, storage, and deployment capabilities.
    """
    ctx.ensure_object(dict)
    ctx.obj['VERBOSE'] = verbose
    ctx.obj['CONFIG_FILE'] = config_file
    
    if verbose:
        click.echo("🔍 Verbose mode enabled")

# Add core commands
cli.add_command(init.init_project)
cli.add_command(deploy.deploy_service)

# Conditionally add config command if available
if hasattr(config, 'update_config'):
    cli.add_command(config.update_config)
else:
    click.echo("⚠️  Config module not fully implemented", err=True)

# Add integrity commands
cli.add_command(integrity.integrity)
cli.add_command(integrity.check_integrity)
cli.add_command(integrity.fix_integrity)

# Add informational commands
@cli.command()
def list_agents():
    """List all available agents and their status"""
    agents = [
        ('agent_core', 'Core functionality (required)', '✅ Always enabled'),
        ('agent_security', 'Authentication & authorization (required)', '✅ Always enabled'),
        ('agent_storage', 'Cassandra & Elasticsearch integration', '📦 Optional'),
        ('agent_ai', 'LLM & RL capabilities', '🤖 Optional'),
        ('agent_ui', 'Web interfaces and dashboards', '🖥️  Optional'),
        ('agent_deployment', 'CI/CD automation', '🚀 Optional'),
        ('agent_docker', 'Container management', '🐳 Optional'),
        ('agent_kafka', 'Event streaming', '📨 Optional'),
        ('agent_redis', 'Caching & queues', '⚡ Optional'),
        ('agent_kubernetes', 'Orchestration', '☸️  Optional')
    ]
    
    click.echo("\n📋 Available RW Agents:\n")
    for name, description, status in agents:
        click.echo(f"  {status} {click.style(name, fg='cyan', bold=True)}")
        click.echo(f"      {description}\n")

@cli.command()
@click.argument('project_dir', type=click.Path(exists=True))
def status(project_dir):
    """Show project status and configuration"""
    project_path = Path(project_dir)
    config_file = project_path / 'rwagent.json'
    
    if not config_file.exists():
        click.echo("❌ Not an RW Agent project (rwagent.json not found)")
        return
    
    import json
    config = json.loads(config_file.read_text())
    
    click.echo(f"\n📊 Project Status: {click.style(config['name'], fg='cyan', bold=True)}")
    click.echo(f"🆔 ID: {config['project_id']}")
    click.echo(f"📦 Version: {config.get('version', 'Unknown')}")
    click.echo(f"🔧 Agents: {', '.join(config['agents'])}")
    
    # Check agent directories
    click.echo("\n📁 Agent Status:")
    for agent in config['agents']:
        agent_path = project_path / 'src' / 'agents' / agent
        status_icon = "✅" if agent_path.exists() else "❌"
        click.echo(f"  {status_icon} {agent}")

if __name__ == "__main__":
    cli(auto_envvar_prefix='RWAGENT')

