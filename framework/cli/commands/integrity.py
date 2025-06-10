# cli/commands/integrity.py
import click
from pathlib import Path
from tabulate import tabulate
from utils.sync import IntegrityManager
from utils.cassandra_manager import CassandraManager
from utils.config import load_project_config
from utils.check import check_project, format_report
import logging
import uuid

logger = logging.getLogger(__name__)

def get_agent_id(project_path: Path) -> str:
    """Retrieve agent ID from project config"""
    config_file = project_path / 'rwagent.json'
    if not config_file.exists():
        raise FileNotFoundError("Project config (rwagent.json) not found")

    config = load_project_config(config_file)
    return config.get('project_id')

@click.command()
@click.option('--check', is_flag=True, help='Verify database synchronization')
@click.option('--fix', is_flag=True, help='Update database with current state')
@click.option('--verbose', is_flag=True, help='Show detailed output')
@click.option('--sync', is_flag=True, help='Force database synchronization')
def integrity(check, fix, verbose, sync):
    """Manage project integrity and database synchronization"""
    base_path = Path.cwd()
    cm = CassandraManager()
    im = IntegrityManager(cm.session)

    try:
        agent_id = get_agent_id(base_path)
    except Exception as e:
        click.secho(f"❌ Error retrieving agent ID: {str(e)}", fg='red')
        return

    try:
        if sync:
            click.secho("🔄 Force-synchronizing with database...", fg='blue')
            im.update_sync_status(agent_id, base_path)
            click.secho("✅ Database synchronization completed", fg='green')
            return

        if check or fix:
            sync_status = im.check_sync_status(agent_id, base_path)

            if verbose:
                click.echo(f"🔍 Scan Details:")
                click.echo(f"- Agent ID: {agent_id}")
                click.echo(f"- Project Path: {base_path}")
                click.echo(f"- Total Files: {sync_status['total_files']}")
                click.echo(f"- Last Sync: {sync_status['last_sync'] or 'Never'}")

            if sync_status['discrepancies']:
                click.secho("\n⚠️  Out of Sync Components:", fg='yellow')
                for issue in sync_status['discrepancies']:
                    click.echo(f"  - {issue}")

                if fix:
                    if click.confirm(f"Update database with {len(sync_status['discrepancies'])} changes?"):
                        im.update_sync_status(agent_id, base_path)
                        click.secho("✅ Database sync updated successfully", fg='green')
            else:
                click.secho("\n✅ System is fully synchronized", fg='green')

        if not (check or fix or sync):
            results = check_project(base_path)
            click.echo(format_report(results))

    except Exception as e:
        logger.error(f"Integrity check failed: {str(e)}", exc_info=verbose)
        click.secho(f"❌ Operation failed: {str(e)}", fg='red')
        raise click.Abort()

@click.command()
def check_integrity():
    """Check project integrity and sync status"""
    ctx = click.get_current_context()
    ctx.invoke(integrity, check=True, fix=False)

@click.command()
@click.option('--sync', is_flag=True, help='Synchronize with database registry')
@click.option('--verbose', is_flag=True, help='Show detailed output')
def fix_integrity(sync, verbose):
    """Repair integrity issues and update database"""
    ctx = click.get_current_context()
    ctx.invoke(integrity, check=True, fix=True, sync=sync, verbose=verbose)

