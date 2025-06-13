import click
from pathlib import Path

@click.command()
@click.option('--key', required=True, help='Configuration key to update')
@click.option('--value', required=True, help='New value for the key')
@click.pass_context
def update_config(ctx, key, value):
    """Update project configuration"""
    project_path = Path(ctx.obj.get('project_path', '.'))
    config_file = project_path / 'rwagent.json'
    
    if not config_file.exists():
        raise click.ClickException("Not an RW Agent project (rwagent.json not found)")
    
    try:
        import json
        with open(config_file) as f:
            config = json.load(f)
        
        config[key] = value
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
            
        click.secho(f"Updated {key} successfully!", fg='green')
    except Exception as e:
        raise click.ClickException(f"Config update failed: {str(e)}")

