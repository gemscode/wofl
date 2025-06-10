CREATE KEYSPACE IF NOT EXISTS rw_agent 
WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};

USE rw_agent;

-- Agent Registry
CREATE TABLE IF NOT EXISTS agent_registry (
    agent_id UUID PRIMARY KEY,
    name TEXT,
    description TEXT,
    version TEXT,
    status TEXT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

-- Agent Metadata
CREATE TABLE IF NOT EXISTS agent_metadata (
    agent_id UUID,
    type TEXT,
    key TEXT,
    value TEXT,
    last_updated TIMESTAMP,
    PRIMARY KEY ((agent_id, type), key)
);

-- Project Metadata
CREATE TABLE IF NOT EXISTS project_metadata (
    project_id UUID PRIMARY KEY,
    project_name TEXT,
    created_at TIMESTAMP,
    last_updated TIMESTAMP,
    dependencies MAP<TEXT, TEXT>
);

-- Agent Storage
CREATE TABLE IF NOT EXISTS agent_storage (
    storage_id UUID PRIMARY KEY,
    agent_id UUID,
    content TEXT,
    content_hash TEXT,
    created_at TIMESTAMP,
    last_accessed TIMESTAMP
) WITH compaction = {'class': 'LeveledCompactionStrategy'};

-- Task Queue
CREATE TABLE IF NOT EXISTS task_queue (
    task_id UUID PRIMARY KEY,
    agent_id UUID,
    task_type TEXT,
    payload TEXT,
    status TEXT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

-- Event Logs
CREATE TABLE IF NOT EXISTS event_logs (
    event_id TIMEUUID,
    agent_id UUID,
    event_type TEXT,
    message TEXT,
    severity TEXT,
    timestamp TIMESTAMP,
    PRIMARY KEY ((event_id), timestamp)
) WITH CLUSTERING ORDER BY (timestamp DESC);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_agent_name ON agent_registry (name);
CREATE INDEX IF NOT EXISTS idx_task_status ON task_queue (status);

