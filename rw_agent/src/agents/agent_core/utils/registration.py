def register_components(self, base_path: Path):
    """Register all components with hashes"""
    im = IntegrityManager(self.session)
    current_hashes = im._get_current_hashes(base_path)
    
    self.session.execute("""
        UPDATE agent_registry 
        SET file_hashes = %s, last_sync = %s 
        WHERE agent_id = %s
    """, [
        current_hashes,
        datetime.now(),
        uuid.UUID(self.agent_id)
    ])

