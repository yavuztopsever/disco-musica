class ResourceService:
    """Service for managing resources."""
    
    def __init__(
        self,
        project_repository: 'ProjectRepository',
        track_repository: 'TrackRepository',
        audio_processor: AudioProcessor,
        midi_processor: MidiProcessor
    ):
        """Initialize the service."""
        self.project_repository = project_repository
        self.track_repository = track_repository
        self.audio_processor = audio_processor
        self.midi_processor = midi_processor
        
    async def create_or_update_project(self, project_path: str) -> str:
        """Create or update a project from a Logic Pro file.
        
        Args:
            project_path: Path to Logic Pro project file.
            
        Returns:
            Project ID.
        """
        # Check if project already exists
        existing_project = await self.project_repository.find_by_path(project_path)
        if existing_project:
            return existing_project.resource_id
            
        # Create new project
        project = ProjectResource(
            name=os.path.basename(project_path),
            basic_info={"original_file_path": project_path}
        )
        await self.project_repository.save(project)
        return project.resource_id 