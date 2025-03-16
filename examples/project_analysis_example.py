"""
Example script demonstrating how to use the ProjectAnalysisService.

This script shows how to analyze a project's audio and MIDI tracks,
create detailed profiles, and update the project with analysis results.
"""

import os
import asyncio
import json
from datetime import datetime
import logging

from modules.services.project_analysis_service import ProjectAnalysisService
from modules.services.project_service import ProjectService
from modules.core.resources import ProjectResource, TrackResource

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def analyze_project(project_id: str, output_dir: str = "analysis_results") -> None:
    """Analyze a project and save the results.
    
    Args:
        project_id: ID of the project to analyze.
        output_dir: Directory to save analysis results.
    """
    try:
        # Initialize services
        project_service = ProjectService()
        analysis_service = ProjectAnalysisService(project_service)
        
        # Perform analysis
        logger.info(f"Starting analysis of project {project_id}")
        results = await analysis_service.analyze_project(project_id)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"analysis_{project_id}_{timestamp}.json")
        
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Analysis results saved to {output_file}")
        
        # Print summary
        print("\nAnalysis Summary:")
        print("-" * 50)
        
        # Project Structure
        structure = results["structure_analysis"]
        print(f"\nProject Structure:")
        print(f"Total Tracks: {structure['total_tracks']}")
        print(f"Audio Tracks: {len(structure['audio_tracks'])}")
        print(f"MIDI Tracks: {len(structure['midi_tracks'])}")
        
        # Audio Analysis
        audio = results["audio_analysis"]
        print(f"\nAudio Analysis:")
        print(f"Total Duration: {audio['overall_metrics']['total_duration']:.2f} seconds")
        print(f"Average Loudness: {audio['overall_metrics']['average_loudness']:.2f}")
        print(f"Dynamic Range: {audio['overall_metrics']['dynamic_range']:.2f}")
        
        # MIDI Analysis
        midi = results["midi_analysis"]
        print(f"\nMIDI Analysis:")
        print(f"Total Notes: {midi['overall_metrics']['total_notes']}")
        print(f"Average Velocity: {midi['overall_metrics']['average_velocity']:.2f}")
        print(f"Note Density: {midi['overall_metrics']['note_density']:.2f} notes/second")
        
        # Musical Analysis
        musical = results["musical_analysis"]
        print(f"\nMusical Analysis:")
        print(f"Key: {musical['harmony']['key']}")
        print(f"Time Signature: {musical['harmony']['time_signature']}")
        print(f"BPM: {musical['harmony']['bpm']}")
        print(f"Genre: {', '.join(musical['style']['genre'])}")
        print(f"Emotion: {', '.join(musical['style']['emotion'])}")
        print(f"Style Tags: {', '.join(musical['style']['style_tags'])}")
        
        # Track Details
        print("\nTrack Details:")
        print("-" * 50)
        
        for track in audio["tracks"]:
            print(f"\nAudio Track: {track['name']}")
            print(f"Duration: {track['duration']:.2f} seconds")
            print(f"Tempo: {track['features']['tempo']:.1f} BPM")
            print(f"Key: {track['features']['key']}")
            print(f"Genre: {track['genre']['primary']} ({track['genre']['confidence']:.2f})")
            
        for track in midi["tracks"]:
            print(f"\nMIDI Track: {track['name']}")
            print(f"Note Count: {track['metrics']['note_count']}")
            print(f"Average Velocity: {track['metrics']['average_velocity']:.2f}")
            print(f"Note Density: {track['metrics']['note_density']:.2f} notes/second")
            print(f"Pitch Range: {track['metrics']['pitch_range']}")
            
    except Exception as e:
        logger.error(f"Failed to analyze project: {e}")
        raise

async def main():
    """Main function."""
    # Example project ID
    project_id = "example_project"
    
    # Set output directory
    output_dir = "analysis_results"
    
    # Run analysis
    await analyze_project(project_id, output_dir)

if __name__ == "__main__":
    asyncio.run(main()) 