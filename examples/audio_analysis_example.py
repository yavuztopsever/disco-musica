"""Example script demonstrating the usage of the audio analysis module."""

import os
import json
from modules.services.audio_analysis import AudioAnalyzer

def analyze_audio_file(file_path: str, output_path: str = None) -> None:
    """Analyze an audio file and optionally save the results.
    
    Args:
        file_path: Path to the audio file to analyze
        output_path: Optional path to save the analysis results
    """
    try:
        # Initialize the audio analyzer
        analyzer = AudioAnalyzer()
        
        # Perform the analysis
        results = analyzer.analyze_file(file_path)
        
        # Print the results in a formatted way
        print("\nAudio Analysis Results:")
        print("-" * 50)
        
        # Print genre information
        print("\nGenre Classification:")
        print(f"Primary Genre: {results['genre']['primary']}")
        print(f"Confidence: {results['genre']['confidence']:.2%}")
        print("\nSecondary Genres:")
        for genre, confidence in results['genre']['secondary']:
            print(f"- {genre}: {confidence:.2%}")
            
        # Print musical features
        print("\nMusical Features:")
        print(f"Key: {results['features']['key']}")
        print(f"Time Signature: {results['features']['time_signature']}")
        print(f"Tempo: {results['features']['tempo']:.1f} BPM")
        print(f"Duration: {results['features']['duration']:.2f} seconds")
        
        # Save results if output path is provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {output_path}")
            
    except Exception as e:
        print(f"Error analyzing audio file: {e}")

def main():
    """Main function to demonstrate audio analysis usage."""
    # Example usage
    audio_file = "path/to/your/audio/file.wav"  # Replace with your audio file path
    output_file = "analysis_results.json"  # Optional output file
    
    if not os.path.exists(audio_file):
        print(f"Error: Audio file not found: {audio_file}")
        return
        
    analyze_audio_file(audio_file, output_file)

if __name__ == "__main__":
    main() 