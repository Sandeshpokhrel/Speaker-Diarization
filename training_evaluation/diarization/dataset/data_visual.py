import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import random

def load_segments(segments_file):
    """Load segments from file and group by merged_id."""
    segments_data = defaultdict(list)
    with open(segments_file, 'r') as f:
        for line in f:
            utt_id, merged_id, start, end = line.strip().split()
            segments_data[merged_id].append({
                'utt_id': utt_id,
                'start': float(start),
                'end': float(end)
            })
    return segments_data

def generate_speaker_colors(segments):
    """Generate consistent colors for each unique speaker."""
    speakers = set()
    for segment in segments:
        speaker = segment['utt_id'].split('_')[0]
        speakers.add(speaker)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(speakers)))
    return dict(zip(speakers, colors))

def plot_diarization(segments_file, output_dir='plots'):
    """Create visualization plots for diarization data."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    segments_data = load_segments(segments_file)
    
    # Plot each merged file
    for merged_id, segments in segments_data.items():
        # Create new figure
        plt.figure(figsize=(15, 5))
        
        # Generate colors for speakers
        speaker_colors = generate_speaker_colors(segments)
        
        # Plot segments
        for i, segment in enumerate(segments):
            speaker = segment['utt_id'].split('_')[0]
            start = segment['start']
            end = segment['end']
            duration = end - start
            
            # Plot the segment
            plt.barh(y=0, width=duration, left=start, 
                    color=speaker_colors[speaker], alpha=0.6,
                    label=speaker)
            
            # Add text label
            plt.text(start + duration/2, 0, speaker,
                    ha='center', va='center')
        
        # Remove duplicate labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(),
                  title="Speakers", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Customize plot
        plt.title(f'Diarization Timeline for {merged_id}')
        plt.xlabel('Time (seconds)')
        plt.yticks([])  # Hide y-axis ticks
        plt.grid(True, axis='x', linestyle='--', alpha=0.7)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(f'{output_dir}/diarization_{merged_id}.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()

def plot_statistics(segments_file):
    """Plot statistics about the diarization data."""
    segments_data = load_segments(segments_file)
    
    # Collect statistics
    speaker_durations = defaultdict(float)
    overlap_counts = defaultdict(int)
    total_durations = []
    
    for merged_id, segments in segments_data.items():
        # Calculate total duration
        total_duration = max(seg['end'] for seg in segments)
        total_durations.append(total_duration)
        
        # Calculate speaker durations
        for segment in segments:
            speaker = segment['utt_id'].split('_')[0]
            duration = segment['end'] - segment['start']
            speaker_durations[speaker] += duration
            
            # Check for overlaps
            for other_segment in segments:
                if segment != other_segment:
                    if (segment['start'] < other_segment['end'] and
                        segment['end'] > other_segment['start']):
                        overlap_counts[merged_id] += 1
    
    # Create statistics plots
    plt.figure(figsize=(15, 10))
    
    # Speaker durations
    plt.subplot(2, 1, 1)
    speakers = list(speaker_durations.keys())
    durations = list(speaker_durations.values())
    plt.bar(speakers, durations)
    plt.title('Total Duration per Speaker')
    plt.xlabel('Speaker')
    plt.ylabel('Duration (seconds)')
    
    # Overlap counts
    plt.subplot(2, 1, 2)
    merged_ids = list(overlap_counts.keys())
    counts = list(overlap_counts.values())
    plt.bar(merged_ids, counts)
    plt.title('Number of Overlaps per Merged File')
    plt.xlabel('Merged File ID')
    plt.ylabel('Number of Overlaps')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('plots/statistics.png', bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    # Set the paths
    segments_file = 'merged_audio/train/details/segments'
    
    # Create visualizations
    plot_diarization(segments_file)
    plot_statistics(segments_file)
    print("Plots have been saved in the 'plots' directory.")