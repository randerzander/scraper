#!/usr/bin/env python3
"""
YouTube helper functions for downloading videos, transcripts, comments, and thumbnails.
Supports both regular YouTube videos and Shorts.
"""

import os
import json
from pathlib import Path
import ffmpeg


def is_youtube_short(url: str) -> bool:
    """
    Determine if a URL is a YouTube Short.
    
    Args:
        url: YouTube URL
        
    Returns:
        True if it's a Short, False otherwise
    """
    return '/shorts/' in url


def download_youtube_video(url: str, output_dir: str = "scratch"):
    """
    Download a YouTube video and its thumbnail using yt-dlp Python API.
    Checks if video already exists in cache before downloading.
    
    Args:
        url: YouTube URL to download
        output_dir: Directory to save the video (default: scratch/)
        
    Returns:
        Tuple of (video_path, video_id, thumbnail_path) or (None, None, None) if failed
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    try:
        import yt_dlp
        import logging
        
        # Suppress yt-dlp logging
        yt_dlp_logger = logging.getLogger('yt_dlp')
        yt_dlp_logger.setLevel(logging.ERROR)
        
        # Extract video ID first to use as primary filename
        with yt_dlp.YoutubeDL({'quiet': True, 'no_warnings': True}) as ydl:
            info = ydl.extract_info(url, download=False)
            video_id = info.get('id')
        
        # Check if video already exists in cache
        cached_video = None
        for ext in ['mp4', 'webm', 'mkv', 'flv']:
            potential_file = output_path / f"{video_id}.{ext}"
            if potential_file.exists():
                cached_video = str(potential_file)
                break
        
        # Check for cached thumbnail
        cached_thumbnail = None
        for ext in ['jpg', 'jpeg', 'png', 'webp']:
            potential_thumb = output_path / f"{video_id}.{ext}"
            if potential_thumb.exists():
                cached_thumbnail = str(potential_thumb)
                break
        
        # If both video and thumbnail exist, return cached versions
        if cached_video:
            return cached_video, video_id, cached_thumbnail
        
        # Use simple video ID-based naming to avoid filename issues
        simple_template = str(output_path / f"{video_id}.%(ext)s")
        
        # Configure yt-dlp options
        ydl_opts = {
            'outtmpl': simple_template,
            'format': 'best',  # Best single file format
            'writethumbnail': True,  # Download thumbnail
            'convert_thumbnails': 'jpg',  # Convert to JPG
            'quiet': True,  # Suppress most output
            'no_warnings': True,  # Suppress warnings
            'noprogress': True,  # No progress bar
            'logger': yt_dlp_logger,  # Use our custom logger
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Download
            ydl.download([url])
            
            # Find the downloaded video file
            downloaded_file = None
            for ext in ['mp4', 'webm', 'mkv', 'flv']:
                potential_file = output_path / f"{video_id}.{ext}"
                if potential_file.exists():
                    downloaded_file = str(potential_file)
                    break
            
            # Find thumbnail file
            thumbnail_path = None
            for ext in ['jpg', 'jpeg', 'png', 'webp']:
                potential_thumb = output_path / f"{video_id}.{ext}"
                if potential_thumb.exists():
                    thumbnail_path = str(potential_thumb)
                    break
            
            if downloaded_file:
                return downloaded_file, video_id, thumbnail_path
            else:
                return None, None, None
            
    except ImportError:
        return None, None, None
    except Exception as e:
        return None, None, None



def download_youtube_comments(video_id: str, output_dir: str = "scratch", max_comments: int = 10):
    """
    Download the top comments from a YouTube video using yt-dlp Python API.
    Checks if comments already exist in cache before downloading.
    
    Args:
        video_id: YouTube video ID
        output_dir: Directory to save comments (default: scratch/)
        max_comments: Maximum number of top comments to download (default: 10)
        
    Returns:
        List of comment dicts or None if failed
    """
    # Check if comments already exist in cache
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    comments_file = output_path / f"{video_id}_comments.json"
    
    if comments_file.exists():
        try:
            with open(comments_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass  # If file is corrupted, re-download
    
    try:
        import yt_dlp
        import logging
        
        # Suppress yt-dlp logging
        yt_dlp_logger = logging.getLogger('yt_dlp')
        yt_dlp_logger.setLevel(logging.ERROR)
        
        url = f"https://www.youtube.com/watch?v={video_id}"
        
        # Configure yt-dlp to get comments
        ydl_opts = {
            'skip_download': True,
            'getcomments': True,
            'extractor_args': {'youtube': {'comment_sort': ['top']}},
            'quiet': True,
            'no_warnings': True,
            'logger': yt_dlp_logger,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            comments_data = info.get('comments', [])
        
        if not comments_data:
            return []
        
        # Sort by like count and take top N
        sorted_comments = sorted(
            comments_data, 
            key=lambda x: x.get('like_count', 0), 
            reverse=True
        )[:max_comments]
        
        # Save to file
        with open(comments_file, 'w', encoding='utf-8') as f:
            json.dump(sorted_comments, f, indent=2, ensure_ascii=False)
        
        return sorted_comments
            
    except ImportError:
        return None
    except Exception as e:
        return None



def get_youtube_transcript(video_id: str, output_dir: str = "scratch"):
    """
    Get transcript from YouTube's API (for regular videos with captions).
    Checks if transcript already exists in cache before downloading.
    
    Args:
        video_id: YouTube video ID
        output_dir: Directory to save transcript (default: scratch/)
        
    Returns:
        Transcript text or None if failed
    """
    # Check if transcript already exists in cache
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    transcript_file = output_path / f"{video_id}_transcript.txt"
    
    if transcript_file.exists():
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                return f.read()
        except:
            pass  # If file is corrupted, re-download
    
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
        
        # Get available transcripts using the correct method name
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        
        # Try to get manually created transcript first, then auto-generated
        transcript = None
        try:
            # Prefer manually created English transcript
            transcript = transcript_list.find_manually_created_transcript(['en'])
        except:
            try:
                # Fall back to auto-generated English transcript
                transcript = transcript_list.find_generated_transcript(['en'])
            except:
                # Get any available transcript
                available = list(transcript_list)
                if available:
                    transcript = available[0]
        
        if not transcript:
            return None
        
        # Fetch the actual transcript
        transcript_data = transcript.fetch()
        
        # Combine all text segments - entries are FetchedTranscriptSnippet objects
        full_text = " ".join([entry.text for entry in transcript_data])
        
        # Save to file
        with open(transcript_file, 'w', encoding='utf-8') as f:
            f.write(full_text)
        
        return full_text
        
    except ImportError:
        return None
    except TranscriptsDisabled:
        return None
    except NoTranscriptFound:
        return None
    except Exception as e:
        return None


def transcribe_audio_with_whisper(video_path: str, model: str = "base", output_dir: str = None):
    """
    Transcribe a video file using pywhispercpp.
    Checks if transcript already exists in cache before transcribing.
    
    Args:
        video_path: Path to the video file
        model: Whisper model to use (tiny, base, small, medium, large)
        output_dir: Directory to save transcript (if None, uses video directory)
        
    Returns:
        Transcription text or None if failed
    """
    video_path = Path(video_path)
    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = video_path.parent
    
    # Check if transcript already exists in cache
    transcript_file = output_path / f"{video_path.stem}.txt"
    if transcript_file.exists():
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                return f.read()
        except:
            pass  # If file is corrupted, re-transcribe
    
    # Extract audio to WAV format
    audio_path = output_path / f"{video_path.stem}.wav"
    
    try:
        (
            ffmpeg
            .input(str(video_path))
            .output(
                str(audio_path),
                acodec='pcm_s16le',
                ar=16000,
                ac=1,
                loglevel='error'
            )
            .overwrite_output()
            .run(quiet=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        return None
    
    # Run pywhispercpp transcription
    try:
        from pywhispercpp.model import Model
        import logging
        import sys
        import os
        
        # Suppress pywhispercpp logging
        logging.getLogger('pywhispercpp.utils').setLevel(logging.ERROR)
        logging.getLogger('pywhispercpp.model').setLevel(logging.ERROR)
        
        # Redirect file descriptors to suppress C library output
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        
        try:
            os.dup2(devnull_fd, 1)
            os.dup2(devnull_fd, 2)
            
            # Initialize model and transcribe
            model_obj = Model(model, n_threads=4)
            segments = model_obj.transcribe(str(audio_path))
        finally:
            # Restore file descriptors
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)
            os.close(devnull_fd)
        
        # Collect all text segments
        transcription = " ".join([segment.text for segment in segments]).strip()
        
        # Save transcription to file
        with open(transcript_file, 'w', encoding='utf-8') as f:
            f.write(transcription)
        
        # Clean up audio file
        try:
            audio_path.unlink()
        except:
            pass
        
        return transcription
            
    except ImportError:
        return None
    except Exception as e:
        return None
