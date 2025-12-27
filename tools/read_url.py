#!/usr/bin/env python3
"""
URL reading utilities with special handling for different content types.

Automatically detects URL type and extracts content appropriately:
- YouTube URLs: Fetches video transcripts
- Wikipedia URLs: Uses Wikipedia API for clean article text
- Regular web pages: Converts HTML to markdown

Provides a unified interface for reading any URL content.
"""

import logging
import requests
import html2text
from pyreadability import Readability
from tools import retriever
from pathlib import Path
from utils import create_tool_spec


logger = logging.getLogger(__name__)


# Tool specification for agent registration
TOOL_SPEC = create_tool_spec(
    name="read_url",
    description="Read and extract content from any URL. Handles YouTube videos (returns transcript), Wikipedia articles (returns article content), and regular web pages (returns markdown). Use this for ALL URLs including YouTube videos, Wikipedia articles, documentation, blog posts, etc.",
    parameters={
        "url": "The URL to read (supports YouTube, Wikipedia, and any web page)"
    },
    required=["url"]
)


def read_url(url: str) -> str:
    """
    Read content from any URL and return it in a readable format.
    
    Automatically handles different URL types:
    - YouTube videos: Returns transcript
    - Wikipedia articles: Returns article content via API
    - Regular web pages: Returns HTML converted to markdown
    
    Also writes all fetched content to the vector database.
    
    Args:
        url: URL to read
        
    Returns:
        Content from the URL in markdown format
    """
    content = None
    
    # Check if it's a YouTube URL
    if 'youtube.com' in url or 'youtu.be' in url:
        content = _scrape_youtube_url(url)
    
    # Check if it's a Wikipedia URL
    elif 'wikipedia.org' in url:
        content = _scrape_wikipedia_url(url)
    
    # Regular web page scraping
    else:
        try:
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            })
            response.raise_for_status()
            logger.debug(f"URL fetch completed - Status: {response.status_code}, Content length: {len(response.text)} chars")
            
            # Use pyreadability to parse HTML and extract main content
            reader = Readability(response.text)
            result = reader.parse()
            
            # Convert HTML content to markdown using html2text
            h = html2text.HTML2Text()
            h.body_width = 0  # Don't wrap lines
            h.ignore_links = False
            markdown_content = h.handle(result['content'])
            
            # Add title if available
            if result.get('title'):
                markdown_content = f"# {result['title']}\n\n{markdown_content}"
            
            content = markdown_content
        except Exception as e:
            logger.error(f"URL scraping failed for {url}: {str(e)}")
            content = f"Error scraping URL: {str(e)}"
    
    # Write to vector database if content was successfully fetched
    if content and not content.startswith("Error"):
        try:
            retriever.write(source=url, text=content)
            logger.debug(f"Wrote content from {url} to vector database")
        except Exception as e:
            logger.warning(f"Failed to write to vector database: {str(e)}")
    
    return content


def _scrape_youtube_url(url: str) -> str:
    """
    Handle YouTube URLs by extracting video ID and fetching transcript.
    For Shorts, automatically downloads and transcribes with whisper.
    If transcript is unavailable or very short (<1500 chars), enriches with thumbnail caption and top comments.
    
    Args:
        url: YouTube URL
        
    Returns:
        Transcript or video information in markdown format
    """
    try:
        from tools.youtube import (
            is_youtube_short, 
            get_youtube_transcript,
            download_youtube_video,
            transcribe_audio_with_whisper,
            download_youtube_comments
        )
        import re
        import tempfile
        
        logger.info(f"Detected YouTube URL, fetching transcript...")
        
        # Extract video ID from various YouTube URL formats
        video_id = None
        
        # Pattern for /watch?v=VIDEO_ID
        match = re.search(r'[?&]v=([^&]+)', url)
        if match:
            video_id = match.group(1)
        
        # Pattern for /shorts/VIDEO_ID
        match = re.search(r'/shorts/([^/?]+)', url)
        if match:
            video_id = match.group(1)
        
        # Pattern for youtu.be/VIDEO_ID
        match = re.search(r'youtu\.be/([^/?]+)', url)
        if match:
            video_id = match.group(1)
        
        if not video_id:
            logger.warning(f"Could not extract video ID from YouTube URL: {url}")
            return f"Error: Could not extract video ID from YouTube URL"
        
        transcript = None
        thumbnail_path = None
        
        # Check if it's a Short
        if is_youtube_short(url):
            logger.info(f"Processing YouTube Short: {video_id}")
            
            # Use scratch directory for download
            scratch_dir = Path("scratch")
            scratch_dir.mkdir(exist_ok=True)
            
            # Download the Short (also gets thumbnail)
            video_path, _, thumbnail_path = download_youtube_video(url, output_dir=str(scratch_dir))
            
            if not video_path:
                logger.error(f"Failed to download YouTube Short: {video_id}")
                return f"""# YouTube Short: {video_id}

**Error:** Failed to download the video.

URL: {url}
Video ID: {video_id}"""
            
            # Transcribe with whisper
            transcript = transcribe_audio_with_whisper(video_path, model="base", output_dir=str(scratch_dir))
            
            if transcript:
                logger.info(f"Transcribed YouTube Short: {len(transcript)} chars")
            
            # Always enrich Shorts with comments and thumbnail caption
            logger.info("Enriching YouTube Short with thumbnail caption and comments...")
            
            # Get top 10 comments - save to scratch
            comments = download_youtube_comments(video_id, output_dir=str(scratch_dir), max_comments=10)
            
            # Caption the thumbnail if available
            thumbnail_caption = None
            if thumbnail_path:
                try:
                    import os
                    from agent import two_round_image_caption, MODEL_CONFIG
                    
                    # Get API key from environment
                    api_key_env = MODEL_CONFIG.get("api_key_env", "OPENROUTER_API_KEY")
                    api_key = os.environ.get(api_key_env, "")
                    
                    thumbnail_caption = two_round_image_caption(thumbnail_path, api_key)
                    logger.info(f"Generated thumbnail caption: {len(thumbnail_caption)} chars")
                except Exception as e:
                    logger.warning(f"Failed to caption thumbnail: {e}")
            
            # Build the enriched Short response
            result = f"""# YouTube Short: {video_id}

URL: {url}
Video ID: {video_id}

"""
            
            # Add transcript section
            if transcript:
                result += f"""## Transcript (Auto-transcribed with Whisper)

{transcript}

"""
            else:
                result += "## Transcript\n\n*No transcript available*\n\n"
            
            # Add thumbnail caption section
            if thumbnail_caption:
                result += f"""## Thumbnail Caption

{thumbnail_caption}

"""
            
            # Add comments section
            if comments:
                result += "## Top 10 Comments\n\n"
                for i, comment in enumerate(comments, 1):
                    author = comment.get('author', 'Unknown')
                    text = comment.get('text', '')
                    likes = comment.get('like_count', 0)
                    result += f"{i}. **{author}** ({likes} likes)\n   {text}\n\n"
                logger.info(f"Added {len(comments)} top comments")
            
            return result
        else:
            # Try to get transcript for regular videos
            logger.info(f"Fetching YouTube transcript for: {video_id}")
            
            # Use a temporary directory that we'll ignore (don't save the file)
            with tempfile.TemporaryDirectory() as tmpdir:
                transcript = get_youtube_transcript(video_id, output_dir=tmpdir)
            
            if transcript:
                logger.info(f"Got YouTube transcript: {len(transcript)} chars")
            else:
                logger.warning(f"No transcript available for YouTube video: {video_id} - checking video length...")
                
                # Check video duration before attempting to download and transcribe
                try:
                    import yt_dlp
                    
                    # Get video info using yt-dlp Python API
                    ydl_opts = {
                        'quiet': True,
                        'no_warnings': True,
                        'extract_flat': True,  # Don't download, just get metadata
                    }
                    
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        video_info = ydl.extract_info(url, download=False)
                    
                    duration_seconds = video_info.get('duration', 0)
                    duration_minutes = duration_seconds / 60
                    
                    logger.info(f"Video duration: {duration_minutes:.1f} minutes")
                    
                    # If video is 10 minutes or longer, don't transcribe
                    if duration_seconds >= 600:  # 10 minutes
                        logger.info(f"Video too long for transcription: {duration_minutes:.1f} minutes")
                        return f"""# YouTube Video: {video_id}

**No transcript available** for this video.

URL: {url}
Video ID: {video_id}
Duration: {duration_minutes:.1f} minutes

**Note:** This video is too long ({duration_minutes:.1f} minutes) to automatically transcribe without existing captions. Transcribing long videos requires significant processing time and resources.

If you need the transcript, you can:
1. Request the video creator to add captions
2. Use dedicated transcription services
3. Download and transcribe manually with the YouTube tools"""
                    
                    logger.info(f"Video length OK ({duration_minutes:.1f} minutes), proceeding with transcription...")
                    
                except Exception as e:
                    logger.warning(f"Could not check video duration: {e}")
                    # Continue anyway if we can't check duration
                
                # Fall back to downloading and transcribing with whisper
                scratch_dir = Path("scratch")
                scratch_dir.mkdir(exist_ok=True)
                
                logger.info(f"Downloading YouTube video for transcription...")
                video_path, _, thumbnail_path = download_youtube_video(url, output_dir=str(scratch_dir))
                
                if not video_path:
                    logger.error(f"Failed to download YouTube video: {video_id}")
                    return f"""# YouTube Video: {video_id}

**No transcript available** for this video and download failed.

URL: {url}
Video ID: {video_id}"""
                
                logger.info(f"Transcribing YouTube video audio with whisper...")
                transcript = transcribe_audio_with_whisper(video_path, model="base", output_dir=str(scratch_dir))
                
                if not transcript:
                    logger.error(f"Failed to transcribe YouTube video: {video_id}")
                    return f"""# YouTube Video: {video_id}

**Error:** No captions available and audio transcription failed.

URL: {url}
Video ID: {video_id}"""
                
                logger.info(f"Successfully transcribed YouTube video - {len(transcript)} chars")
        
        # If we have no transcript or a very short transcript (<1500 chars),
        # enrich with thumbnail caption and top comments
        if not transcript or len(transcript) < 1500:
            logger.info(f"Transcript is {'missing' if not transcript else 'short'} ({len(transcript) if transcript else 0} chars), enriching with comments and thumbnail...")
            
            # Get top 10 comments - save to scratch
            scratch_dir = Path("scratch")
            scratch_dir.mkdir(exist_ok=True)
            comments = download_youtube_comments(video_id, output_dir=str(scratch_dir), max_comments=10)
            
            # Build the enriched response
            result = f"""# YouTube {'Short' if is_youtube_short(url) else 'Video'}: {video_id}

URL: {url}
Video ID: {video_id}
"""
            
            if transcript:
                result += f"""
## Transcript (Auto-transcribed with Whisper)

{transcript}
"""
            
            # Add comments if available
            if comments:
                result += "\n## Top 10 Comments\n\n"
                for i, comment in enumerate(comments, 1):
                    author = comment.get('author', 'Unknown')
                    text = comment.get('text', '')
                    likes = comment.get('like_count', 0)
                    result += f"{i}. **{author}** ({likes} likes)\n   {text}\n\n"
                
                logger.info(f"Added {len(comments)} top comments")
            
            return result
        
        # Regular response with full transcript
        video_type = "Short" if is_youtube_short(url) else "Video"
        return f"""# YouTube {video_type} Transcript
            
**Video ID:** {video_id}
**URL:** {url}

## Transcript{' (Auto-transcribed with Whisper)' if is_youtube_short(url) or not get_youtube_transcript(video_id, output_dir=tempfile.gettempdir()) else ''}

{transcript}"""
            
    except ImportError as e:
        logger.error(f"YouTube tools not available: {e}")
        return f"Error: YouTube transcript functionality not available. Missing dependency: {str(e)}"
    except Exception as e:
        logger.error(f"Error fetching YouTube transcript: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error fetching YouTube transcript: {str(e)}"


def _scrape_wikipedia_url(url: str) -> str:
    """
    Handle Wikipedia URLs by using the Wikipedia API.
    
    Args:
        url: Wikipedia URL
        
    Returns:
        Article content in markdown format
    """
    try:
        from tools.wiki_scrape import scrape_wikipedia
        
        logger.info(f"Detected Wikipedia URL, using Wikipedia API...")
        return scrape_wikipedia(url)
        
    except ImportError as e:
        logger.error(f"Wikipedia tools not available: {e}")
        return f"Error: Wikipedia functionality not available. Missing dependency: {str(e)}"
    except Exception as e:
        logger.error(f"Error fetching Wikipedia article: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error fetching Wikipedia article: {str(e)}"
