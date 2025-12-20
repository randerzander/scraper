#!/usr/bin/env python3
"""
Wikipedia scraping utilities using the Wikipedia API.
Fetches clean article text instead of scraping HTML.
"""

import logging
import re


logger = logging.getLogger(__name__)

# Suppress wikipediaapi INFO logs
logging.getLogger('wikipediaapi').setLevel(logging.WARNING)


def scrape_wikipedia(url: str) -> str:
    """
    Fetch Wikipedia article content using the Wikipedia API.
    
    Args:
        url: Wikipedia URL to fetch
        
    Returns:
        Article content in markdown format
    """
    try:
        import wikipediaapi
        
        # Extract article title from URL
        title = _extract_wikipedia_title(url)
        
        if not title:
            logger.warning(f"Could not extract article title from URL: {url}")
            return f"Error: Could not extract article title from Wikipedia URL: {url}"
        
        # Extract language from URL (default to 'en')
        lang = _extract_wikipedia_language(url)
        
        # Create Wikipedia API object with increased timeout
        wiki = wikipediaapi.Wikipedia(
            user_agent='UsefoolBot/1.0 (Discord bot)',
            language=lang,
            timeout=30.0
        )
        
        # Get the Wikipedia page
        page = wiki.page(title)
        
        if not page.exists():
            logger.warning(f"Wikipedia page not found: {title}")
            return f"Error: Wikipedia article not found: {title}\n\nPlease check the article title and try again."
        
        # Build markdown content
        content = f"# {page.title}\n\n"
        
        # Add summary (first section)
        if page.summary:
            content += f"## Summary\n\n{page.summary}\n\n"
        
        # Add full content
        content += f"## Full Article\n\n{page.text}\n\n"
        
        # Add metadata
        content += f"---\n\n"
        content += f"**URL:** {page.fullurl}\n\n"
        
        if page.categories:
            cats = [cat.replace('Category:', '') for cat in list(page.categories.keys())[:10]]
            content += f"**Categories:** {', '.join(cats)}\n\n"
        
        if page.links:
            content += f"**Related Articles:** {len(page.links)} links\n\n"
        
        return content
        
    except ImportError:
        logger.error("Wikipedia-API module not available")
        return "Error: Wikipedia functionality not available. Missing dependency: wikipedia-api"
    except Exception as e:
        logger.error(f"Error fetching Wikipedia article: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error fetching Wikipedia article: {str(e)}"


def _extract_wikipedia_title(url: str) -> str:
    """
    Extract the article title from a Wikipedia URL.
    
    Handles various Wikipedia URL formats:
    - https://en.wikipedia.org/wiki/Article_Title
    - https://en.m.wikipedia.org/wiki/Article_Title
    - https://wikipedia.org/wiki/Article_Title
    
    Args:
        url: Wikipedia URL
        
    Returns:
        Article title or None if extraction failed
    """
    # Pattern for /wiki/ARTICLE_TITLE
    match = re.search(r'/wiki/([^#?]+)', url)
    if match:
        title = match.group(1)
        # URL decode and replace underscores with spaces
        import urllib.parse
        title = urllib.parse.unquote(title)
        title = title.replace('_', ' ')
        return title
    
    return None


def _extract_wikipedia_language(url: str) -> str:
    """
    Extract the language code from a Wikipedia URL.
    
    Args:
        url: Wikipedia URL
        
    Returns:
        Language code (e.g., 'en', 'es', 'fr') or 'en' if not found
    """
    # Pattern for https://XX.wikipedia.org
    match = re.search(r'https?://([a-z]{2,3})\.wikipedia\.org', url)
    if match:
        return match.group(1)
    
    return 'en'  # Default to English


def search_wikipedia(query: str, max_results: int = 5, lang: str = 'en') -> str:
    """
    Search Wikipedia and return results.
    
    Args:
        query: Search query
        max_results: Maximum number of results to return (default: 5)
        lang: Language code (default: 'en')
        
    Returns:
        Search results in markdown format
    """
    try:
        import wikipediaapi
        
        # Create Wikipedia API object with increased timeout
        wiki = wikipediaapi.Wikipedia(
            user_agent='UsefoolBot/1.0 (Discord bot)',
            language=lang,
            timeout=30.0
        )
        
        # Get search results - note: wikipedia-api doesn't have search
        # We'll just try to get the page directly
        page = wiki.page(query)
        
        if page.exists():
            content = f"# Wikipedia Article Found\n\n"
            content += f"**{page.title}**\n\n"
            content += f"{page.summary}\n\n"
            content += f"[Read more]({page.fullurl})\n"
            return content
        else:
            return f"No Wikipedia article found for query: {query}"
        
    except ImportError:
        logger.error("Wikipedia-API module not available")
        return "Error: Wikipedia functionality not available. Missing dependency: wikipedia-api"
    except Exception as e:
        logger.error(f"Error searching Wikipedia: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error searching Wikipedia: {str(e)}"
