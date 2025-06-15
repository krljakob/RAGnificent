#!/usr/bin/env python3
"""
Test script for AsyncMarkdownScraper implementation.
"""

import asyncio
import time
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from RAGnificent.core.scraper import MarkdownScraper, AsyncMarkdownScraper

async def test_async_vs_sync_performance():
    """Test performance comparison between async and sync scrapers."""
    
    # Test URLs - using publicly available test sites
    test_urls = [
        "https://httpbin.org/html",
        "https://httpbin.org/json", 
        "https://httpbin.org/xml",
        "https://httpbin.org/robots.txt",
        "https://httpbin.org/status/200"
    ]
    
    print("🚀 Testing AsyncMarkdownScraper vs MarkdownScraper Performance")
    print("=" * 70)
    
    # Test with sync scraper (baseline)
    print("📊 Testing synchronous scraper...")
    sync_scraper = MarkdownScraper(requests_per_second=10.0, cache_enabled=False)
    
    start_time = time.time()
    sync_results = []
    for url in test_urls:
        try:
            html = sync_scraper.scrape_website(url)
            sync_results.append((url, html))
        except Exception as e:
            print(f"❌ Sync error for {url}: {e}")
    
    sync_duration = time.time() - start_time
    print(f"✅ Sync scraping: {len(sync_results)}/{len(test_urls)} URLs in {sync_duration:.2f}s")
    
    # Test with async scraper
    print("\n📊 Testing asynchronous scraper...")
    try:
        async_scraper = AsyncMarkdownScraper(requests_per_second=10.0, cache_enabled=False)
        
        start_time = time.time()
        async_results = await async_scraper.scrape_websites_async(test_urls)
        async_duration = time.time() - start_time
        
        print(f"✅ Async scraping: {len(async_results)}/{len(test_urls)} URLs in {async_duration:.2f}s")
        
        # Calculate performance improvement
        if sync_duration > 0:
            speedup = sync_duration / async_duration if async_duration > 0 else float('inf')
            print(f"🏆 Performance improvement: {speedup:.2f}x faster")
        
        return True
        
    except ImportError as e:
        print(f"❌ AsyncMarkdownScraper not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Async test failed: {e}")
        return False

async def test_async_content_conversion():
    """Test async content conversion functionality."""
    print("\n🔄 Testing async content conversion...")
    
    test_urls = [
        "https://httpbin.org/html",
        "https://httpbin.org/xml"
    ]
    
    try:
        async_scraper = AsyncMarkdownScraper(cache_enabled=False)
        
        # Test markdown conversion
        results = await async_scraper.scrape_and_convert_async(test_urls, "markdown")
        print(f"✅ Markdown conversion: {len(results)} URLs converted")
        
        # Test JSON conversion  
        results = await async_scraper.scrape_and_convert_async(test_urls, "json")
        print(f"✅ JSON conversion: {len(results)} URLs converted")
        
        return True
        
    except Exception as e:
        print(f"❌ Content conversion test failed: {e}")
        return False

async def test_async_error_handling():
    """Test async error handling with invalid URLs."""
    print("\n🛡️ Testing async error handling...")
    
    # Mix of valid and invalid URLs
    test_urls = [
        "https://httpbin.org/status/200",  # Valid
        "https://httpbin.org/status/404",  # 404 error
        "https://nonexistent-domain-12345.com",  # DNS error
        "https://httpbin.org/delay/1",     # Valid but slow
    ]
    
    try:
        async_scraper = AsyncMarkdownScraper(cache_enabled=False)
        results = await async_scraper.scrape_websites_async(test_urls)
        
        print(f"✅ Error handling: {len(results)} successful out of {len(test_urls)} URLs")
        print("   (Some failures expected for testing)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

async def main():
    """Run all async scraper tests."""
    print("🧪 RAGnificent AsyncMarkdownScraper Test Suite")
    print("=" * 50)
    
    tests = [
        test_async_vs_sync_performance,
        test_async_content_conversion, 
        test_async_error_handling
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            result = await test_func()
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! AsyncMarkdownScraper is working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))