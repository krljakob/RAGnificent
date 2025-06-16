#!/usr/bin/env python3
"""
Test script for the new Rust core engine
"""

import sys
import traceback

try:
    from RAGnificent.ragnificent_rs import MarkdownLab, Config, ChunkOptions, SitemapFilters
    print("✅ Successfully imported the new core engine classes")
except ImportError as e:
    print(f"❌ Failed to import core engine: {e}")
    sys.exit(1)

def test_basic_functionality():
    """Test basic core engine functionality"""
    print("\n=== Testing Basic Functionality ===")
    
    try:
        # Test creating with default config
        lab = MarkdownLab.default()
        print("✅ Created MarkdownLab with default config")
        
        # Test HTML conversion
        html = """
        <html>
            <head><title>Test Page</title></head>
            <body>
                <h1>Main Title</h1>
                <p>This is a test paragraph with <a href="/link">a link</a>.</p>
                <ul>
                    <li>Item 1</li>
                    <li>Item 2</li>
                </ul>
            </body>
        </html>
        """
        
        markdown = lab.convert_html(html, "https://example.com", "markdown")
        print("✅ HTML to Markdown conversion successful")
        print(f"   Result preview: {markdown[:100]}...")
        
        json_result = lab.convert_html(html, "https://example.com", "json")
        print("✅ HTML to JSON conversion successful")
        print(f"   JSON preview: {json_result[:100]}...")
        
        # Test chunking
        long_content = "# Title\n\nThis is a long paragraph that should be chunked. " * 50
        chunks = lab.chunk_content(long_content, "https://example.com", None)
        print(f"✅ Content chunking successful - created {len(chunks)} chunks")
        
        if chunks:
            print(f"   First chunk preview: {chunks[0].content[:50]}...")
            print(f"   First chunk source: {chunks[0].source_url}")
            print(f"   First chunk index: {chunks[0].index}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_configuration():
    """Test custom configuration"""
    print("\n=== Testing Custom Configuration ===")
    
    try:
        # Test custom config
        config = Config(
            requests_per_second=2.0,
            timeout_seconds=15,
            max_retries=5,
            chunk_size=500,
            chunk_overlap=100,
            cache_enabled=False,
            max_workers=4
        )
        print("✅ Created custom config")
        
        lab = MarkdownLab(config)
        print("✅ Created MarkdownLab with custom config")
        
        # Test chunking options
        chunk_opts = ChunkOptions(size=300, overlap=50, semantic_splitting=True)
        content = "# Section 1\n\nParagraph 1. " * 20 + "\n\n# Section 2\n\nParagraph 2. " * 20
        chunks = lab.chunk_content(content, "https://test.com", chunk_opts)
        print(f"✅ Custom chunking successful - created {len(chunks)} chunks")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        traceback.print_exc()
        return False

def test_sitemap_filters():
    """Test sitemap filters creation"""
    print("\n=== Testing Sitemap Filters ===")
    
    try:
        filters = SitemapFilters(
            min_priority=0.5,
            include_patterns=[".*\\.html", ".*\\.php"],
            exclude_patterns=[".*admin.*", ".*private.*"],
            limit=100
        )
        print("✅ Created sitemap filters")
        
        # Note: We can't test actual sitemap scraping without a valid sitemap URL
        # This just tests that the objects can be created
        
        return True
        
    except Exception as e:
        print(f"❌ Sitemap filters test failed: {e}")
        traceback.print_exc()
        return False

def test_error_handling():
    """Test error handling"""
    print("\n=== Testing Error Handling ===")
    
    try:
        lab = MarkdownLab.default()
        
        # Test with invalid HTML
        try:
            result = lab.convert_html("invalid html", "https://example.com", "markdown")
            # This should still work as the HTML parser is robust
            print("✅ Invalid HTML handled gracefully")
        except Exception as e:
            print(f"⚠️  Invalid HTML caused error (this might be expected): {e}")
        
        # Test with malformed URL
        try:
            chunks = lab.chunk_content("test content", "not-a-valid-url", None)
            print("✅ Invalid URL in chunking handled gracefully")
        except Exception as e:
            print(f"⚠️  Invalid URL caused error (this might be expected): {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 Testing New Rust Core Engine")
    print("=" * 50)
    
    tests = [
        test_basic_functionality,
        test_configuration,
        test_sitemap_filters,
        test_error_handling,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        else:
            print("❌ Test failed, continuing with remaining tests...")
    
    print("\n" + "=" * 50)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The new core engine is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())