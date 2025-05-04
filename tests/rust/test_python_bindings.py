import pytest
import ragnificent_rs


def test_convert_html_to_markdown():
    html = """
        <html>
            <head><title>Test Page</title></head>
            <body>
                <h1>Main Title</h1>
                <p>This is a test paragraph.</p>
            </body>
        </html>
    """
    base_url = "https://example.com"
    markdown = ragnificent_rs.convert_html_to_markdown(html, base_url)

    assert "# Test Page" in markdown
    assert "# Main Title" in markdown
    assert "This is a test paragraph." in markdown


def test_chunk_markdown():
    markdown = """
# Title

## Section 1

This is a test paragraph.

## Section 2

* List item 1
* List item 2
    """

    chunks = ragnificent_rs.chunk_markdown(markdown, 500, 50)
    assert len(chunks) > 0
    assert any("# Title" in chunk for chunk in chunks)
    assert any("## Section 1" in chunk for chunk in chunks)
    assert any("## Section 2" in chunk for chunk in chunks)


def test_render_js_page():
    url = "https://example.com"
    # Add required wait_time parameter
    wait_time = 5000  # 5 seconds in milliseconds
    html = ragnificent_rs.render_js_page(url, wait_time)
    assert isinstance(html, str)
    assert len(html) > 0


def test_error_handling():
    # Use very malformed inputs that should trigger errors
    # Try various problematic inputs to ensure at least one raises an error
    with pytest.raises(Exception):
        # Using an invalid HTML that should cause the parser to fail
        ragnificent_rs.convert_html_to_markdown("<unclosed", "invalid-url")
    
    with pytest.raises(Exception):
        # Using negative values for chunk parameters should cause errors
        ragnificent_rs.chunk_markdown("test", -1, -1)
    
    with pytest.raises(Exception):
        # Invalid URL should cause rendering to fail
        ragnificent_rs.render_js_page("not-a-valid-url", 1000)
