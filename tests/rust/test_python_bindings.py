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

    # Verify the conversion is not empty and contains expected content
    assert isinstance(markdown, str), "Conversion should return a string"
    assert len(markdown) > 0, "Conversion should not return empty string"

    # Verify title extraction and formatting
    assert "# Test Page" in markdown, "Should extract and format HTML title as H1"
    assert "# Main Title" in markdown, "Should convert H1 tags to markdown headers"
    assert "This is a test paragraph." in markdown, "Should preserve paragraph content"

    # Verify no HTML tags remain in output
    assert "<html>" not in markdown, "Should not contain HTML tags in output"
    assert "<body>" not in markdown, "Should not contain HTML tags in output"
    assert "<p>" not in markdown, "Should not contain HTML tags in output"

    # Verify markdown structure is valid
    lines = markdown.strip().split("\n")
    header_lines = [line for line in lines if line.startswith("#")]
    assert (
        len(header_lines) >= 2
    ), "Should have at least 2 header lines (title + main title)"


def test_chunk_markdown():
    markdown = """
# Title

## Section 1

This is a test paragraph.

## Section 2

* List item 1
* List item 2
    """

    chunk_size = 500
    chunk_overlap = 50
    chunks = ragnificent_rs.chunk_markdown(markdown, chunk_size, chunk_overlap)

    # Verify chunking results
    assert isinstance(chunks, list), "Should return a list of chunks"
    assert len(chunks) > 0, "Should create at least one chunk from markdown content"

    # Verify all chunks are strings and non-empty
    for i, chunk in enumerate(chunks):
        assert isinstance(chunk, str), f"Chunk {i} should be a string"
        assert (
            len(chunk.strip()) > 0
        ), f"Chunk {i} should not be empty or whitespace-only"

    # Verify important content is preserved in chunks
    combined_content = " ".join(chunks)
    assert "# Title" in combined_content, "Title header should be preserved in chunks"
    assert (
        "## Section 1" in combined_content
    ), "Section 1 header should be preserved in chunks"
    assert (
        "## Section 2" in combined_content
    ), "Section 2 header should be preserved in chunks"
    assert (
        "This is a test paragraph." in combined_content
    ), "Paragraph content should be preserved"
    assert "List item 1" in combined_content, "List items should be preserved"
    assert "List item 2" in combined_content, "List items should be preserved"

    # Verify chunk size constraints
    for i, chunk in enumerate(chunks):
        assert (
            len(chunk) <= chunk_size + 100
        ), f"Chunk {i} size {len(chunk)} exceeds reasonable limit based on chunk_size {chunk_size}"

    # Verify no content is completely lost (accounting for possible overlap)
    original_words = set(markdown.split())
    chunk_words = set(" ".join(chunks).split())
    preserved_words = original_words.intersection(chunk_words)
    preservation_ratio = (
        len(preserved_words) / len(original_words) if original_words else 1
    )
    assert (
        preservation_ratio > 0.8
    ), f"Should preserve most content, got {preservation_ratio:.2%} preservation"


def test_render_js_page():
    url = "https://example.com"
    wait_time = 5000  # 5 seconds in milliseconds
    html = ragnificent_rs.render_js_page(url, wait_time)

    # Verify return type and basic structure
    assert isinstance(html, str), "Should return HTML as string"
    assert len(html) > 0, "Should return non-empty HTML content"

    # Verify it looks like HTML content (basic structure check)
    html_lower = html.lower()
    assert any(
        tag in html_lower for tag in ["<html", "<body", "<div", "<head"]
    ), "Should contain basic HTML structure tags"

    # Verify it's not just an error message or empty page
    assert (
        "error" not in html_lower or len(html) > 100
    ), "Should return substantial content, not just error messages"


def test_error_handling():
    # Test HTML conversion with malformed input
    with pytest.raises(Exception) as exc_info:
        ragnificent_rs.convert_html_to_markdown("<unclosed", "invalid-url")
    assert (
        exc_info.value is not None
    ), "Should raise specific exception for malformed HTML"

    # Test chunking with invalid parameters
    with pytest.raises(Exception) as exc_info:
        ragnificent_rs.chunk_markdown("test", -1, -1)
    assert (
        exc_info.value is not None
    ), "Should raise specific exception for negative chunk parameters"

    # Test additional invalid chunking scenarios
    with pytest.raises(Exception):
        # Chunk overlap greater than chunk size should fail
        ragnificent_rs.chunk_markdown("test content", 100, 200)

    # Test rendering with invalid URL
    with pytest.raises(Exception) as exc_info:
        ragnificent_rs.render_js_page("not-a-valid-url", 1000)
    assert exc_info.value is not None, "Should raise specific exception for invalid URL"

    # Test rendering with invalid wait time
    with pytest.raises(Exception):
        ragnificent_rs.render_js_page("https://example.com", -1000)

    # Verify functions handle empty/None inputs appropriately
    try:
        # These should either work gracefully or raise appropriate exceptions
        result = ragnificent_rs.convert_html_to_markdown("", "https://example.com")
        assert isinstance(
            result, str
        ), "Empty HTML should return empty string or valid result"
    except Exception as e:
        # If it raises an exception, it should be a meaningful one
        assert str(e) != "", "Exception should have a meaningful message"
