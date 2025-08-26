import pytest

from RAGnificent.core.scraper import MarkdownScraper


def test_markdown_conversion_no_duplicate_elements():
    html = """
    <html>
      <head><title>Page</title></head>
      <body>
        <main>
          <h1>Header</h1>
          <p>One</p>
          <p>Two</p>
        </main>
      </body>
    </html>
    """
    s = MarkdownScraper()
    md = s.convert_to_markdown(html, url="https://example.com")
    # Expect title once and paragraphs once each
    assert md.count("# Header") == 1
    assert md.count("One") == 1
    assert md.count("Two") == 1
