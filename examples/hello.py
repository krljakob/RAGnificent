#!/usr/bin/env python3
"""
Simple demo example of using RAGnificent to convert HTML to markdown.
"""

from RAGnificent.ragnificent_rs import convert_html_to_markdown

# Sample HTML
html = """
<html>
<head>
    <title>Hello RAGnificent</title>
</head>
<body>
    <h1>Hello from RAGnificent!</h1>
    <p>This is a simple example of converting HTML to Markdown.</p>
    <ul>
        <li>Simple to use</li>
        <li>Fast performance with Rust</li>
        <li>Multiple output formats</li>
    </ul>
</body>
</html>
"""

# Convert HTML to markdown
markdown = convert_html_to_markdown(html)

# Print the result
