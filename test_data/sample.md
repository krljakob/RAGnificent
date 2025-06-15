# Sample Markdown Document

This is a sample markdown document for testing semantic chunking and text processing.

## Introduction

Markdown is a lightweight markup language with plain text formatting syntax. It's designed to be converted to HTML and many other formats.

### Features

- Easy to read and write
- Platform independent
- Widely supported
- Great for documentation

## Technical Details

Markdown supports various formatting options including **bold text**, *italic text*, and `inline code`.

### Code Blocks

```python
def process_markdown(text):
    """Process markdown text and return formatted output."""
    lines = text.split('\n')
    processed = []
    for line in lines:
        if line.startswith('#'):
            processed.append(f"<h{line.count('#')}>{line.lstrip('#').strip()}</h{line.count('#')}>")
        else:
            processed.append(f"<p>{line}</p>")
    return '\n'.join(processed)
```

### Lists and Links

1. First item with [external link](https://example.com)
2. Second item with emphasis
3. Third item with code: `process_data()`

## Performance Considerations

When processing large documents, consider:

- Memory usage optimization
- Streaming processing for large files
- Caching frequently accessed data
- Parallel processing where applicable

## Conclusion

This sample document demonstrates various markdown features that can be used for testing chunking algorithms and text processing optimization.
EOF < /dev/null
