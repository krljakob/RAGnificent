use pyo3::prelude::*;

#[cfg(test)]
mod tests;

pub mod chunker;
pub mod html_parser;
pub mod js_renderer;
pub mod markdown_converter;

/// Python-friendly enumeration of output formats
///
/// Attributes:
///     MARKDOWN (int): Format for Markdown output (0)
///     JSON (int): Format for JSON output (1)
///     XML (int): Format for XML output (2)
#[pyclass]
#[derive(Clone, Copy)]
pub enum OutputFormat {
    Markdown = 0,
    Json = 1,
    Xml = 2,
}

#[pymethods]
impl OutputFormat {
    /// Convert a string representation to an OutputFormat
    ///
    /// Args:
    ///     format_str (str): String representation of format ("markdown", "json", or "xml")
    ///
    /// Returns:
    ///     OutputFormat: The corresponding OutputFormat enum value
    ///
    /// Example:
    ///     >>> format = OutputFormat.from_str("json")
    ///     >>> print(format)  # OutputFormat.JSON
    #[staticmethod]
    fn from_str(format_str: &str) -> Self {
        match format_str.to_lowercase().as_str() {
            "json" => OutputFormat::Json,
            "xml" => OutputFormat::Xml,
            _ => OutputFormat::Markdown,
        }
    }
}

impl From<OutputFormat> for markdown_converter::OutputFormat {
    fn from(py_format: OutputFormat) -> Self {
        match py_format {
            OutputFormat::Markdown => markdown_converter::OutputFormat::Markdown,
            OutputFormat::Json => markdown_converter::OutputFormat::Json,
            OutputFormat::Xml => markdown_converter::OutputFormat::Xml,
        }
    }
}

/// A Python module implemented in Rust for RAGnificent.
///
/// This module provides high-performance implementations of key RAGnificent functions:
/// - HTML to Markdown/JSON/XML conversion
/// - Semantic text chunking for RAG applications
/// - JavaScript page rendering (when compiled with the 'real_rendering' feature)
///
/// Example:
///     >>> from ragnificent_rs import convert_html_to_markdown, OutputFormat
///     >>> html = "<h1>Title</h1><p>Content</p>"
///     >>> markdown = convert_html_to_markdown(html, "https://example.com")
///     >>> json_output = convert_html_to_format(html, "https://example.com", "json")
#[pymodule]
fn ragnificent_rs(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<OutputFormat>()?;
    m.add_function(wrap_pyfunction!(convert_html_to_markdown, py)?)?;
    m.add_function(wrap_pyfunction!(convert_html_to_format, py)?)?;
    m.add_function(wrap_pyfunction!(chunk_markdown, py)?)?;
    m.add_function(wrap_pyfunction!(render_js_page, py)?)?;
    Ok(())
}

/// Converts HTML content to markdown format
///
/// Args:
///     html (str): The HTML content to convert
///     base_url (str): The base URL for resolving relative links
///
/// Returns:
///     str: The converted markdown content
///
/// Raises:
///     RuntimeError: If conversion fails
#[pyfunction]
pub fn convert_html_to_markdown(html: &str, base_url: &str) -> PyResult<String> {
    let result = markdown_converter::convert_to_markdown(html, base_url)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(result)
}

/// Converts HTML content to the specified format (markdown, JSON, or XML)
///
/// Args:
///     html (str): The HTML content to convert
///     base_url (str): The base URL for resolving relative links
///     format (str, optional): The output format - "markdown" (default), "json", or "xml"
///
/// Returns:
///     str: The converted content in the specified format
///
/// Raises:
///     RuntimeError: If conversion fails
///
/// Example:
///     >>> html = "<h1>Title</h1><p>Content with <a href='/link'>link</a></p>"
///     >>> json_content = convert_html_to_format(html, "https://example.com", "json")
///     >>> xml_content = convert_html_to_format(html, "https://example.com", "xml")
#[pyfunction]
fn convert_html_to_format(html: &str, base_url: &str, format: Option<String>) -> PyResult<String> {
    let output_format = match format.as_deref() {
        Some("json") => markdown_converter::OutputFormat::Json,
        Some("xml") => markdown_converter::OutputFormat::Xml,
        _ => markdown_converter::OutputFormat::Markdown,
    };

    let result = markdown_converter::convert_html(html, base_url, output_format)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(result)
}

/// Chunks markdown content into semantic sections for RAG (Retrieval Augmented Generation)
///
/// Args:
///     markdown (str): The markdown content to chunk
///     chunk_size (int): Maximum size of chunks in characters
///     chunk_overlap (int): Overlap between chunks in characters
///
/// Returns:
///     List[str]: List of markdown chunks
///
/// Raises:
///     RuntimeError: If chunking fails
///
/// Example:
///     >>> markdown = "# Title\n\nContent paragraph 1\n\n## Section\n\nContent paragraph 2"
///     >>> chunks = chunk_markdown(markdown, 1000, 200)
///     >>> len(chunks)
///     2
#[pyfunction]
fn chunk_markdown(
    markdown: &str,
    chunk_size: usize,
    chunk_overlap: usize,
) -> PyResult<Vec<String>> {
    let chunks = chunker::create_semantic_chunks(markdown, chunk_size, chunk_overlap)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(chunks)
}

/// Renders a JavaScript-enabled page and returns the HTML content
///
/// This function requires the 'real_rendering' feature to be enabled during compilation.
/// It uses a headless browser to execute JavaScript and return the fully rendered HTML.
///
/// Args:
///     url (str): The URL of the page to render
///     wait_time (int, optional): Time to wait for JavaScript execution in milliseconds (default: 2000)
///
/// Returns:
///     str: The fully rendered HTML content
///
/// Raises:
///     RuntimeError: If rendering fails or if the 'real_rendering' feature is not enabled
///
/// Example:
///     >>> html = render_js_page("https://example.com/js-heavy-page", 5000)
///     >>> "dynamically-generated-content" in html
///     True
#[pyfunction]
fn render_js_page(url: &str, wait_time: Option<u64>) -> PyResult<String> {
    let runtime = tokio::runtime::Runtime::new()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let html = runtime
        .block_on(async { js_renderer::render_page(url, wait_time.unwrap_or(2000)).await })
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(html)
}
