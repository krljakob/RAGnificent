use pyo3::prelude::*;

#[cfg(test)]
mod tests;

pub mod chunker;
pub mod html_parser;
pub mod js_renderer;
pub mod markdown_converter;

/// output formats for conversion
#[pyclass]
#[derive(Clone, Copy)]
pub enum OutputFormat {
    Markdown = 0,
    Json = 1,
    Xml = 2,
}

#[pymethods]
impl OutputFormat {
    /// convert string to OutputFormat
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

/// rust implementations for RAGnificent HTML conversion and chunking
#[pymodule]
fn ragnificent_rs(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<OutputFormat>()?;
    m.add_function(wrap_pyfunction!(convert_html_to_markdown, py)?)?;
    m.add_function(wrap_pyfunction!(convert_html_to_format, py)?)?;
    m.add_function(wrap_pyfunction!(chunk_markdown, py)?)?;
    m.add_function(wrap_pyfunction!(render_js_page, py)?)?;
    Ok(())
}

/// convert HTML to markdown
#[pyfunction]
pub fn convert_html_to_markdown(html: &str, base_url: &str) -> PyResult<String> {
    let result = markdown_converter::convert_to_markdown(html, base_url)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(result)
}

/// converts HTML content to the specified format (markdown, JSON, or XML)
///
/// args:
///     html (str): The HTML content to convert
///     base_url (str): The base URL for resolving relative links
///     format (str, optional): The output format - "markdown" (default), "json", or "xml"
///
/// returns:
///     str: The converted content in the specified format
///
/// raises:
///     RuntimeError: If conversion fails
///
/// example:
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

/// chunks markdown content into semantic sections for RAG (Retrieval Augmented Generation)
///
/// args:
///     markdown (str): The markdown content to chunk
///     chunk_size (int): Maximum size of chunks in characters
///     chunk_overlap (int): Overlap between chunks in characters
///
/// returns:
///     list[str]: List of markdown chunks
///
/// raises:
///     runtimeError: If chunking fails
///
/// example:
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

/// renders a JavaScript-enabled page and returns the HTML content
///
/// this function requires the 'real_rendering' feature to be enabled during compilation.
/// it uses a headless browser to execute JavaScript and return the fully rendered HTML.
///
/// args:
///     url (str): The URL of the page to render
///     wait_time (int, optional): Time to wait for JavaScript execution in milliseconds (default: 2000)
///
/// returns:
///     str: The fully rendered HTML content
///
/// raises:
///     runtimeError: If rendering fails or if the 'real_rendering' feature is not enabled
///
/// example:
///     >>> html = render_js_page("https://example.com/js-heavy-page", 5000)
///     >>> "dynamically-generated-content" in html
///     true
#[pyfunction]
fn render_js_page(url: &str, wait_time: Option<u64>) -> PyResult<String> {
    let runtime = tokio::runtime::Runtime::new()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let html = runtime
        .block_on(async { js_renderer::render_page(url, wait_time.unwrap_or(2000)).await })
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(html)
}
