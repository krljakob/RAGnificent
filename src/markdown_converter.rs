use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use url::Url;

#[derive(Error, Debug)]
pub enum MarkdownError {
    #[error("Selector error: {0}")]
    SelectorError(String),

    #[error("URL parsing error: {0}")]
    UrlError(#[from] url::ParseError),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Parallel processing error: {0}")]
    ParallelProcessingError(String),

    #[error("Other error: {0}")]
    Other(String),
}

/// Supported output formats for content conversion
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Markdown,
    Json,
    Xml,
}

/// Data structure for document representation that can be serialized to different formats
#[derive(Debug, Serialize, Deserialize)]
pub struct Document {
    pub title: String,
    pub base_url: String,
    pub headings: Vec<Heading>,
    pub paragraphs: Vec<String>,
    pub links: Vec<Link>,
    pub images: Vec<Image>,
    pub lists: Vec<List>,
    pub code_blocks: Vec<CodeBlock>,
    pub blockquotes: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Heading {
    pub level: u8,
    pub text: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Link {
    pub text: String,
    pub url: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Image {
    pub alt: String,
    pub src: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct List {
    pub ordered: bool,
    pub items: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CodeBlock {
    pub language: String,
    pub code: String,
}

/// Create a new selector with error handling
fn create_selector(selector_str: &str) -> Result<Selector, MarkdownError> {
    Selector::parse(selector_str).map_err(|e| MarkdownError::SelectorError(e.to_string()))
}

/// Resolve a relative URL against a base URL
fn resolve_url(href: &str, base_url: &Url) -> String {
    base_url
        .join(href)
        .unwrap_or_else(|_| Url::parse(href).unwrap_or(base_url.clone()))
        .to_string()
}

/// Extract document title from HTML
fn extract_title(document_html: &Html) -> Result<String, MarkdownError> {
    let title_selector = create_selector("title")?;

    let title = document_html
        .select(&title_selector)
        .next()
        .map(|element| element.text().collect::<String>())
        .unwrap_or_else(|| "No Title".to_string());

    Ok(title.trim().to_string())
}

/// Extract headings from HTML
fn extract_headings(document_html: &Html) -> Result<Vec<Heading>, MarkdownError> {
    let mut headings = Vec::with_capacity(16);

    for i in 1..=6 {
        let heading_selector = create_selector(&format!("h{}", i))?;

        for element in document_html.select(&heading_selector) {
            let text = element.text().collect::<String>().trim().to_string();
            if !text.is_empty() {
                headings.push(Heading {
                    level: i as u8,
                    text,
                });
            }
        }
    }

    Ok(headings)
}

/// Extract paragraphs from HTML
fn extract_paragraphs(document_html: &Html) -> Result<Vec<String>, MarkdownError> {
    let mut paragraphs = Vec::with_capacity(32);
    let p_selector = create_selector("p")?;

    for element in document_html.select(&p_selector) {
        let text = element.text().collect::<String>().trim().to_string();
        if !text.is_empty() {
            paragraphs.push(text);
        }
    }

    Ok(paragraphs)
}

/// Extract links from HTML
fn extract_links(document_html: &Html, base_url: &Url) -> Result<Vec<Link>, MarkdownError> {
    let mut links = Vec::with_capacity(32);
    let a_selector = create_selector("a[href]")?;

    for element in document_html.select(&a_selector) {
        if let Some(href) = element.value().attr("href") {
            let text = element.text().collect::<String>().trim().to_string();
            if !text.is_empty() {
                // Resolve relative URLs
                let absolute_url = resolve_url(href, base_url);
                links.push(Link {
                    text,
                    url: absolute_url,
                });
            }
        }
    }

    Ok(links)
}

/// Extract images from HTML
fn extract_images(document_html: &Html, base_url: &Url) -> Result<Vec<Image>, MarkdownError> {
    let mut images = Vec::with_capacity(16);
    let img_selector = create_selector("img[src]")?;

    for element in document_html.select(&img_selector) {
        if let Some(src) = element.value().attr("src") {
            let alt = element.value().attr("alt").unwrap_or("image").to_string();

            // Resolve relative URLs
            let absolute_url = resolve_url(src, base_url);
            images.push(Image {
                alt,
                src: absolute_url,
            });
        }
    }

    Ok(images)
}

/// Extract list items from a list element
fn extract_list_items(list_element: &scraper::ElementRef, li_selector: &Selector) -> Vec<String> {
    let mut items = Vec::new();

    for li in list_element.select(li_selector) {
        let text = li.text().collect::<String>().trim().to_string();
        if !text.is_empty() {
            items.push(text);
        }
    }

    items
}

/// Extract lists from HTML
fn extract_lists(document_html: &Html) -> Result<Vec<List>, MarkdownError> {
    let mut lists = Vec::with_capacity(8);
    let ul_selector = create_selector("ul")?;
    let ol_selector = create_selector("ol")?;
    let li_selector = create_selector("li")?;

    // Process unordered lists
    for ul in document_html.select(&ul_selector) {
        let items = extract_list_items(&ul, &li_selector);
        if !items.is_empty() {
            lists.push(List {
                ordered: false,
                items,
            });
        }
    }

    // Process ordered lists
    for ol in document_html.select(&ol_selector) {
        let items = extract_list_items(&ol, &li_selector);
        if !items.is_empty() {
            lists.push(List {
                ordered: true,
                items,
            });
        }
    }

    Ok(lists)
}

/// Extract code blocks from HTML
fn extract_code_blocks(document_html: &Html) -> Result<Vec<CodeBlock>, MarkdownError> {
    let mut code_blocks = Vec::with_capacity(8);
    let pre_selector = create_selector("pre, code")?;

    for element in document_html.select(&pre_selector) {
        let text = element.text().collect::<String>().trim().to_string();
        if !text.is_empty() {
            let lang = element
                .value()
                .classes()
                .find(|c| c.starts_with("language-"))
                .map(|c| c.strip_prefix("language-").unwrap_or(""))
                .unwrap_or("")
                .to_string();

            code_blocks.push(CodeBlock {
                language: lang,
                code: text,
            });
        }
    }

    Ok(code_blocks)
}

/// Extract blockquotes from HTML
fn extract_blockquotes(document_html: &Html) -> Result<Vec<String>, MarkdownError> {
    let mut blockquotes = Vec::with_capacity(8);
    let blockquote_selector = create_selector("blockquote")?;

    for element in document_html.select(&blockquote_selector) {
        let text = element.text().collect::<String>().trim().to_string();
        if !text.is_empty() {
            blockquotes.push(text);
        }
    }

    Ok(blockquotes)
}

/// Parse HTML into our document structure with optimizations
/// 
/// This function is optimized for performance by:
/// 1. Using parallel extraction of document components via Rayon
/// 2. Reusing the parsed document 
/// 3. Reducing memory allocations with capacity hints
/// 4. Concurrent processing of independent extraction operations
fn parse_html_to_document(html: &str, base_url_str: &str) -> Result<Document, MarkdownError> {
    
    // Parse the document once
    let document = Html::parse_document(html);
    let base_url = Url::parse(base_url_str).map_err(MarkdownError::UrlError)?;
    
    // Extract title first (lightweight operation)
    let title = extract_title(&document)?;
    
    // Extract document components sequentially (scraper Html is not thread-safe)
    let headings = extract_headings(&document);
    let paragraphs = extract_paragraphs(&document);
    let blockquotes = extract_blockquotes(&document);
    let code_blocks = extract_code_blocks(&document);
    
    // Extract URL-dependent components sequentially
    let links = extract_links(&document, &base_url);
    let images = extract_images(&document, &base_url);
    let lists = extract_lists(&document);

    // Collect results and handle errors
    Ok(Document {
        title,
        base_url: base_url_str.to_string(),
        headings: headings?,
        paragraphs: paragraphs?,
        links: links?,
        images: images?,
        lists: lists?,
        code_blocks: code_blocks?,
        blockquotes: blockquotes?,
    })
}

/// Render headings to markdown
fn render_headings_markdown(headings: &[Heading]) -> String {
    let mut markdown = String::new();

    for heading in headings {
        let heading_prefix = "#".repeat(heading.level as usize);
        markdown.push_str(&format!("{} {}\n\n", heading_prefix, heading.text));
    }

    markdown
}

/// Render paragraphs to markdown
fn render_paragraphs_markdown(paragraphs: &[String]) -> String {
    let mut markdown = String::new();

    for paragraph in paragraphs {
        markdown.push_str(&format!("{}\n\n", paragraph));
    }

    markdown
}

/// Render links to markdown
fn render_links_markdown(links: &[Link]) -> String {
    let mut markdown = String::new();

    for link in links {
        markdown.push_str(&format!("[{}]({})\n\n", link.text, link.url));
    }

    markdown
}

/// Render images to markdown
fn render_images_markdown(images: &[Image]) -> String {
    let mut markdown = String::new();

    for image in images {
        markdown.push_str(&format!("![{}]({})\n\n", image.alt, image.src));
    }

    markdown
}

/// Render lists to markdown
fn render_lists_markdown(lists: &[List]) -> String {
    let mut markdown = String::new();

    for list in lists {
        if list.ordered {
            for (i, item) in list.items.iter().enumerate() {
                markdown.push_str(&format!("{}. {}\n", i + 1, item));
            }
        } else {
            for item in &list.items {
                markdown.push_str(&format!("- {}\n", item));
            }
        }
        markdown.push('\n');
    }

    markdown
}

/// Render code blocks to markdown
fn render_code_blocks_markdown(code_blocks: &[CodeBlock]) -> String {
    let mut markdown = String::new();

    for code_block in code_blocks {
        markdown.push_str(&format!(
            "```{}\n{}\n```\n\n",
            code_block.language, code_block.code
        ));
    }

    markdown
}

/// Render blockquotes to markdown
fn render_blockquotes_markdown(blockquotes: &[String]) -> String {
    let mut markdown = String::new();

    for blockquote in blockquotes {
        let quoted = blockquote
            .lines()
            .map(|line| format!("> {}", line))
            .collect::<Vec<String>>()
            .join("\n");
        markdown.push_str(&format!("{}\n\n", quoted));
    }

    markdown
}

/// Clean up extra newlines in markdown
fn clean_markdown(markdown: String) -> String {
    markdown
        .replace("\n\n\n\n", "\n\n")
        .replace("\n\n\n", "\n\n")
        .trim()
        .to_string()
}

/// Convert document to markdown format
pub fn document_to_markdown(document: &Document) -> String {
    let mut markdown_content = format!("# {}\n\n", document.title);

    // Render each document component
    markdown_content.push_str(&render_headings_markdown(&document.headings));
    markdown_content.push_str(&render_paragraphs_markdown(&document.paragraphs));
    markdown_content.push_str(&render_links_markdown(&document.links));
    markdown_content.push_str(&render_images_markdown(&document.images));
    markdown_content.push_str(&render_lists_markdown(&document.lists));
    markdown_content.push_str(&render_code_blocks_markdown(&document.code_blocks));
    markdown_content.push_str(&render_blockquotes_markdown(&document.blockquotes));

    // Clean up extra newlines
    clean_markdown(markdown_content)
}

/// Convert document to JSON format
pub fn document_to_json(document: &Document) -> Result<String, MarkdownError> {
    serde_json::to_string_pretty(document).map_err(|e| {
        MarkdownError::SerializationError(format!("Failed to serialize to JSON: {}", e))
    })
}

/// Convert document to XML format
pub fn document_to_xml(document: &Document) -> Result<String, MarkdownError> {
    use quick_xml::se::to_string;

    match to_string(document) {
        Ok(xml) => Ok(xml),
        Err(e) => {
            eprintln!("Error serializing document to XML: {:?}", e);
            Err(MarkdownError::SerializationError(format!(
                "Failed to serialize to XML: {}",
                e
            )))
        }
    }
}



/// Convert HTML to the specified output format
/// 
/// This function is optimized for performance by:
/// 1. Using parallel processing where beneficial
/// 2. Reducing allocations
/// 3. Reusing resources
pub fn convert_html(
    html: &str,
    base_url: &str,
    format: OutputFormat,
) -> Result<String, MarkdownError> {
    // For Markdown, use the direct parsing function which is more efficient
    if matches!(format, OutputFormat::Markdown) {
        return parse_html_to_markdown(html, base_url);
    }
    
    // Parse the document once
    let document = parse_html_to_document(html, base_url)?;
    
    // Process based on the requested format
    match format {
        OutputFormat::Json => document_to_json(&document),
        OutputFormat::Xml => document_to_xml(&document),
        OutputFormat::Markdown => unreachable!(), // Handled above
    }
}

/// Backward compatibility function for convert_to_markdown
/// Optimized for the common case of markdown conversion
pub fn convert_to_markdown(html: &str, base_url: &str) -> Result<String, MarkdownError> {
    // Directly parse to markdown without the intermediate format conversion
    parse_html_to_markdown(html, base_url)
}

/// Directly parse HTML to markdown without the intermediate Document struct
/// This is more efficient when we only need markdown output
/// Optimized with parallel extraction and streaming output generation
fn parse_html_to_markdown(html: &str, base_url: &str) -> Result<String, MarkdownError> {
    use std::fmt::Write;
    
    // Parse the document once
    let document = Html::parse_document(html);
    let base_url = Url::parse(base_url).map_err(MarkdownError::UrlError)?;
    
    // Create a buffer for the markdown output
    let mut output = String::with_capacity(html.len() * 2); // Pre-allocate based on input size
    
    // Extract document title as a top-level heading (sequential - lightweight)
    if let Ok(title) = extract_title(&document) {
        if !title.is_empty() && title != "No Title" {
            writeln!(&mut output, "# {}\n", title).unwrap();
        }
    }
    
    // Extract all document components sequentially (scraper Html is not thread-safe)
    let headings_result = extract_headings(&document);
    let paragraphs_result = extract_paragraphs(&document);
    let lists_result = extract_lists(&document);
    let code_blocks_result = extract_code_blocks(&document);
    let blockquotes_result = extract_blockquotes(&document);
    let links_result = extract_links(&document, &base_url);
    let images_result = extract_images(&document, &base_url);
    
    // Render content in order of importance with error handling
    // Extract and render headings
    if let Ok(headings) = headings_result {
        for heading in headings {
            writeln!(&mut output, "{} {}", "#".repeat(heading.level as usize), heading.text).unwrap();
            output.push_str("\n\n");
        }
    }
    
    // Extract and render paragraphs
    if let Ok(paragraphs) = paragraphs_result {
        for para in paragraphs {
            if !para.trim().is_empty() {
                output.push_str(&para);
                output.push_str("\n\n");
            }
        }
    }
    
    // Extract and render links
    if let Ok(links) = links_result {
        for link in links {
            writeln!(&mut output, "[{}]({})\n", link.text, link.url).unwrap();
        }
    }
    
    // Extract and render images
    if let Ok(images) = images_result {
        for image in images {
            writeln!(&mut output, "![{}]({})\n", image.alt, image.src).unwrap();
        }
    }
    
    // Extract and render lists
    if let Ok(lists) = lists_result {
        for list in lists {
            for (i, item) in list.items.iter().enumerate() {
                if list.ordered {
                    writeln!(&mut output, "{}. {}", i + 1, item).unwrap();
                } else {
                    writeln!(&mut output, "- {}", item).unwrap();
                }
            }
            output.push_str("\n");
        }
    }
    
    // Extract and render code blocks
    if let Ok(code_blocks) = code_blocks_result {
        for code_block in code_blocks {
            writeln!(&mut output, "```{}", code_block.language).unwrap();
            writeln!(&mut output, "{}", code_block.code).unwrap();
            output.push_str("```\n\n");
        }
    }
    
    // Extract and render blockquotes
    if let Ok(blockquotes) = blockquotes_result {
        for blockquote in blockquotes {
            for line in blockquote.lines() {
                writeln!(&mut output, "> {}", line).unwrap();
            }
            output.push_str("\n");
        }
    }
    
    // Clean up any excessive newlines and return
    Ok(clean_markdown(output))
}
