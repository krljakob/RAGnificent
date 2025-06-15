use scraper::{Html, Selector};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ParserError {
    #[error("Selector error: {0}")]
    SelectorError(String),

    #[error("Element not found: {0}")]
    NotFound(String),

    #[error("Other error: {0}")]
    Other(String),
}

/// Extracts the main content from an HTML document by identifying and
/// cleaning up the most relevant section.
/// 
/// This function is optimized for performance by:
/// 1. Using a static array for selectors
/// 2. Parsing selectors only once
/// 3. Early return on first match
/// 4. Avoiding unnecessary allocations
pub fn extract_main_content(html: &str) -> Result<Html, ParserError> {
    // Parse the document once
    let document = Html::parse_document(html);
    
    // Define content container selectors in order of preference
    static CONTAINER_SELECTORS: &[&str] = &[
        "main", 
        "article", 
        "#content", 
        ".content", 
        "body"
    ];
    
    // Try each selector in order of preference
    for &selector_str in CONTAINER_SELECTORS {
        // Parse selector and try to find the element
        if let Ok(selector) = Selector::parse(selector_str) {
            if let Some(element) = document.select(&selector).next() {
                // Found a main content container
                return Ok(Html::parse_fragment(&element.html()));
            }
        }
    }
    
    // Fallback: If no specific content container is found, return the whole document.
    // Note: Downstream consumers should handle processing of large documents gracefully.
    Ok(document)
}

/// Cleans up HTML by removing unwanted elements like scripts, ads, etc.
/// 
/// This function is optimized for performance by:
/// 1. Using a single document parse
/// 2. String-based replacement for efficiency
/// 3. Pre-compiled selectors
/// 4. Minimizing allocations
pub fn clean_html(html: &str) -> Result<String, ParserError> {
    // Pre-allocate selectors to avoid re-parsing
    static UNWANTED_SELECTORS: &[&str] = &[
        "script", "style", "iframe", "noscript",
        ".advertisement", ".ad", ".banner", "#cookie-notice",
        "header", "footer", "nav", ".sidebar", 
        ".menu", ".comments", ".related", ".share", ".social",
    ];

    // Parse the document once
    let document = Html::parse_document(html);
    
    // Pre-parse all selectors once
    let selectors: Vec<Selector> = UNWANTED_SELECTORS
        .iter()
        .filter_map(|&s| Selector::parse(s).ok())
        .collect();
    
    // Start with the original HTML
    let mut cleaned_html = document.html();
    
    // Remove unwanted elements by replacing their HTML with empty string
    for selector in &selectors {
        for element in document.select(selector) {
            let element_html = element.html();
            // Replace all occurrences of this element's HTML
            cleaned_html = cleaned_html.replace(&element_html, "");
        }
    }
    
    // Return the cleaned HTML
    Ok(cleaned_html)
}

/// Extracts all unique links from the HTML document
/// 
/// This function is optimized for performance by:
/// 1. Using a HashSet for O(1) lookups to avoid duplicates
/// 2. Minimizing string allocations
/// 3. Reusing the same selector
pub fn extract_links(html: &str, base_url: &str) -> Result<Vec<String>, ParserError> {
    use std::collections::HashSet;
    use url::Url;
    
    // Parse the document once
    let document = Html::parse_document(html);
    
    // Parse base URL once
    let base_url = Url::parse(base_url).map_err(|e| ParserError::Other(e.to_string()))?;
    
    // Create selector once
    static LINK_SELECTOR: &str = "a[href]";
    let selector = Selector::parse(LINK_SELECTOR)
        .map_err(|e| ParserError::SelectorError(e.to_string()))?;
    
    // Use HashSet to avoid duplicates
    let mut unique_links = HashSet::with_capacity(50); // Pre-allocate based on expected size
    
    // Collect all links
    for element in document.select(&selector) {
        if let Some(href) = element.value().attr("href") {
            // Skip javascript and fragment links
            if href.starts_with("javascript:") || href.starts_with('#') {
                continue;
            }
            
            // Process the URL
            let url = if href.starts_with("http") {
                // For absolute URLs, parse directly
                match Url::parse(href) {
                    Ok(url) => url.to_string(),
                    Err(_) => continue, // Skip invalid URLs
                }
            } else {
                // For relative URLs, resolve against base URL
                match base_url.join(href) {
                    Ok(url) => url.to_string(),
                    Err(_) => continue, // Skip invalid relative URLs
                }
            };
            
            // Add to set (automatically handles duplicates)
            unique_links.insert(url);
        }
    }
    
    // Convert HashSet to Vec for the return value
    Ok(unique_links.into_iter().collect())
}

/// Utility function to get text content of an element, cleaning up whitespace
pub fn get_element_text(element: &scraper::ElementRef) -> String {
    element
        .text()
        .collect::<Vec<_>>()
        .join(" ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}
