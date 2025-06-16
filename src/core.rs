use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;

use crate::html_parser;
use crate::markdown_converter;
use crate::chunker;

/// Main error type for the Markdown Lab engine
#[derive(Error, Debug)]
pub enum MarkdownLabError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("URL parse error: {0}")]
    UrlParse(#[from] url::ParseError),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON serialization error: {0}")]
    JsonSerde(#[from] serde_json::Error),
    #[error("XML serialization error: {0}")]
    XmlSerde(#[from] quick_xml::Error),
    #[error("HTML parsing error: {0}")]
    HtmlParse(String),
    #[error("Chunking error: {0}")]
    Chunking(String),
    #[error("Configuration error: {0}")]
    Config(String),
    #[error("Network timeout")]
    Timeout,
    #[error("Rate limit exceeded")]
    RateLimit,
}

/// Output format for converted content
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Markdown,
    Json,
    Xml,
}

impl From<&str> for OutputFormat {
    fn from(format_str: &str) -> Self {
        match format_str.to_lowercase().as_str() {
            "json" => OutputFormat::Json,
            "xml" => OutputFormat::Xml,
            _ => OutputFormat::Markdown,
        }
    }
}

/// Configuration for the Markdown Lab engine
#[derive(Debug, Clone)]
pub struct Config {
    /// Maximum requests per second
    pub requests_per_second: f64,
    /// Request timeout in seconds
    pub timeout: Duration,
    /// Maximum number of retry attempts
    pub max_retries: u32,
    /// Maximum chunk size for content chunking
    pub chunk_size: usize,
    /// Overlap between chunks
    pub chunk_overlap: usize,
    /// Whether caching is enabled
    pub cache_enabled: bool,
    /// Maximum age for cached responses
    pub cache_max_age: Duration,
    /// Maximum number of concurrent requests
    pub max_workers: usize,
    /// User agent string
    pub user_agent: String,
    /// Domain-specific rate limits
    pub domain_limits: HashMap<String, f64>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            requests_per_second: 1.0,
            timeout: Duration::from_secs(30),
            max_retries: 3,
            chunk_size: 1000,
            chunk_overlap: 200,
            cache_enabled: true,
            cache_max_age: Duration::from_secs(3600),
            max_workers: 8,
            user_agent: "Mozilla/5.0 (RAGnificent/1.0) Rust/Web-Scraper".to_string(),
            domain_limits: HashMap::new(),
        }
    }
}

/// Filters for sitemap processing
#[derive(Debug, Clone, Default)]
pub struct SitemapFilters {
    /// Minimum priority (0.0 to 1.0)
    pub min_priority: Option<f64>,
    /// Include patterns (regex)
    pub include_patterns: Vec<String>,
    /// Exclude patterns (regex)  
    pub exclude_patterns: Vec<String>,
    /// Maximum number of URLs to process
    pub limit: Option<usize>,
}

/// Options for content chunking
#[derive(Debug, Clone)]
pub struct ChunkOptions {
    /// Maximum chunk size in characters
    pub size: usize,
    /// Overlap between chunks in characters
    pub overlap: usize,
    /// Whether to preserve semantic boundaries
    pub semantic_splitting: bool,
}

impl Default for ChunkOptions {
    fn default() -> Self {
        Self {
            size: 1000,
            overlap: 200,
            semantic_splitting: true,
        }
    }
}

/// A processed document with metadata
#[derive(Debug, Clone)]
pub struct Document {
    /// The original URL
    pub url: String,
    /// Document title
    pub title: Option<String>,
    /// Processed content
    pub content: String,
    /// Content format
    pub format: OutputFormat,
    /// Processing timestamp
    pub timestamp: std::time::SystemTime,
    /// Metadata extracted during processing
    pub metadata: HashMap<String, String>,
}

/// A content chunk for RAG applications
#[derive(Debug, Clone)]
pub struct Chunk {
    /// Chunk content
    pub content: String,
    /// Source URL
    pub source_url: String,
    /// Chunk index in the document
    pub index: usize,
    /// Start position in original content
    pub start_pos: usize,
    /// End position in original content
    pub end_pos: usize,
    /// Chunk metadata
    pub metadata: HashMap<String, String>,
}

/// HTTP client with connection pooling and rate limiting
pub struct HttpClient {
    client: reqwest::Client,
    cache: Option<Arc<tokio::sync::RwLock<HashMap<String, (String, std::time::Instant)>>>>,
    rate_limiter: Arc<tokio::sync::Semaphore>,
    config: Arc<Config>,
}

impl HttpClient {
    /// Create a new HTTP client with the given configuration
    pub fn new(config: Arc<Config>) -> Result<Self, MarkdownLabError> {
        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .user_agent(&config.user_agent)
            .pool_max_idle_per_host(config.max_workers)
            .build()?;

        let cache = if config.cache_enabled {
            Some(Arc::new(tokio::sync::RwLock::new(HashMap::new())))
        } else {
            None
        };

        let rate_limiter = Arc::new(tokio::sync::Semaphore::new(config.max_workers));

        Ok(Self {
            client,
            cache,
            rate_limiter,
            config,
        })
    }

    /// Fetch content from a single URL
    pub async fn get(&self, url: &str) -> Result<String, MarkdownLabError> {
        // Check cache first
        if let Some(cached) = self.check_cache(url).await {
            return Ok(cached);
        }

        // Rate limiting
        let _permit = self.rate_limiter.acquire().await.unwrap();
        
        // Add delay for rate limiting
        let delay = Duration::from_millis((1000.0 / self.config.requests_per_second) as u64);
        tokio::time::sleep(delay).await;

        // Fetch with retries
        let mut last_error = None;
        for attempt in 0..self.config.max_retries {
            match self.client.get(url).send().await {
                Ok(response) => {
                    if response.status().is_success() {
                        let content = response.text().await?;
                        
                        // Cache the response
                        self.cache_response(url, &content).await;
                        
                        return Ok(content);
                    } else {
                        last_error = Some(MarkdownLabError::Http(
                            reqwest::Error::from(response.error_for_status().unwrap_err())
                        ));
                    }
                }
                Err(e) => {
                    last_error = Some(MarkdownLabError::Http(e));
                    if attempt < self.config.max_retries - 1 {
                        let backoff = Duration::from_millis(1000 * (1 << attempt));
                        tokio::time::sleep(backoff).await;
                    }
                }
            }
        }

        Err(last_error.unwrap())
    }

    /// Fetch content from multiple URLs concurrently
    pub async fn get_many(&self, urls: Vec<&str>) -> Result<Vec<(String, Result<String, MarkdownLabError>)>, MarkdownLabError> {
        let futures = urls.into_iter().map(|url| {
            let url = url.to_string();
            async move {
                let result = self.get(&url).await;
                (url, result)
            }
        });

        let results = futures::future::join_all(futures).await;
        Ok(results)
    }

    async fn check_cache(&self, url: &str) -> Option<String> {
        if let Some(cache) = &self.cache {
            let cache_read = cache.read().await;
            if let Some((content, timestamp)) = cache_read.get(url) {
                if timestamp.elapsed() < self.config.cache_max_age {
                    return Some(content.clone());
                }
            }
        }
        None
    }

    async fn cache_response(&self, url: &str, content: &str) {
        if let Some(cache) = &self.cache {
            let mut cache_write = cache.write().await;
            cache_write.insert(url.to_string(), (content.to_string(), std::time::Instant::now()));
        }
    }
}

/// HTML processor for converting HTML to various formats
pub struct HtmlProcessor {
    config: Arc<Config>,
}

impl HtmlProcessor {
    pub fn new(config: Arc<Config>) -> Self {
        Self { config }
    }

    /// Extract structured content from HTML
    pub fn extract_content(&self, html: &str, base_url: &str) -> Result<Document, MarkdownLabError> {
        // Parse HTML using scraper crate for performance
        let document = scraper::Html::parse_document(html);
        
        // Extract title
        let title_selector = scraper::Selector::parse("title").unwrap();
        let title = document
            .select(&title_selector)
            .next()
            .map(|element| element.text().collect::<String>().trim().to_string());

        // Extract main content (remove scripts, styles, etc.)
        let content_html_doc = html_parser::extract_main_content(html)
            .map_err(|e| MarkdownLabError::HtmlParse(e.to_string()))?;
        
        // Extract HTML string from the parsed document
        let content_html = content_html_doc.html();

        Ok(Document {
            url: base_url.to_string(),
            title,
            content: content_html,
            format: OutputFormat::Markdown, // Will be converted later
            timestamp: std::time::SystemTime::now(),
            metadata: HashMap::new(),
        })
    }

    /// Convert document to markdown format
    pub fn to_markdown(&self, doc: &Document) -> Result<String, MarkdownLabError> {
        markdown_converter::convert_to_markdown(&doc.content, &doc.url)
            .map_err(|e| MarkdownLabError::HtmlParse(e.to_string()))
    }

    /// Convert document to JSON format
    pub fn to_json(&self, doc: &Document) -> Result<String, MarkdownLabError> {
        markdown_converter::convert_html(&doc.content, &doc.url, markdown_converter::OutputFormat::Json)
            .map_err(|e| MarkdownLabError::HtmlParse(e.to_string()))
    }

    /// Convert document to XML format  
    pub fn to_xml(&self, doc: &Document) -> Result<String, MarkdownLabError> {
        markdown_converter::convert_html(&doc.content, &doc.url, markdown_converter::OutputFormat::Xml)
            .map_err(|e| MarkdownLabError::HtmlParse(e.to_string()))
    }

    /// Convert to specified format
    pub fn convert(&self, doc: &Document, format: OutputFormat) -> Result<String, MarkdownLabError> {
        match format {
            OutputFormat::Markdown => self.to_markdown(doc),
            OutputFormat::Json => self.to_json(doc),
            OutputFormat::Xml => self.to_xml(doc),
        }
    }
}

/// Content chunker for RAG applications
pub struct ContentChunker {
    config: Arc<Config>,
}

impl ContentChunker {
    pub fn new(config: Arc<Config>) -> Self {
        Self { config }
    }

    /// Chunk content with the given options
    pub fn chunk_content(&self, content: &str, source_url: &str, options: &ChunkOptions) -> Result<Vec<Chunk>, MarkdownLabError> {
        let chunks = chunker::create_semantic_chunks(content, options.size, options.overlap)
            .map_err(|e| MarkdownLabError::Chunking(e.to_string()))?;

        let mut result = Vec::new();
        let mut start_pos = 0;

        for (index, chunk_content) in chunks.into_iter().enumerate() {
            let end_pos = start_pos + chunk_content.len();
            
            result.push(Chunk {
                content: chunk_content,
                source_url: source_url.to_string(),
                index,
                start_pos,
                end_pos,
                metadata: HashMap::new(),
            });

            // Calculate next start position with overlap
            start_pos = if options.overlap > 0 && end_pos > options.overlap {
                end_pos - options.overlap
            } else {
                end_pos
            };
        }

        Ok(result)
    }

    /// Chunk a document
    pub fn chunk_document(&self, doc: &Document, options: &ChunkOptions) -> Result<Vec<Chunk>, MarkdownLabError> {
        self.chunk_content(&doc.content, &doc.url, options)
    }
}

/// Main Markdown Lab engine - consolidates all functionality
pub struct MarkdownLab {
    http_client: HttpClient,
    html_processor: HtmlProcessor,
    chunker: ContentChunker,
    config: Arc<Config>,
}

impl MarkdownLab {
    /// Create a new Markdown Lab engine
    pub fn new(config: Config) -> Result<Self, MarkdownLabError> {
        let config = Arc::new(config);
        let http_client = HttpClient::new(config.clone())?;
        let html_processor = HtmlProcessor::new(config.clone());
        let chunker = ContentChunker::new(config.clone());

        Ok(Self {
            http_client,
            html_processor,
            chunker,
            config,
        })
    }

    /// Create with default configuration
    pub fn default() -> Result<Self, MarkdownLabError> {
        Self::new(Config::default())
    }

    /// Scrape a single URL and return processed document
    pub async fn scrape_url(&self, url: &str) -> Result<Document, MarkdownLabError> {
        let html = self.http_client.get(url).await?;
        let mut doc = self.html_processor.extract_content(&html, url)?;
        
        // Convert to markdown by default
        doc.content = self.html_processor.to_markdown(&doc)?;
        doc.format = OutputFormat::Markdown;
        
        Ok(doc)
    }

    /// Scrape multiple URLs concurrently
    pub async fn scrape_urls(&self, urls: Vec<&str>) -> Result<Vec<Result<Document, MarkdownLabError>>, MarkdownLabError> {
        let results = self.http_client.get_many(urls).await?;
        
        let mut documents = Vec::new();
        for (url, html_result) in results {
            match html_result {
                Ok(html) => {
                    match self.html_processor.extract_content(&html, &url) {
                        Ok(mut doc) => {
                            // Convert to markdown
                            match self.html_processor.to_markdown(&doc) {
                                Ok(markdown) => {
                                    doc.content = markdown;
                                    doc.format = OutputFormat::Markdown;
                                    documents.push(Ok(doc));
                                }
                                Err(e) => documents.push(Err(e)),
                            }
                        }
                        Err(e) => documents.push(Err(e)),
                    }
                }
                Err(e) => documents.push(Err(e)),
            }
        }
        
        Ok(documents)
    }

    /// Convert HTML content to specified format
    pub fn convert_html(&self, html: &str, base_url: &str, format: OutputFormat) -> Result<String, MarkdownLabError> {
        let doc = self.html_processor.extract_content(html, base_url)?;
        self.html_processor.convert(&doc, format)
    }

    /// Chunk content for RAG applications
    pub fn chunk_content(&self, content: &str, source_url: &str, options: Option<ChunkOptions>) -> Result<Vec<Chunk>, MarkdownLabError> {
        let chunk_options = options.unwrap_or_default();
        self.chunker.chunk_content(content, source_url, &chunk_options)
    }

    /// Process sitemap and scrape discovered URLs
    pub async fn scrape_sitemap(&self, base_url: &str, filters: SitemapFilters) -> Result<Vec<Result<Document, MarkdownLabError>>, MarkdownLabError> {
        // Get sitemap content
        let sitemap_url = format!("{}/sitemap.xml", base_url.trim_end_matches('/'));
        let sitemap_xml = self.http_client.get(&sitemap_url).await?;
        
        // Parse sitemap and extract URLs
        let urls = self.parse_sitemap(&sitemap_xml, &filters)?;
        
        // Scrape discovered URLs
        let url_refs: Vec<&str> = urls.iter().map(|s| s.as_str()).collect();
        self.scrape_urls(url_refs).await
    }

    /// Parse sitemap XML and extract URLs based on filters
    fn parse_sitemap(&self, sitemap_xml: &str, filters: &SitemapFilters) -> Result<Vec<String>, MarkdownLabError> {
        use quick_xml::Reader;
        use quick_xml::events::Event;

        let mut reader = Reader::from_str(sitemap_xml);
        reader.config_mut().check_end_names = false;

        let mut urls = Vec::new();
        let mut current_url = String::new();
        let mut current_priority = None;
        let mut buf = Vec::new();

        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Start(ref e)) => {
                    match e.name().as_ref() {
                        b"loc" => {
                            current_url.clear();
                        }
                        b"priority" => {
                            current_priority = None;
                        }
                        _ => {}
                    }
                }
                Ok(Event::Text(e)) => {
                    let text = e.unescape().unwrap().into_owned();
                    if !current_url.is_empty() || current_priority.is_none() {
                        if current_url.is_empty() {
                            current_url = text;
                        } else if current_priority.is_none() {
                            current_priority = text.parse::<f64>().ok();
                        }
                    }
                }
                Ok(Event::End(ref e)) => {
                    if e.name().as_ref() == b"url" {
                        // Check filters
                        if self.url_passes_filters(&current_url, current_priority, filters) {
                            urls.push(current_url.clone());
                        }
                        current_url.clear();
                        current_priority = None;
                        
                        // Check limit
                        if let Some(limit) = filters.limit {
                            if urls.len() >= limit {
                                break;
                            }
                        }
                    }
                }
                Ok(Event::Eof) => break,
                Err(e) => return Err(MarkdownLabError::HtmlParse(format!("XML parsing error: {}", e))),
                _ => {}
            }
            buf.clear();
        }

        Ok(urls)
    }

    /// Check if URL passes the given filters
    fn url_passes_filters(&self, url: &str, priority: Option<f64>, filters: &SitemapFilters) -> bool {
        // Check priority filter
        if let Some(min_priority) = filters.min_priority {
            if let Some(url_priority) = priority {
                if url_priority < min_priority {
                    return false;
                }
            }
        }

        // Check include patterns
        if !filters.include_patterns.is_empty() {
            let mut matches_include = false;
            for pattern in &filters.include_patterns {
                if let Ok(regex) = regex::Regex::new(pattern) {
                    if regex.is_match(url) {
                        matches_include = true;
                        break;
                    }
                }
            }
            if !matches_include {
                return false;
            }
        }

        // Check exclude patterns
        for pattern in &filters.exclude_patterns {
            if let Ok(regex) = regex::Regex::new(pattern) {
                if regex.is_match(url) {
                    return false;
                }
            }
        }

        true
    }

    /// Get engine configuration
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Update configuration (creates new engine instance)
    pub fn with_config(self, config: Config) -> Result<Self, MarkdownLabError> {
        Self::new(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_markdown_lab_creation() {
        let config = Config::default();
        let lab = MarkdownLab::new(config).expect("Failed to create MarkdownLab");
        
        // Basic test - ensure we can create the engine
        assert_eq!(lab.config().requests_per_second, 1.0);
        assert_eq!(lab.config().chunk_size, 1000);
    }

    #[tokio::test]
    async fn test_config_customization() {
        let mut config = Config::default();
        config.requests_per_second = 2.0;
        config.chunk_size = 500;
        config.cache_enabled = false;
        
        let lab = MarkdownLab::new(config).expect("Failed to create MarkdownLab");
        
        assert_eq!(lab.config().requests_per_second, 2.0);
        assert_eq!(lab.config().chunk_size, 500);
        assert_eq!(lab.config().cache_enabled, false);
    }

    #[tokio::test]
    async fn test_html_conversion() {
        let lab = MarkdownLab::default().expect("Failed to create MarkdownLab");
        
        let html = r#"
            <html>
                <head><title>Test Page</title></head>
                <body>
                    <h1>Main Title</h1>
                    <p>This is a test paragraph with <a href="/link">a link</a>.</p>
                </body>
            </html>
        "#;
        
        let result = lab.convert_html(html, "https://example.com", OutputFormat::Markdown);
        assert!(result.is_ok(), "HTML conversion failed: {:?}", result);
        
        let markdown = result.unwrap();
        assert!(markdown.contains("Main Title"));
        assert!(markdown.contains("test paragraph"));
    }

    #[tokio::test]
    async fn test_chunking() {
        let lab = MarkdownLab::default().expect("Failed to create MarkdownLab");
        
        let content = "# Title\n\nThis is a long paragraph that should be chunked into smaller pieces for better processing. ".repeat(20);
        
        let chunks = lab.chunk_content(&content, "https://example.com", None);
        assert!(chunks.is_ok(), "Chunking failed: {:?}", chunks);
        
        let chunk_list = chunks.unwrap();
        assert!(!chunk_list.is_empty(), "No chunks were created");
        
        // Check that chunks have proper metadata
        for chunk in &chunk_list {
            assert_eq!(chunk.source_url, "https://example.com");
            assert!(!chunk.content.is_empty());
        }
    }

    #[test]
    fn test_output_format_conversion() {
        assert_eq!(OutputFormat::from("markdown"), OutputFormat::Markdown);
        assert_eq!(OutputFormat::from("json"), OutputFormat::Json);
        assert_eq!(OutputFormat::from("xml"), OutputFormat::Xml);
        assert_eq!(OutputFormat::from("invalid"), OutputFormat::Markdown); // Default fallback
    }

    #[test]
    fn test_sitemap_filters() {
        let filters = SitemapFilters {
            min_priority: Some(0.5),
            include_patterns: vec![".*\\.html".to_string()],
            exclude_patterns: vec![".*admin.*".to_string()],
            limit: Some(100),
        };
        
        // Basic test that we can create filters
        assert_eq!(filters.min_priority, Some(0.5));
        assert_eq!(filters.limit, Some(100));
    }
}