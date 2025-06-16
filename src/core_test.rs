#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;

    #[test]
    async fn test_markdown_lab_creation() {
        let config = Config::default();
        let lab = MarkdownLab::new(config).expect("Failed to create MarkdownLab");
        
        // Basic test - ensure we can create the engine
        assert_eq!(lab.config().requests_per_second, 1.0);
        assert_eq!(lab.config().chunk_size, 1000);
    }

    #[test]
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

    #[test]
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

    #[test]
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