use rayon::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ChunkerError {
    #[error("Regex error: {0}")]
    RegexError(#[from] regex::Error),

    #[error("Parsing error: {0}")]
    ParsingError(String),

    #[error("Other error: {0}")]
    Other(String),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Chunk {
    pub content: String,
    pub metadata: ChunkMetadata,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChunkMetadata {
    pub heading: Option<String>,
    pub level: usize,
    pub position: usize,
    pub word_count: usize,
    pub char_count: usize,
    pub semantic_density: f32, // A measure of the information density
    pub sentence_count: usize, // Number of sentences in the chunk
    pub avg_sentence_length: f32, // Average sentence length
    pub entity_density: f32, // Density of named entities (uppercase words)
    pub technical_terms: usize, // Count of technical/domain-specific terms
    pub readability_score: f32, // Simple readability measure
    pub topic_keywords: Vec<String>, // Key terms representing the topic
}

/// Creates semantically meaningful chunks from markdown content with improved handling of document structure
pub fn create_semantic_chunks(
    markdown: &str,
    chunk_size: usize,
    chunk_overlap: usize,
) -> Result<Vec<String>, ChunkerError> {
    let heading_regex = Regex::new(r"^(#{1,6})\s+(.+)$")?;
    let chunks = semantic_chunking(markdown, chunk_size, chunk_overlap, &heading_regex)?;

    // Return just the content strings for Python integration
    Ok(chunks.into_iter().map(|chunk| chunk.content).collect())
}

/// Internal function that does the actual semantic chunking
fn semantic_chunking(
    markdown: &str,
    chunk_size: usize,
    chunk_overlap: usize,
    heading_regex: &Regex,
) -> Result<Vec<Chunk>, ChunkerError> {
    let lines: Vec<&str> = markdown.lines().collect();
    let mut chunks: Vec<Chunk> = Vec::with_capacity(lines.len() / 10 + 1);

    let mut current_chunk = String::new();
    let mut current_heading: Option<String> = None;
    let mut current_level = 0;
    let mut current_position = 0;

    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];

        // Check if this is a heading
        if let Some(captures) = heading_regex.captures(line) {
            let heading_level = captures[1].len();
            let heading_text = &captures[2];

            // If we've accumulated content, save it as a chunk before starting a new section
            if !current_chunk.is_empty() {
                chunks.push(create_chunk_object(
                    &current_chunk,
                    current_heading.clone(),
                    current_level,
                    current_position,
                ));
                current_position += 1;
            }

            // Set the new heading info
            current_heading = Some(heading_text.to_string());
            current_level = heading_level;
            current_chunk = line.to_string();
        } else {
            // Add line to current chunk
            if !current_chunk.is_empty() {
                current_chunk.push('\n');
            }
            current_chunk.push_str(line);

            // Check if current chunk is too large
            if current_chunk.len() > chunk_size {
                let split_point = find_good_split_point(&current_chunk, chunk_size - chunk_overlap);

                let (first_part, remaining) = current_chunk.split_at(split_point);

                // Save the first part as a chunk
                chunks.push(create_chunk_object(
                    first_part,
                    current_heading.clone(),
                    current_level,
                    current_position,
                ));
                current_position += 1;

                // Start a new chunk with the overlap
                current_chunk = remaining.trim().to_string();
            }
        }

        i += 1;
    }

    // Add the final chunk
    if !current_chunk.is_empty() {
        chunks.push(create_chunk_object(
            &current_chunk,
            current_heading,
            current_level,
            current_position,
        ));
    }

    // After collecting all chunk strings, parallelize chunk object creation if large
    if chunks.len() > 100 {
        let chunks: Vec<Chunk> = chunks.into_par_iter().map(|chunk| chunk).collect();
        Ok(chunks)
    } else {
        Ok(chunks)
    }
}

/// Helper function to create a chunk object with metadata
/// Uses SIMD-optimized operations and advanced NLP analysis for enhanced metadata
fn create_chunk_object(
    content: &str,
    heading: Option<String>,
    level: usize,
    position: usize,
) -> Chunk {
    // Use SIMD-optimized counting operations
    let words = crate::simd_text::count_words_simd(content);
    let chars = crate::simd_text::count_non_whitespace_chars_simd(content);

    // Calculate semantic density score using optimized operations
    let semantic_density = calculate_semantic_density(content);

    // Perform advanced NLP analysis
    let nlp_analysis = analyze_text_nlp(content);

    Chunk {
        content: content.to_string(),
        metadata: ChunkMetadata {
            heading,
            level,
            position,
            word_count: words,
            char_count: chars,
            semantic_density,
            sentence_count: nlp_analysis.sentence_count,
            avg_sentence_length: nlp_analysis.avg_sentence_length,
            entity_density: nlp_analysis.entity_density,
            technical_terms: nlp_analysis.technical_terms,
            readability_score: nlp_analysis.readability_score,
            topic_keywords: nlp_analysis.topic_keywords,
        },
    }
}

/// Find a good split point that doesn't break in the middle of a sentence or paragraph
fn find_good_split_point(text: &str, approximate_position: usize) -> usize {
    if approximate_position >= text.len() {
        return text.len();
    }

    // Look forward for paragraph break (double newline)
    if let Some(pos) = text[approximate_position..].find("\n\n") {
        return approximate_position + pos + 2; // Include both newlines
    }

    // Look forward for single newline
    if let Some(pos) = text[approximate_position..].find('\n') {
        return approximate_position + pos + 1; // Include the newline
    }

    // Look forward for sentence break
    for (i, c) in text[approximate_position..].char_indices() {
        if c == '.' || c == '!' || c == '?' {
            // Find next non-whitespace or end of string
            let mut end_pos = approximate_position + i + 1;
            // Use char_indices to avoid O(n^2) complexity
            for (idx, c) in text.char_indices().skip(end_pos) {
                if !c.is_whitespace() {
                    end_pos = idx;
                    break;
                }
                // If we reach the end and all are whitespace, set end_pos to text.len()
                end_pos = text.len();
            }
            return end_pos;
        }
    }

    // Fall back to word boundary
    for (i, c) in text[approximate_position..].char_indices() {
        if c.is_whitespace() {
            return approximate_position + i + 1;
        }
    }

    // Last resort
    approximate_position
}

/// Calculate semantic density score using SIMD-optimized operations
/// Enhanced implementation with better performance for large texts
fn calculate_semantic_density(text: &str) -> f32 {
    // Use SIMD-optimized semantic density calculation
    crate::simd_text::calculate_semantic_density_simd(text)
}

/// Advanced NLP analysis for enhanced chunk metadata
struct NlpAnalysis {
    sentence_count: usize,
    avg_sentence_length: f32,
    entity_density: f32,
    technical_terms: usize,
    readability_score: f32,
    topic_keywords: Vec<String>,
}

/// Perform advanced NLP analysis on text content
fn analyze_text_nlp(text: &str) -> NlpAnalysis {
    if text.is_empty() {
        return NlpAnalysis {
            sentence_count: 0,
            avg_sentence_length: 0.0,
            entity_density: 0.0,
            technical_terms: 0,
            readability_score: 0.0,
            topic_keywords: Vec::new(),
        };
    }

    // Count sentences using multiple delimiters
    let sentence_count = count_sentences(text);
    
    // Calculate average sentence length
    let word_count = crate::simd_text::count_words_simd(text);
    let avg_sentence_length = if sentence_count > 0 {
        word_count as f32 / sentence_count as f32
    } else {
        0.0
    };

    // Calculate entity density (uppercase words, potential named entities)
    let entity_density = calculate_entity_density(text);

    // Count technical terms
    let technical_terms = count_technical_terms(text);

    // Calculate readability score (simplified Flesch-like measure)
    let readability_score = calculate_readability_score(text, word_count, sentence_count);

    // Extract topic keywords
    let topic_keywords = extract_topic_keywords(text);

    NlpAnalysis {
        sentence_count,
        avg_sentence_length,
        entity_density,
        technical_terms,
        readability_score,
        topic_keywords,
    }
}

/// Count sentences in text using multiple sentence delimiters
fn count_sentences(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }

    let mut count = 0;
    let mut chars = text.chars().peekable();
    
    while let Some(ch) = chars.next() {
        if matches!(ch, '.' | '!' | '?') {
            // Check if it's not an abbreviation (simplified check)
            if let Some(&next_ch) = chars.peek() {
                if next_ch.is_whitespace() || next_ch == '\n' {
                    count += 1;
                }
            } else {
                // End of text
                count += 1;
            }
        }
    }

    // Ensure at least 1 sentence if there's content
    if count == 0 && !text.trim().is_empty() {
        1
    } else {
        count
    }
}

/// Calculate entity density (proportion of capitalized words)
fn calculate_entity_density(text: &str) -> f32 {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return 0.0;
    }

    let entity_count = words
        .iter()
        .filter(|word| {
            // Count words that start with uppercase and are longer than 1 char
            word.len() > 1 && word.chars().next().unwrap().is_uppercase()
        })
        .count();

    entity_count as f32 / words.len() as f32
}

/// Count technical/domain-specific terms
fn count_technical_terms(text: &str) -> usize {
    let technical_keywords = [
        // Programming terms
        "function", "method", "class", "interface", "algorithm", "implementation",
        "optimization", "performance", "architecture", "framework", "library",
        "database", "query", "index", "schema", "protocol", "API", "REST",
        "JSON", "XML", "HTTP", "HTTPS", "TCP", "UDP", "SQL", "NoSQL",
        
        // Scientific terms
        "analysis", "research", "study", "experiment", "hypothesis", "theory",
        "model", "data", "statistics", "correlation", "regression", "variance",
        "distribution", "probability", "algorithm", "computation", "process",
        
        // Business terms
        "strategy", "optimization", "efficiency", "productivity", "methodology",
        "framework", "implementation", "deployment", "scalability", "integration",
        
        // Technical concepts
        "system", "network", "infrastructure", "security", "encryption",
        "authentication", "authorization", "middleware", "pipeline", "workflow"
    ];

    let text_lower = text.to_lowercase();
    let mut count = 0;

    for keyword in &technical_keywords {
        // Count occurrences of each keyword
        let mut start = 0;
        while let Some(pos) = text_lower[start..].find(keyword) {
            count += 1;
            start += pos + keyword.len();
        }
    }

    count
}

/// Calculate a simplified readability score
/// Based on average sentence length and word complexity
fn calculate_readability_score(text: &str, word_count: usize, sentence_count: usize) -> f32 {
    if sentence_count == 0 || word_count == 0 {
        return 0.0;
    }

    let avg_sentence_length = word_count as f32 / sentence_count as f32;
    
    // Count complex words (longer than 6 characters)
    let complex_words = text
        .split_whitespace()
        .filter(|word| word.len() > 6)
        .count();
    
    let complex_word_ratio = complex_words as f32 / word_count as f32;

    // Simplified readability formula (higher = more readable)
    // Based on shorter sentences and fewer complex words being more readable
    let base_score = 100.0;
    let sentence_penalty = avg_sentence_length * 2.0; // Penalty for long sentences
    let complexity_penalty = complex_word_ratio * 50.0; // Penalty for complex words

    (base_score - sentence_penalty - complexity_penalty).max(0.0).min(100.0)
}

/// Extract key topic words from text
fn extract_topic_keywords(text: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }

    // Common stop words to filter out
    let stop_words = [
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "is", "are", "was", "were", "be", "been", "have",
        "has", "had", "do", "does", "did", "will", "would", "could", "should",
        "may", "might", "can", "this", "that", "these", "those", "i", "you",
        "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"
    ];

    // Extract words and count frequency
    let mut word_counts = std::collections::HashMap::new();
    
    for word in text.split_whitespace() {
        let clean_word = word
            .to_lowercase()
            .trim_matches(|c: char| !c.is_alphanumeric())
            .to_string();
        
        // Filter out stop words and short words
        if clean_word.len() > 3 && !stop_words.contains(&clean_word.as_str()) {
            *word_counts.entry(clean_word).or_insert(0) += 1;
        }
    }

    // Sort by frequency and take top keywords
    let mut words_by_frequency: Vec<(String, usize)> = word_counts.into_iter().collect();
    words_by_frequency.sort_by(|a, b| b.1.cmp(&a.1));

    // Return top 5 keywords
    words_by_frequency
        .into_iter()
        .take(5)
        .map(|(word, _)| word)
        .collect()
}
