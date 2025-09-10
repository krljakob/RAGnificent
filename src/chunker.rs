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
}

/// creates semantically meaningful chunks from markdown content with improved handling of document structure
pub fn create_semantic_chunks(
    markdown: &str,
    chunk_size: usize,
    chunk_overlap: usize,
) -> Result<Vec<String>, ChunkerError> {
    // validate input parameters
    if chunk_size == 0 {
        return Err(ChunkerError::Other(
            "chunk_size must be greater than 0".to_string(),
        ));
    }

    if chunk_overlap >= chunk_size {
        return Err(ChunkerError::Other(
            "chunk_overlap must be less than chunk_size".to_string(),
        ));
    }

    let heading_regex = Regex::new(r"^(#{1,6})\s+(.+)$")?;
    let chunks = semantic_chunking(markdown, chunk_size, chunk_overlap, &heading_regex)?;

    // return just the content strings for Python integration
    Ok(chunks.into_iter().map(|chunk| chunk.content).collect())
}

/// internal function that does the actual semantic chunking
fn semantic_chunking(
    markdown: &str,
    chunk_size: usize,
    chunk_overlap: usize,
    heading_regex: &Regex,
) -> Result<Vec<Chunk>, ChunkerError> {
    let lines: Vec<&str> = markdown.lines().collect();
    let mut chunks: Vec<Chunk> = Vec::new();

    let mut current_chunk = String::new();
    let mut current_heading: Option<String> = None;
    let mut current_level = 0;
    let mut current_position = 0;

    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];

        // check if this is a heading
        if let Some(captures) = heading_regex.captures(line) {
            let heading_level = captures[1].len();
            let heading_text = &captures[2];

            // if we've accumulated content, save it as a chunk before starting a new section
            if !current_chunk.is_empty() {
                chunks.push(create_chunk_object(
                    &current_chunk,
                    current_heading.clone(),
                    current_level,
                    current_position,
                ));
                current_position += 1;
            }

            // set the new heading info
            current_heading = Some(heading_text.to_string());
            current_level = heading_level;
            current_chunk = line.to_string();
        } else {
            // add line to current chunk
            if !current_chunk.is_empty() {
                current_chunk.push('\n');
            }
            current_chunk.push_str(line);

            // check if current chunk is too large
            if current_chunk.len() > chunk_size {
                let split_point = find_good_split_point(&current_chunk, chunk_size - chunk_overlap);

                let (first_part, remaining) = current_chunk.split_at(split_point);

                // save the first part as a chunk
                chunks.push(create_chunk_object(
                    first_part,
                    current_heading.clone(),
                    current_level,
                    current_position,
                ));
                current_position += 1;

                // start a new chunk with the overlap
                current_chunk = remaining.trim().to_string();
            }
        }

        i += 1;
    }

    // add the final chunk
    if !current_chunk.is_empty() {
        chunks.push(create_chunk_object(
            &current_chunk,
            current_heading,
            current_level,
            current_position,
        ));
    }

    Ok(chunks)
}

/// helper function to create a chunk object with metadata
fn create_chunk_object(
    content: &str,
    heading: Option<String>,
    level: usize,
    position: usize,
) -> Chunk {
    let words = content.split_whitespace().count();
    let chars = content.chars().count();

    // calculate a very basic semantic density score
    // higher score = more semantic meaning relative to length
    let semantic_density = calculate_semantic_density(content);

    Chunk {
        content: content.to_string(),
        metadata: ChunkMetadata {
            heading,
            level,
            position,
            word_count: words,
            char_count: chars,
            semantic_density,
        },
    }
}

/// find a good split point that doesn't break in the middle of a sentence or paragraph
fn find_good_split_point(text: &str, approximate_position: usize) -> usize {
    if approximate_position >= text.len() {
        return text.len();
    }

    // look forward for paragraph break (double newline)
    if let Some(pos) = text[approximate_position..].find("\n\n") {
        return approximate_position + pos + 2; // Include both newlines
    }

    // look forward for single newline
    if let Some(pos) = text[approximate_position..].find('\n') {
        return approximate_position + pos + 1; // Include the newline
    }

    // look forward for sentence break
    for (i, c) in text[approximate_position..].char_indices() {
        if c == '.' || c == '!' || c == '?' {
            // find next non-whitespace or end of string
            let mut end_pos = approximate_position + i + 1;
            while end_pos < text.len()
                && text.chars().nth(end_pos).is_some_and(|c| c.is_whitespace())
            {
                end_pos += 1;
            }
            return end_pos;
        }
    }

    // fall back to word boundary
    for (i, c) in text[approximate_position..].char_indices() {
        if c.is_whitespace() {
            return approximate_position + i + 1;
        }
    }

    // last resort
    approximate_position
}

/// calculate semantic density score
/// this is a simple implementation that can be enhanced later
fn calculate_semantic_density(text: &str) -> f32 {
    let word_count = text.split_whitespace().count() as f32;
    if word_count == 0.0 {
        return 0.0;
    }

    // count semantic indicators like entity names, numbers, special terms
    let mut semantic_indicators = 0.0;

    // check for specialized keywords
    let keywords = [
        "function",
        "class",
        "method",
        "algorithm",
        "process",
        "system",
        "data",
        "model",
        "analysis",
        "implementation",
    ];

    for word in text.split_whitespace() {
        // count words that start with uppercase (potential named entities)
        if word.chars().next().is_some_and(|c| c.is_uppercase()) {
            semantic_indicators += 0.5;
        }

        // count numbers (dates, quantities, etc.)
        if word.chars().any(|c| c.is_numeric()) {
            semantic_indicators += 0.3;
        }

        // count domain keywords
        if keywords.iter().any(|&k| word.to_lowercase().contains(k)) {
            semantic_indicators += 0.7;
        }
    }

    // calculate ratio (scale it between 0.0-1.0)
    let density = (semantic_indicators / word_count).min(1.0);

    // weight longer chunks slightly higher (they're more coherent if they stay together)
    let length_bonus = (word_count / 100.0).min(0.2); // Max 0.2 bonus

    density + length_bonus
}
