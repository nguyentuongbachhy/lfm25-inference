use std::cmp::min;
use std::collections::HashMap;

/// Dynamic N-Gram / Prompt-Lookup Drafter for Training-Free Speculative Decoding.
/// Searches preceding context (prompt + generated tokens) for recurring n-gram matches
/// and proposes candidate continuation tokens with zero GPU compute and zero VRAM.
///
/// Features:
/// - Language-Agnostic: works purely on token IDs (Vietnamese, English, French, German, Spanish, Italian, Code).
/// - Dynamic Depth Scaling: scales proposed draft length based on matched prefix depth (2-gram -> 2, 3-gram -> 3, 4-gram+ -> full).
/// - Frequency-Weighted Selection: when an n-gram appears multiple times, selects the continuation with the highest recurrence frequency.
#[derive(Debug, Clone)]
pub struct NgramDrafter {
    min_ngram: usize,
    max_ngram: usize,
    draft_length: usize,
}

impl Default for NgramDrafter {
    fn default() -> Self {
        Self {
            min_ngram: 3,
            max_ngram: 5,
            draft_length: 3,
        }
    }
}

impl NgramDrafter {
    pub fn new(min_ngram: usize, max_ngram: usize, draft_length: usize) -> Self {
        Self {
            min_ngram: min_ngram.max(1),
            max_ngram: max_ngram.max(min_ngram),
            draft_length,
        }
    }

    /// Proposes up to `draft_length` candidate tokens given the complete token history.
    /// Scans backwards for matching suffixes, weighting by frequency and scaling by depth.
    pub fn draft(&self, history: &[u32]) -> Vec<u32> {
        if self.draft_length == 0 || history.len() <= self.min_ngram {
            return Vec::new();
        }

        let max_n = min(self.max_ngram, history.len() - 1);
        for n in (self.min_ngram..=max_n).rev() {
            let suffix = &history[history.len() - n..];
            let search_limit = history.len() - n;

            // Maximum tokens allowed based on matching depth:
            // 2-gram match: max 2 tokens
            // 3-gram match: max 3 tokens
            // 4-gram+ match: full draft_length
            let max_tokens_for_depth = match n {
                2 => min(self.draft_length, 2),
                3 => min(self.draft_length, 3),
                _ => self.draft_length,
            };

            // Collect all occurrences of this n-gram prefix
            // Key: continuation slice, Value: (frequency, most_recent_index)
            let mut candidates: HashMap<&[u32], (usize, usize)> = HashMap::new();

            for i in (0..search_limit).rev() {
                if &history[i..i + n] == suffix {
                    let draft_start = i + n;
                    let draft_end = min(draft_start + max_tokens_for_depth, history.len());
                    if draft_start < draft_end {
                        let candidate_slice = &history[draft_start..draft_end];
                        let entry = candidates.entry(candidate_slice).or_insert((0, i));
                        entry.0 += 1;
                        if i > entry.1 {
                            entry.1 = i;
                        }
                    }
                }
            }

            // Pick candidate with highest frequency, tie-broken by recency
            if let Some((best_slice, _)) = candidates
                .into_iter()
                .max_by(|a, b| a.1.0.cmp(&b.1.0).then_with(|| a.1.1.cmp(&b.1.1)))
            {
                return best_slice.to_vec();
            }
        }

        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_draft_finds_exact_ngram_continuation() {
        let drafter = NgramDrafter::new(2, 3, 3);
        let history = vec![1, 2, 10, 20, 30, 40, 50, 60, 99, 10, 20, 30];
        let candidates = drafter.draft(&history);
        assert_eq!(candidates, vec![40, 50, 60]);
    }

    #[test]
    fn test_draft_falls_back_to_shorter_ngram() {
        let drafter = NgramDrafter::new(2, 4, 3);
        // 3-gram [20, 30, 99] not found, but 2-gram [30, 99] is found
        // 2-gram match scales to max 2 tokens
        let history = vec![1, 2, 30, 99, 40, 50, 10, 20, 30, 99];
        let candidates = drafter.draft(&history);
        assert_eq!(candidates, vec![40, 50]);
    }

    #[test]
    fn test_draft_depth_scaling_limits_2gram_to_2_tokens() {
        let drafter = NgramDrafter::new(2, 4, 5); // draft_length = 5
        let history = vec![10, 20, 30, 40, 50, 60, 70, 99, 10, 20];
        let candidates = drafter.draft(&history);
        // 2-gram match [10, 20] scales to at most 2 tokens
        assert_eq!(candidates, vec![30, 40]);
    }

    #[test]
    fn test_draft_depth_scaling_allows_5_tokens_for_4gram() {
        let drafter = NgramDrafter::new(2, 5, 5);
        let history = vec![10, 20, 30, 40, 50, 60, 70, 80, 99, 10, 20, 30, 40];
        let candidates = drafter.draft(&history);
        // 4-gram match [10, 20, 30, 40] allows full 5 tokens
        assert_eq!(candidates, vec![50, 60, 70, 80, 99]);
    }

    #[test]
    fn test_draft_prefers_higher_frequency_continuation() {
        let drafter = NgramDrafter::new(2, 3, 2);
        // [10, 20] is followed by [30, 40] TWICE, and by [90, 91] ONCE (more recently)
        let history = vec![
            10, 20, 30, 40, // occurrence 1
            1, 2, 10, 20, 30, 40, // occurrence 2
            3, 4, 10, 20, 90, 91, // occurrence 3 (more recent, but lower frequency)
            5, 6, 10, 20, // query suffix
        ];
        let candidates = drafter.draft(&history);
        // Frequency 2 wins over recency 1
        assert_eq!(candidates, vec![30, 40]);
    }

    #[test]
    fn test_draft_multilingual_vietnamese_subwords() {
        let drafter = NgramDrafter::new(2, 4, 3);
        // Token IDs representing:
        // [Trí=1001, tuệ=1002, nhân=1003, tạo=1004, Việt=2001, Nam=2002]
        let history = vec![
            1001, 1002, 1003, 1004, // "Trí tuệ nhân tạo"
            50, 51, 2001, 2002, // "Việt Nam"
            60, 61, 1001, 1002, // Query suffix "Trí tuệ"
        ];
        let candidates = drafter.draft(&history);
        // Should predict continuation ["nhân", "tạo"]
        assert_eq!(candidates, vec![1003, 1004]);
    }

    #[test]
    fn test_draft_empty_when_no_match() {
        let drafter = NgramDrafter::new(2, 3, 3);
        let history = vec![1, 2, 3, 4, 5, 6, 7];
        let candidates = drafter.draft(&history);
        assert!(candidates.is_empty());
    }

    #[test]
    fn test_draft_empty_when_below_min_ngram() {
        let drafter = NgramDrafter::new(2, 4, 3);
        let history = vec![10]; // only 1 token, min_ngram is 2
        let candidates = drafter.draft(&history);
        assert!(candidates.is_empty());
    }
}
