use std::cmp::min;

/// Dynamic N-Gram / Prompt-Lookup Drafter for Training-Free Speculative Decoding.
/// Searches preceding context (prompt + generated tokens) for recurring n-gram matches
/// and proposes candidate continuation tokens with zero GPU compute and zero VRAM.
#[derive(Debug, Clone)]
pub struct NgramDrafter {
    min_ngram: usize,
    max_ngram: usize,
    draft_length: usize,
}

impl Default for NgramDrafter {
    fn default() -> Self {
        Self {
            min_ngram: 1,
            max_ngram: 4,
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
    /// Scans backwards for the most recent occurrence of the longest matching suffix.
    pub fn draft(&self, history: &[u32]) -> Vec<u32> {
        if self.draft_length == 0 || history.len() <= self.min_ngram {
            return Vec::new();
        }

        let max_n = min(self.max_ngram, history.len() - 1);
        for n in (self.min_ngram..=max_n).rev() {
            let suffix = &history[history.len() - n..];
            let search_limit = history.len() - n;

            // Search backwards for the most recent match
            for i in (0..search_limit).rev() {
                if &history[i..i + n] == suffix {
                    let draft_start = i + n;
                    let draft_end = min(draft_start + self.draft_length, history.len());
                    if draft_start < draft_end {
                        return history[draft_start..draft_end].to_vec();
                    }
                }
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
        // History contains pattern [10, 20, 30] followed by [40, 50, 60]
        let history = vec![1, 2, 10, 20, 30, 40, 50, 60, 99, 10, 20, 30];
        let candidates = drafter.draft(&history);
        assert_eq!(candidates, vec![40, 50, 60]);
    }

    #[test]
    fn test_draft_falls_back_to_shorter_ngram() {
        let drafter = NgramDrafter::new(2, 4, 2);
        // 3-gram [20, 30, 99] not found, but 2-gram [30, 99] is found
        let history = vec![1, 2, 30, 99, 40, 50, 10, 20, 30, 99];
        let candidates = drafter.draft(&history);
        assert_eq!(candidates, vec![40, 50]);
    }

    #[test]
    fn test_draft_empty_when_no_match() {
        let drafter = NgramDrafter::new(2, 3, 3);
        let history = vec![1, 2, 3, 4, 5, 6, 7];
        let candidates = drafter.draft(&history);
        assert!(candidates.is_empty());
    }

    #[test]
    fn test_draft_empty_when_history_too_short() {
        let drafter = NgramDrafter::new(3, 4, 3);
        let history = vec![1, 2];
        let candidates = drafter.draft(&history);
        assert!(candidates.is_empty());
    }

    #[test]
    fn test_draft_picks_most_recent_occurrence() {
        let drafter = NgramDrafter::new(2, 2, 2);
        // [10, 20] appears twice: once followed by [30, 40], later followed by [50, 60]
        let history = vec![10, 20, 30, 40, 1, 2, 10, 20, 50, 60, 3, 4, 10, 20];
        let candidates = drafter.draft(&history);
        assert_eq!(candidates, vec![50, 60]);
    }
}
