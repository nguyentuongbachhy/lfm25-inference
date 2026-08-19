use std::collections::HashMap;

use anyhow::{Result, ensure};
use serde::Serialize;

use super::KvPageSize;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct PrefixCacheSnapshot {
    pub nodes: usize,
    pub cached_pages: usize,
    pub checkpoints: usize,
    pub hits: u64,
    pub misses: u64,
    pub matched_tokens: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefixMatch {
    pub token_len: usize,
    pub checkpoint_slot: u32,
}

struct RadixNode {
    children: HashMap<Box<[u32]>, usize>,
    physical_page: Option<u32>,
    checkpoint_slot: Option<u32>,
    token_len: usize,
    last_access: u64,
}

impl RadixNode {
    fn root() -> Self {
        Self {
            children: HashMap::new(),
            physical_page: None,
            checkpoint_slot: None,
            token_len: 0,
            last_access: 0,
        }
    }

    fn page(physical_page: u32, token_len: usize, last_access: u64) -> Self {
        Self {
            children: HashMap::new(),
            physical_page: Some(physical_page),
            checkpoint_slot: None,
            token_len,
            last_access,
        }
    }
}

/// CPU-side radix index over page-sized token blocks.
///
/// The radix tree never owns GPU tensors itself. Each non-root node owns one
/// immutable physical KV page through `KvPageAllocator::pin_cached_pages`, and
/// selected nodes additionally reference a GPU convolution-state checkpoint.
/// Prefix lookup therefore stays cheap while the existing paged-attention
/// kernels continue consuming ordinary physical page IDs.
pub struct PageRadixCache {
    page_size: usize,
    nodes: Vec<RadixNode>,
    max_cached_pages: usize,
    cached_pages: usize,
    checkpoints: usize,
    clock: u64,
    hits: u64,
    misses: u64,
    matched_tokens: u64,
}

impl PageRadixCache {
    pub fn new(page_size: KvPageSize, max_cached_pages: usize) -> Result<Self> {
        ensure!(
            max_cached_pages > 0,
            "radix cache requires a positive page budget"
        );
        Ok(Self {
            page_size: page_size.value(),
            nodes: vec![RadixNode::root()],
            max_cached_pages,
            cached_pages: 0,
            checkpoints: 0,
            clock: 0,
            hits: 0,
            misses: 0,
            matched_tokens: 0,
        })
    }

    /// Finds the deepest prefix that has both immutable KV pages and a matching
    /// convolution-state checkpoint. `maximum_tokens` should leave at least one
    /// uncached prompt token so the normal forward path still produces logits.
    /// This is the admission lookup and therefore updates hit/miss statistics.
    pub fn longest_checkpoint(
        &mut self,
        tokens: &[u32],
        maximum_tokens: usize,
        physical_pages: &mut Vec<u32>,
    ) -> Option<PrefixMatch> {
        self.lookup_checkpoint(tokens, maximum_tokens, physical_pages, true)
    }

    /// Same lookup used for in-flight coalescing, but it does not alter
    /// admission hit/miss counters. Access timestamps are still updated so a
    /// future eviction policy observes real cache use.
    pub fn probe_checkpoint(
        &mut self,
        tokens: &[u32],
        maximum_tokens: usize,
        physical_pages: &mut Vec<u32>,
    ) -> Option<PrefixMatch> {
        self.lookup_checkpoint(tokens, maximum_tokens, physical_pages, false)
    }

    fn lookup_checkpoint(
        &mut self,
        tokens: &[u32],
        maximum_tokens: usize,
        physical_pages: &mut Vec<u32>,
        record_stats: bool,
    ) -> Option<PrefixMatch> {
        physical_pages.clear();
        let usable_tokens = tokens.len().min(maximum_tokens);
        let page_count = usable_tokens / self.page_size;
        if page_count == 0 {
            if record_stats {
                self.misses = self.misses.saturating_add(1);
            }
            return None;
        }

        self.clock = self.clock.saturating_add(1);
        let access = self.clock;
        let mut node_id = 0usize;
        let mut best = None;
        let mut best_pages = 0usize;

        for page_index in 0..page_count {
            let start = page_index * self.page_size;
            let end = start + self.page_size;
            let block = &tokens[start..end];
            let Some(&child) = self.nodes[node_id].children.get(block) else {
                break;
            };
            node_id = child;
            let node = &mut self.nodes[node_id];
            node.last_access = access;
            physical_pages.push(
                node.physical_page
                    .expect("non-root radix node must own a physical page"),
            );
            if let Some(checkpoint_slot) = node.checkpoint_slot {
                best = Some(PrefixMatch {
                    token_len: node.token_len,
                    checkpoint_slot,
                });
                best_pages = physical_pages.len();
            }
        }

        if let Some(hit) = best {
            physical_pages.truncate(best_pages);
            if record_stats {
                self.hits = self.hits.saturating_add(1);
                self.matched_tokens = self
                    .matched_tokens
                    .saturating_add(u64::try_from(hit.token_len).unwrap_or(u64::MAX));
            }
            Some(hit)
        } else {
            physical_pages.clear();
            if record_stats {
                self.misses = self.misses.saturating_add(1);
            }
            None
        }
    }

    /// Returns whether a page-aligned prefix can be published without exceeding
    /// the fixed radix page budget. Existing path nodes do not consume budget a
    /// second time.
    pub fn can_publish(&self, tokens: &[u32], prefix_tokens: usize) -> bool {
        if prefix_tokens == 0
            || prefix_tokens > tokens.len()
            || !prefix_tokens.is_multiple_of(self.page_size)
        {
            return false;
        }
        let page_count = prefix_tokens / self.page_size;
        let mut node_id = 0usize;
        let mut existing_pages = 0usize;
        for page_index in 0..page_count {
            let start = page_index * self.page_size;
            let end = start + self.page_size;
            let block = &tokens[start..end];
            let Some(&child) = self.nodes[node_id].children.get(block) else {
                break;
            };
            node_id = child;
            existing_pages += 1;
        }
        if existing_pages == page_count && self.nodes[node_id].checkpoint_slot.is_some() {
            return false;
        }
        let new_pages = page_count.saturating_sub(existing_pages);
        self.cached_pages.saturating_add(new_pages) <= self.max_cached_pages
    }

    /// Inserts a checkpoint and returns only physical pages that became newly
    /// owned by radix nodes. The caller pins those pages in the allocator.
    pub fn insert_checkpoint(
        &mut self,
        tokens: &[u32],
        prefix_tokens: usize,
        physical_pages: &[u32],
        checkpoint_slot: u32,
    ) -> Result<Vec<u32>> {
        ensure!(prefix_tokens > 0, "cached prefix cannot be empty");
        ensure!(
            prefix_tokens <= tokens.len(),
            "cached prefix exceeds token input"
        );
        ensure!(
            prefix_tokens.is_multiple_of(self.page_size),
            "cached prefix must be page aligned"
        );
        let page_count = prefix_tokens / self.page_size;
        ensure!(
            physical_pages.len() == page_count,
            "cached prefix physical page count mismatch"
        );
        ensure!(
            self.can_publish(tokens, prefix_tokens),
            "radix cache prefix cannot be published"
        );

        self.clock = self.clock.saturating_add(1);
        let access = self.clock;
        let mut node_id = 0usize;
        let mut newly_cached_pages = Vec::new();

        for (page_index, &physical_page) in physical_pages.iter().enumerate() {
            let start = page_index * self.page_size;
            let end = start + self.page_size;
            let block = &tokens[start..end];
            if let Some(&child) = self.nodes[node_id].children.get(block) {
                node_id = child;
                self.nodes[node_id].last_access = access;
                continue;
            }

            let child = self.nodes.len();
            self.nodes.push(RadixNode::page(physical_page, end, access));
            self.nodes[node_id]
                .children
                .insert(block.to_vec().into_boxed_slice(), child);
            node_id = child;
            newly_cached_pages.push(physical_page);
        }

        ensure!(
            self.nodes[node_id].checkpoint_slot.is_none(),
            "radix checkpoint already exists"
        );
        self.nodes[node_id].checkpoint_slot = Some(checkpoint_slot);
        self.nodes[node_id].last_access = access;
        self.cached_pages = self
            .cached_pages
            .checked_add(newly_cached_pages.len())
            .ok_or_else(|| anyhow::anyhow!("radix cached page count overflow"))?;
        self.checkpoints = self.checkpoints.saturating_add(1);
        Ok(newly_cached_pages)
    }

    pub fn snapshot(&self) -> PrefixCacheSnapshot {
        PrefixCacheSnapshot {
            nodes: self.nodes.len(),
            cached_pages: self.cached_pages,
            checkpoints: self.checkpoints,
            hits: self.hits,
            misses: self.misses,
            matched_tokens: self.matched_tokens,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(pages: usize) -> Vec<u32> {
        (0..pages * 16).map(|value| value as u32).collect()
    }

    #[test]
    fn longest_match_stops_at_deepest_checkpoint() -> Result<()> {
        let mut cache = PageRadixCache::new(KvPageSize::P16, 16)?;
        let input = tokens(4);
        assert!(cache.can_publish(&input, 32));
        assert_eq!(cache.insert_checkpoint(&input, 32, &[4, 5], 7)?, vec![4, 5]);
        assert!(cache.can_publish(&input, 64));
        assert_eq!(
            cache.insert_checkpoint(&input, 64, &[99, 98, 6, 7], 8)?,
            vec![6, 7]
        );

        let mut pages = Vec::new();
        let hit = cache
            .longest_checkpoint(&input, 63, &mut pages)
            .expect("prefix hit");
        assert_eq!(hit.token_len, 32);
        assert_eq!(hit.checkpoint_slot, 7);
        assert_eq!(pages, [4, 5]);

        let hit = cache
            .longest_checkpoint(&input, 64, &mut pages)
            .expect("long prefix hit");
        assert_eq!(hit.token_len, 64);
        assert_eq!(hit.checkpoint_slot, 8);
        assert_eq!(pages, [4, 5, 6, 7]);
        Ok(())
    }

    #[test]
    fn refresh_probe_does_not_change_admission_statistics() -> Result<()> {
        let mut cache = PageRadixCache::new(KvPageSize::P16, 16)?;
        let input = tokens(2);
        cache.insert_checkpoint(&input, 32, &[1, 2], 3)?;
        let mut pages = Vec::new();
        assert!(cache.probe_checkpoint(&input, 32, &mut pages).is_some());
        assert_eq!(cache.snapshot().hits, 0);
        assert_eq!(cache.snapshot().misses, 0);
        assert_eq!(cache.snapshot().matched_tokens, 0);
        Ok(())
    }

    #[test]
    fn page_budget_counts_only_new_radix_nodes() -> Result<()> {
        let mut cache = PageRadixCache::new(KvPageSize::P16, 3)?;
        let input = tokens(4);
        cache.insert_checkpoint(&input, 32, &[1, 2], 1)?;
        assert!(cache.can_publish(&input, 48));
        cache.insert_checkpoint(&input, 48, &[9, 9, 3], 2)?;
        assert!(!cache.can_publish(&input, 64));
        assert_eq!(cache.snapshot().cached_pages, 3);
        Ok(())
    }
}
