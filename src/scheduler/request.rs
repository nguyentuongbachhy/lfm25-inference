use anyhow::{Result, ensure};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestPhase {
    Free,
    QueuedPrefill,
    Decoding,
}

pub struct SequenceRequest {
    pub request_id: u64,
    pub phase: RequestPhase,
    pub arrival_us: u64,
    pub first_token_deadline_us: u64,
    pub next_token_deadline_us: u64,
    pub prompt_len: usize,
    pub prefilled: usize,
    pub maximum_tokens: usize,
    pub reserved_pages: usize,
    tokens: Vec<u32>,
}

impl SequenceRequest {
    pub fn vacant(maximum_sequence_tokens: usize) -> Self {
        Self {
            request_id: 0,
            phase: RequestPhase::Free,
            arrival_us: 0,
            first_token_deadline_us: 0,
            next_token_deadline_us: 0,
            prompt_len: 0,
            prefilled: 0,
            maximum_tokens: maximum_sequence_tokens,
            reserved_pages: 0,
            tokens: Vec::with_capacity(maximum_sequence_tokens),
        }
    }

    pub fn initialize(
        &mut self,
        request_id: u64,
        prompt: &[u32],
        maximum_tokens: usize,
        now_us: u64,
        ttft_slo_us: u64,
        tpot_slo_us: u64,
        reserved_pages: usize,
    ) -> Result<()> {
        ensure!(self.phase == RequestPhase::Free, "request slot is not free");
        ensure!(!prompt.is_empty(), "request prompt is empty");
        ensure!(
            maximum_tokens >= prompt.len() && maximum_tokens <= self.tokens.capacity(),
            "request exceeds fixed token capacity"
        );
        self.tokens.clear();
        self.tokens.extend_from_slice(prompt);
        self.request_id = request_id;
        self.phase = RequestPhase::QueuedPrefill;
        self.arrival_us = now_us;
        self.first_token_deadline_us = now_us.saturating_add(ttft_slo_us);
        self.next_token_deadline_us = now_us.saturating_add(tpot_slo_us);
        self.prompt_len = prompt.len();
        self.prefilled = 0;
        self.maximum_tokens = maximum_tokens;
        self.reserved_pages = reserved_pages;
        Ok(())
    }

    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    pub fn push_token(&mut self, token: u32, now_us: u64, tpot_slo_us: u64) -> Result<()> {
        ensure!(
            self.tokens.len() < self.maximum_tokens,
            "request token capacity exhausted"
        );
        self.tokens.push(token);
        self.phase = RequestPhase::Decoding;
        self.next_token_deadline_us = now_us.saturating_add(tpot_slo_us);
        Ok(())
    }

    pub fn clear(&mut self) {
        self.tokens.clear();
        self.phase = RequestPhase::Free;
        self.prefilled = 0;
        self.reserved_pages = 0;
    }
}
