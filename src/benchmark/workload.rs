use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct ServingWorkload {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub concurrency: usize,
}

pub fn standard_workload_matrix() -> Vec<ServingWorkload> {
    const PROMPTS: [usize; 6] = [32, 128, 512, 1024, 2048, 8192];
    const CONCURRENCY: [usize; 7] = [1, 2, 4, 8, 16, 32, 64];
    let mut workloads = Vec::with_capacity(PROMPTS.len() * CONCURRENCY.len());
    for prompt_tokens in PROMPTS {
        for concurrency in CONCURRENCY {
            workloads.push(ServingWorkload {
                prompt_tokens,
                completion_tokens: 128,
                concurrency,
            });
        }
    }
    workloads
}
