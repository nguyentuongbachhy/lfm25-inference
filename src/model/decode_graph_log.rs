use super::lfm2::DecodeExecutor;

impl Drop for DecodeExecutor {
    fn drop(&mut self) {
        let stats = self.graph_stats();
        eprintln!(
            "decode graphs: entries={} captures={} replays={} capture_failures={} direct_steps={}",
            stats.entries,
            stats.captures,
            stats.replays,
            stats.capture_failures,
            stats.direct_steps,
        );
    }
}
