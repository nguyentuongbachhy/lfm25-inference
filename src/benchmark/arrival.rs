use anyhow::{Result, ensure};

#[derive(Debug, Clone, Copy)]
pub enum ArrivalPattern {
    Poisson {
        requests_per_second: f64,
        seed: u64,
    },
    #[cfg(test)]
    Bursty {
        burst_size: usize,
        interval_ms: u64,
    },
}

pub struct ArrivalSchedule {
    next_us: u64,
    #[cfg(test)]
    remaining_in_burst: usize,
    pattern: ArrivalPattern,
    #[cfg(test)]
    started: bool,
    rng_state: u64,
}

impl ArrivalSchedule {
    pub fn new(pattern: ArrivalPattern) -> Result<Self> {
        match pattern {
            ArrivalPattern::Poisson {
                requests_per_second,
                ..
            } => ensure!(
                requests_per_second.is_finite() && requests_per_second > 0.0,
                "arrival rate must be positive"
            ),
            #[cfg(test)]
            ArrivalPattern::Bursty {
                burst_size,
                interval_ms,
            } => ensure!(
                burst_size > 0 && interval_ms > 0,
                "burst parameters must be positive"
            ),
        }
        let rng_state = match pattern {
            ArrivalPattern::Poisson { seed, .. } => seed.max(1),
            #[cfg(test)]
            ArrivalPattern::Bursty { .. } => 1,
        };
        Ok(Self {
            next_us: 0,
            #[cfg(test)]
            remaining_in_burst: 0,
            pattern,
            #[cfg(test)]
            started: false,
            rng_state,
        })
    }

    pub fn next_arrival_us(&mut self) -> u64 {
        match self.pattern {
            ArrivalPattern::Poisson {
                requests_per_second,
                ..
            } => {
                let current = self.next_us;
                self.rng_state ^= self.rng_state << 13;
                self.rng_state ^= self.rng_state >> 7;
                self.rng_state ^= self.rng_state << 17;
                let uniform = ((self.rng_state >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0);
                let interval_us = (-uniform.ln() * 1_000_000.0 / requests_per_second)
                    .round()
                    .max(1.0) as u64;
                self.next_us = self.next_us.saturating_add(interval_us);
                current
            }
            #[cfg(test)]
            ArrivalPattern::Bursty {
                burst_size,
                interval_ms,
            } => {
                if self.remaining_in_burst == 0 {
                    self.remaining_in_burst = burst_size;
                    if self.started {
                        self.next_us = self
                            .next_us
                            .saturating_add(interval_ms.saturating_mul(1000));
                    }
                    self.started = true;
                }
                self.remaining_in_burst -= 1;
                self.next_us
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn burst_arrivals_share_timestamp_then_advance() -> Result<()> {
        let mut arrivals = ArrivalSchedule::new(ArrivalPattern::Bursty {
            burst_size: 2,
            interval_ms: 10,
        })?;
        assert_eq!(
            [
                arrivals.next_arrival_us(),
                arrivals.next_arrival_us(),
                arrivals.next_arrival_us()
            ],
            [0, 0, 10_000]
        );
        Ok(())
    }

    #[test]
    fn poisson_arrivals_are_deterministic_and_strictly_increasing() -> Result<()> {
        let pattern = ArrivalPattern::Poisson {
            requests_per_second: 20.0,
            seed: 7,
        };
        let mut first = ArrivalSchedule::new(pattern)?;
        let mut second = ArrivalSchedule::new(pattern)?;
        let first_times = [
            first.next_arrival_us(),
            first.next_arrival_us(),
            first.next_arrival_us(),
        ];
        let second_times = [
            second.next_arrival_us(),
            second.next_arrival_us(),
            second.next_arrival_us(),
        ];
        assert_eq!(first_times, second_times);
        assert!(first_times[0] < first_times[1] && first_times[1] < first_times[2]);
        Ok(())
    }
}
