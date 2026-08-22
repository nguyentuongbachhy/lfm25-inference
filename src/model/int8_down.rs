const INT8_TINY_M_DOWN_MAX_BATCH: usize = 2;
const INT8_TINY_M_DOWN_LAYERS: [usize; 7] = [0, 1, 2, 3, 4, 5, 7];

fn int8_tiny_m_down_enabled_from_env() -> bool {
    std::env::var("LFM25_INT8_TINY_M_DOWN")
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "on" | "yes"
            )
        })
        .unwrap_or(false)
}

struct Int8DownState {
    weights: Vec<Option<ops::Int8PerChannelWeight>>,
    workspace: ops::Int8TinyMWorkspace,
}

impl Int8DownState {
    fn new(runtime: &CudaRuntime, model: &Lfm2Model, enabled: bool) -> Result<Option<Self>> {
        if !enabled {
            return Ok(None);
        }

        let intermediate = model.config.effective_intermediate_size();
        let mut weights = Vec::with_capacity(model.weights.layers.len());
        for (layer, layer_weights) in model.weights.layers.iter().enumerate() {
            let selected = INT8_TINY_M_DOWN_LAYERS.contains(&layer)
                // Existing selective-FP8 down sites always keep the production
                // fused E4M3 path. The INT8 experiment only replaces BF16 tails.
                && layer_weights.feed_forward.down.fp8.is_none();
            weights.push(if selected {
                Some(ops::quantize_weight_s8_per_channel(
                    runtime,
                    &layer_weights.feed_forward.down.bf16,
                )?)
            } else {
                None
            });
        }

        ensure!(
            weights.iter().filter(|weight| weight.is_some()).count()
                <= INT8_TINY_M_DOWN_LAYERS.len(),
            "INT8 tiny-M down state installed too many weights"
        );
        Ok(Some(Self {
            weights,
            workspace: ops::Int8TinyMWorkspace::new(
                runtime,
                INT8_TINY_M_DOWN_MAX_BATCH,
                intermediate,
            )?,
        }))
    }

    fn try_run(
        &mut self,
        runtime: &CudaRuntime,
        layer: usize,
        m: usize,
        packed_gate_up: &Tensor<bf16>,
        output: &mut Tensor<bf16>,
    ) -> Result<bool> {
        if m == 0 || m > INT8_TINY_M_DOWN_MAX_BATCH {
            return Ok(false);
        }
        let Some(weight) = self.weights.get(layer).and_then(Option::as_ref) else {
            return Ok(false);
        };

        ops::silu_mul_packed_bf16_to_int8_tiny_m_into(
            runtime,
            packed_gate_up,
            &mut self.workspace,
        )?;
        ops::linear_int8_tiny_m_prequantized_into(
            runtime,
            m,
            weight,
            &self.workspace,
            output,
        )?;
        Ok(true)
    }
}
