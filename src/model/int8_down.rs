const INT8_TINY_M_DOWN_MAX_BATCH: usize = 2;
const INT8_TINY_M_DOWN_LAYERS: [usize; 7] = [0, 1, 2, 3, 4, 5, 7];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Int8DownMode {
    W8A8,
    W8A16,
}

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
    workspace: Option<ops::Int8TinyMWorkspace>,
    w8a16_activation: Option<Tensor<bf16>>,
    mode: Int8DownMode,
}

impl Int8DownState {
    fn new(runtime: &CudaRuntime, model: &Lfm2Model, enabled: bool) -> Result<Option<Self>> {
        let layers: &[usize] = if enabled {
            &INT8_TINY_M_DOWN_LAYERS
        } else {
            &[]
        };
        Self::new_with_layers(runtime, model, layers, Int8DownMode::W8A8)
    }

    fn new_with_layers(
        runtime: &CudaRuntime,
        model: &Lfm2Model,
        selected_layers: &[usize],
        mode: Int8DownMode,
    ) -> Result<Option<Self>> {
        if selected_layers.is_empty() {
            return Ok(None);
        }
        ensure!(
            selected_layers
                .iter()
                .all(|&layer| layer < model.weights.layers.len()),
            "INT8 tiny-M down layer mask contains an out-of-range layer"
        );
        let mut ordered = selected_layers.to_vec();
        ordered.sort_unstable();
        ordered.dedup();
        ensure!(
            ordered.len() == selected_layers.len(),
            "INT8 tiny-M down layer mask contains duplicates"
        );

        let intermediate = model.config.effective_intermediate_size();
        let mut weights = Vec::with_capacity(model.weights.layers.len());
        for (layer, layer_weights) in model.weights.layers.iter().enumerate() {
            let selected = selected_layers.contains(&layer)
                // Existing selective-FP8 down sites always keep the production
                // fused E4M3 path. INT8 experiments only replace BF16 tails.
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
            weights.iter().filter(|weight| weight.is_some()).count() <= selected_layers.len(),
            "INT8 tiny-M down state installed too many weights"
        );
        let workspace = match mode {
            Int8DownMode::W8A8 => Some(ops::Int8TinyMWorkspace::new(
                runtime,
                INT8_TINY_M_DOWN_MAX_BATCH,
                intermediate,
            )?),
            Int8DownMode::W8A16 => None,
        };
        let w8a16_activation = match mode {
            Int8DownMode::W8A8 => None,
            Int8DownMode::W8A16 => Some(runtime.alloc_uninit::<bf16>(Shape::new([
                INT8_TINY_M_DOWN_MAX_BATCH,
                intermediate,
            ]))?),
        };
        Ok(Some(Self {
            weights,
            workspace,
            w8a16_activation,
            mode,
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

        match self.mode {
            Int8DownMode::W8A8 => {
                let workspace = self
                    .workspace
                    .as_mut()
                    .context("W8A8 down state is missing its quantization workspace")?;
                ops::silu_mul_packed_bf16_to_int8_tiny_m_into(
                    runtime,
                    packed_gate_up,
                    workspace,
                )?;
                ops::linear_int8_tiny_m_prequantized_into(
                    runtime,
                    m,
                    weight,
                    workspace,
                    output,
                )?;
            }
            Int8DownMode::W8A16 => {
                let activated = self
                    .w8a16_activation
                    .as_mut()
                    .context("W8A16 down state is missing its BF16 activation scratch")?;
                ops::silu_mul_packed_bf16_into(runtime, packed_gate_up, activated)?;
                ops::linear_w8a16_tiny_m_into(runtime, activated, weight, output)?;
            }
        }
        Ok(true)
    }
}

#[cfg(test)]
impl DecodeExecutor {
    pub(crate) fn set_int8_tiny_m_down_layers(
        &mut self,
        runtime: &CudaRuntime,
        model: &Lfm2Model,
        selected_layers: &[usize],
    ) -> Result<()> {
        self.int8_down = Int8DownState::new_with_layers(
            runtime,
            model,
            selected_layers,
            Int8DownMode::W8A8,
        )?;
        Ok(())
    }

    pub(crate) fn set_w8a16_tiny_m_down_layers(
        &mut self,
        runtime: &CudaRuntime,
        model: &Lfm2Model,
        selected_layers: &[usize],
    ) -> Result<()> {
        self.int8_down = Int8DownState::new_with_layers(
            runtime,
            model,
            selected_layers,
            Int8DownMode::W8A16,
        )?;
        Ok(())
    }

    pub(crate) fn int8_test_logits(&self) -> &Tensor<bf16> {
        &self.logits
    }
}
