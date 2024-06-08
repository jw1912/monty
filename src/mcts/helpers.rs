use crate::{params::MctsParams, tree::Edge};

pub struct SearchHelpers;

impl SearchHelpers {
    pub fn get_cpuct(params: &MctsParams, parent: &Edge) -> f32 {
        // baseline CPUCT value
        let mut cpuct = params.cpuct();

        // scale CPUCT as visits increase
        cpuct *= 1.0 + (((parent.visits() + 8192) / 8192) as f32).ln();

        // scale CPUCT with variance of Q
        if parent.visits() > 1 {
            let frac = parent.var().sqrt() / params.cpuct_var_scale();
            cpuct *= 1.0 + params.cpuct_var_weight() * (frac - 1.0);
        }

        cpuct
    }

    pub fn get_fpu(parent: &Edge) -> f32 {
        1.0 - parent.q()
    }
}