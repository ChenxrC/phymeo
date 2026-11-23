// Optional bundle semantic: output display-linear values only for the 4x pipeline.
// The original 1x pipeline leaves SpecId60 at zero, retaining the existing OETF.
layout(constant_id=59) const uint PR_MSAA_COLOR_ABI=1u;
layout(constant_id=60) const uint PR_MSAA_LINEAR_OUTPUT=0u;
