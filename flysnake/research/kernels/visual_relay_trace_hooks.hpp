// Observation only. These buffers never supply input to the neural equations.
// The derivative kernel retains every original arithmetic/state operation.
static int visual_trace_n = 0;
static const uint8_t* visual_trace_kind = nullptr;
static double* visual_trace_events = nullptr;
static float* visual_trace_min = nullptr;
static float* visual_trace_max = nullptr;

extern "C" EXPORT void visual_trace_setup(int n, const uint8_t* kind,
    double* events, float* minimum, float* maximum) {
    visual_trace_n = n;
    visual_trace_kind = kind;
    visual_trace_events = events;
    visual_trace_min = minimum;
    visual_trace_max = maximum;
}

static inline void visual_trace_receive(int source, int target, float weight,
    bool accepted) {
    if (!visual_trace_events || target >= visual_trace_n) return;
    // Negative columns contain positive magnitudes, in the original g units.
    // 0..3: accepted positive/negative, refractory-rejected positive/negative.
    // 4..7: accepted R1-R6 positive/negative, accepted R8 positive/negative.
    // 8..9: number of accepted/rejected actual edge delivery events.
    const bool positive = weight >= 0.f;
    const double amount = positive ? weight : -double(weight);
    const int column = accepted ? (positive ? 0 : 1) : (positive ? 2 : 3);
    visual_trace_events[column * visual_trace_n + target] += amount;
    visual_trace_events[(accepted ? 8 : 9) * visual_trace_n + target] += 1.;
    if (accepted && visual_trace_kind[source]) {
        const int sensory_column = visual_trace_kind[source] == 1
            ? (positive ? 4 : 5) : (positive ? 6 : 7);
        visual_trace_events[sensory_column * visual_trace_n + target] += amount;
    }
}

static inline void visual_trace_voltage(int cell, float voltage) {
    if (!visual_trace_min || cell >= visual_trace_n) return;
    visual_trace_min[cell] = std::min(visual_trace_min[cell], voltage);
    visual_trace_max[cell] = std::max(visual_trace_max[cell], voltage);
}
