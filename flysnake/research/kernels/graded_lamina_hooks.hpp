// Independent, unfitted graded-release candidate on original L1-L5 cells.
// These globals point to explicit checkpointed arrays, not scene/action labels.
static int lamina_n = 0, lamina_count = 0, lamina_stride = 10;
static bool lamina_enabled = false;
static const int32_t* lamina_indices = nullptr;
static const uint8_t* lamina_mask = nullptr;
static float* lamina_rates = nullptr;
static float* lamina_positive_queue = nullptr;
static float* lamina_negative_queue = nullptr;
static double* lamina_delivered = nullptr;
static float lamina_rest_rate_hz = 50.f, lamina_release_span_mV = 12.f;
static float lamina_maximum_hz = 100.f;

extern "C" EXPORT void lamina_graded_setup(int n, int count, int stride,
    const int32_t* indices, const uint8_t* mask, float* rates,
    float* positive_queue, float* negative_queue, double* delivered,
    float rest_rate, float span_mV, float maximum_hz, int enabled) {
    lamina_n = n; lamina_count = count; lamina_stride = stride;
    lamina_indices = indices; lamina_mask = mask; lamina_rates = rates;
    lamina_positive_queue = positive_queue; lamina_negative_queue = negative_queue;
    lamina_delivered = delivered; lamina_rest_rate_hz = rest_rate;
    lamina_release_span_mV = span_mV; lamina_maximum_hz = maximum_hz;
    lamina_enabled = enabled != 0;
}
