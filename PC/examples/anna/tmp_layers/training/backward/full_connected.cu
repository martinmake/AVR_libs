#include "swap_layers.h"

#ifdef LAYER_IS_HIDDEN
d_output -= LAYER_OUTPUTS_COUNT;
d_input  -= LAYER_INPUTS_COUNT;
#endif

#if defined(LAYER_IS_OUTPUT) && LAYER_COUNT > 1
d_weights            -= idx * LAYER_NEURON_WEIGHTS_COUNT;
d_weighted_gradients -= idx * LAYER_NEURON_WEIGHTS_COUNT;
d_weights            += idx;
d_weighted_gradients += idx;
#elif LAYER_IS_HIDDEN
d_weights            -= LAYER_NEURON_WEIGHTS_COUNT * LAYER_INPUTS_COUNT;
d_weighted_gradients -= LAYER_NEURON_WEIGHTS_COUNT * LAYER_INPUTS_COUNT;
#endif

if (idx < LAYER_OUTPUTS_COUNT)
{
#ifdef LAYER_IS_OUTPUT
	*s_gradient = LAYER_ACTIVATION_FUNCTION_DERIVATIVE(*d_output) * (d_desired_output[idx] - *d_output);
#else
	float error_back = 0;
	for (uint16_t i = 0; i < NEXT_LAYER_NEURONS_COUNT)
		error_back += s_gradients[i] * d_weights[i * NEXT_LAYER_WEIGHTS_COUNT];
	__syncthreads();
	*s_gradient = LAYER_ACTIVATION_FUNCTION_DERIVATIVE(*d_output) * error_back;
#endif

	for (uint16_t i = 0; i < LAYER_INPUTS_COUNT; i++)
		d_weighted_gradients[i] += *s_gradient * d_input[i] * LEARNING_RATE;
	d_weighted_gradients[LAYER_INPUTS_COUNT] += *s_gradient * LEARNING_RATE;
}

__syncthreads();

#undef LAYER
