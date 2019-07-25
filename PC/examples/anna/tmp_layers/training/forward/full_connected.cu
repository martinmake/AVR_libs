#include "swap_layers.h"

if (idx < LAYER_NEURONS_COUNT)
{
	float output = 0.0;

#ifdef LAYER_IS_FIRST
	d_output             += idx;
	d_weights            += idx * LAYER_NEURON_WEIGHTS_COUNT;
	d_weighted_gradients += idx * LAYER_NEURON_WEIGHTS_COUNT;
#endif

	for (uint32_t i = 0; i < LAYER_INPUTS_COUNT; i++)
		output += d_input[i] * d_weights[i];
	output += d_weights[LAYER_INPUTS_COUNT];

	*d_output = LAYER_ACTIVATION_FUNCTION(output);
}

#ifdef LAYER_IS_HIDDEN
	d_output             += LAYER_OUTPUTS_COUNT;
	d_input              += LAYER_OUTPUTS_COUNT;
	d_weights            += ((LAYER_NEURONS_COUNT - idx) * LAYER_NEURON_WEIGHTS_COUNT) + (idx * NEXT_LAYER_NEURON_WEIGHTS_COUNT);
	d_weighted_gradients += ((LAYER_NEURONS_COUNT - idx) * LAYER_NEURON_WEIGHTS_COUNT) + (idx * NEXT_LAYER_NEURON_WEIGHTS_COUNT);
#endif

__syncthreads();

#undef LAYER
