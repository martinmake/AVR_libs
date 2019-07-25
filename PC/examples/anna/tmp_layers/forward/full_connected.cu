#include "swap_layers.h"

if (idx < LAYER_NEURONS_COUNT)
{
	float output = 0.0;

#ifdef LAYER_IS_FIRST
	d_weights += idx * LAYER_NEURON_WEIGHTS_COUNT;
#endif

	for (uint32_t i = 0; i < LAYER_INPUTS_COUNT; i++)
		output += d_input[i] * d_weights[i];
	output += d_weights[LAYER_INPUTS_COUNT];

#ifdef LAYER_IS_FIRST
	d_input   = d_output;
	d_output += idx;
#endif
	__syncthreads();
	*d_output = LAYER_ACTIVATION_FUNCTION(output);
}

#ifdef LAYER_IS_HIDDEN
	d_weights += ((LAYER_NEURONS_COUNT - idx) * LAYER_NEURON_WEIGHTS_COUNT) + (idx * NEXT_LAYER_NEURON_WEIGHTS_COUNT);
#endif

__syncthreads();

#undef LAYER
