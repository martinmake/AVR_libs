#include <usart/usart0.h>
#include <adc.h>
#include <servo/timer1/all.h>

using namespace Usart;
using namespace Timer;

void init(void)
{
	sei();

	usart0.init({ TIO_BAUD });
	stdout = usart0.stream();

	Timer1::Spec spec;
	spec.mode                                 = Timer1::MODE::FAST_PWM;
	spec.clock_source                         = Timer1::CLOCK_SOURCE::IO_CLK_OVER_8;
	spec.on_compare_match_output_A_pin_action = Timer1::ON_COMPARE_MATCH_OUTPUT_PIN_ACTION::CLEAR;
	spec.on_compare_match_output_B_pin_action = Timer1::ON_COMPARE_MATCH_OUTPUT_PIN_ACTION::CLEAR;
	spec.output_compare_value_A               = 2 * F_CPU / 8 / 1000;
	spec.output_compare_value_B               = 2 * F_CPU / 8 / 1000;
	timer1.init(spec);

	adc.init({ });
	adc.channel(0);
	adc.sample();

	adc.start_sampling();
}

int main(void)
{
	init();

	while (true)
	{
		printf("%04u\n", adc.value);
		timer1.output_compare_value_A(2 * (F_CPU / 8 / (1000 + adc.value)));
		timer1.output_compare_value_B(2 * (F_CPU / 8 / (1000 + adc.value)));
	}
}
