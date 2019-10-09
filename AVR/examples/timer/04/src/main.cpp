#include <avr/io.h>

#include <util.h>
#include <led.h>
#include <adc.h>
#include <timer/timer0.h>
#include <usart/usart0.h>

using namespace Timer;

Led<PORT::D, 6, POLARITY::INVERTED> led;

void init(void)
{
	Timer0::Init timer0_init;
	timer0_init.mode                                 = Timer0::MODE::FAST_PWM;
	timer0_init.clock_source                         = Timer0::CLOCK_SOURCE::IO_CLK_OVER_1;
	timer0_init.on_compare_match_output_A_pin_action = Timer0::ON_COMPARE_MATCH_OUTPUT_PIN_ACTION::SET;
	timer0.init(timer0_init);

	adc.init({ });
	adc.channel(5);
	adc.on_conversion = []()
	{
		uint8_t value = adc.value / 4;

		if (value)
			timer0.on_compare_match_output_A_pin_action(Timer0::ON_COMPARE_MATCH_OUTPUT_PIN_ACTION::SET);
		else
		{
			timer0.on_compare_match_output_A_pin_action(Timer0::ON_COMPARE_MATCH_OUTPUT_PIN_ACTION::PASS);
			led.turn_off();
		}

		timer0.output_compare_register_A(value);
	};

	usart0.init({ TIO_BAUD });
	stdout = usart0.stream();

	sei();

	adc.start_sampling();
}

int main(void)
{
	init();

	while (1) printf("%u\n", adc.value);
}
