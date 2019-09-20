#if defined(__AVR_ATmega328P__)

#include "timer/timer1.h"

Timer1 timer1;

Timer1::Timer1(void)
{
}
Timer1::Timer1(const Init& init_struct)
{
	init(init_struct);
}
void Timer1::init(const Init& init_struct)
{
	switch (init_struct.on_compare_match_output_A)
	{
		case ON_COMPARE_MATCH_OUTPUT::PASS:
			CLEAR(TCCR1A, COM1A0);
			CLEAR(TCCR1A, COM1A1);
			break;
		case ON_COMPARE_MATCH_OUTPUT::TOGGLE:
			SET  (TCCR1A, COM1A0);
			CLEAR(TCCR1A, COM1A1);
			SET(DDRB, PB1);
			break;
		case ON_COMPARE_MATCH_OUTPUT::CLEAR:
			CLEAR(TCCR1A, COM1A0);
			SET  (TCCR1A, COM1A1);
			SET(DDRB, PB1);
			break;
		case ON_COMPARE_MATCH_OUTPUT::SET:
			SET  (TCCR1A, COM1A0);
			SET  (TCCR1A, COM1A1);
			SET(DDRB, PB1);
			break;
	}
	switch (init_struct.on_compare_match_output_B)
	{
		case ON_COMPARE_MATCH_OUTPUT::PASS:
			CLEAR(TCCR1A, COM1B0);
			CLEAR(TCCR1A, COM1B1);
			break;
		case ON_COMPARE_MATCH_OUTPUT::TOGGLE:
			SET  (TCCR1A, COM1B0);
			CLEAR(TCCR1A, COM1B1);
			SET(DDRB, PB2);
			break;
		case ON_COMPARE_MATCH_OUTPUT::CLEAR:
			CLEAR(TCCR1A, COM1B0);
			SET  (TCCR1A, COM1B1);
			SET(DDRB, PB2);
			break;
		case ON_COMPARE_MATCH_OUTPUT::SET:
			SET  (TCCR1A, COM1B0);
			SET  (TCCR1A, COM1B1);
			SET(DDRB, PB2);
			break;
	}

	switch (init_struct.input_capture)
	{
		case INPUT_CAPTURE::ENABLED:
			CLEAR(DDRB, PB0);
			break;
		case INPUT_CAPTURE::DISABLED:
			break;
	}

	switch (init_struct.mode)
	{
		case MODE::NORMAL:
			CLEAR(TCCR1A, WGM10);
			CLEAR(TCCR1A, WGM11);
			CLEAR(TCCR1A, WGM12);
			CLEAR(TCCR1A, WGM13);
			break;
		case MODE::CTC:
			switch (init_struct.input_capture)
			{
				case INPUT_CAPTURE::ENABLED:
					CLEAR(TCCR1A, WGM10);
					CLEAR(TCCR1A, WGM11);
					SET  (TCCR1A, WGM12);
					SET  (TCCR1A, WGM13);
					break;
				case INPUT_CAPTURE::DISABLED:
					CLEAR(TCCR1A, WGM10);
					CLEAR(TCCR1A, WGM11);
					SET  (TCCR1A, WGM12);
					CLEAR(TCCR1A, WGM13);
					break;
			} break;
		case MODE::FAST_PWM:
			switch (init_struct.input_capture)
			{
				case INPUT_CAPTURE::ENABLED:
					CLEAR(TCCR1A, WGM10);
					SET  (TCCR1A, WGM11);
					SET  (TCCR1A, WGM12);
					SET  (TCCR1A, WGM13);
					break;
				case INPUT_CAPTURE::DISABLED:
					SET  (TCCR1A, WGM10);
					SET  (TCCR1A, WGM11);
					SET  (TCCR1A, WGM12);
					SET  (TCCR1A, WGM13);
					break;
			} break;
		case MODE::PHASE_CORRECT_PWM:
			switch (init_struct.input_capture)
			{
				case INPUT_CAPTURE::ENABLED:
					CLEAR(TCCR1A, WGM10);
					SET  (TCCR1A, WGM11);
					CLEAR(TCCR1A, WGM12);
					SET  (TCCR1A, WGM13);
					break;
				case INPUT_CAPTURE::DISABLED:
					SET  (TCCR1A, WGM10);
					SET  (TCCR1A, WGM11);
					CLEAR(TCCR1A, WGM12);
					SET  (TCCR1A, WGM13);
					break;
			} break;
		case MODE::PHASE_AND_FREQUENCY_CORRECT_PWM:
			switch (init_struct.input_capture)
			{
				case INPUT_CAPTURE::ENABLED:
					CLEAR(TCCR1A, WGM10);
					CLEAR(TCCR1A, WGM11);
					CLEAR(TCCR1A, WGM12);
					SET  (TCCR1A, WGM13);
					break;
				case INPUT_CAPTURE::DISABLED:
					SET  (TCCR1A, WGM10);
					CLEAR(TCCR1A, WGM11);
					CLEAR(TCCR1A, WGM12);
					SET  (TCCR1A, WGM13);
					break;
			} break;
	}

	switch (init_struct.clock_source)
	{
		case CLOCK_SOURCE::IO_CLK_OVER_1:
			SET  (TCCR1B, CS10);
			CLEAR(TCCR1B, CS11);
			CLEAR(TCCR1B, CS12);
			break;
		case CLOCK_SOURCE::IO_CLK_OVER_8:
			SET  (TCCR1B, CS10);
			CLEAR(TCCR1B, CS11);
			CLEAR(TCCR1B, CS12);
			break;
		case CLOCK_SOURCE::IO_CLK_OVER_64:
			SET  (TCCR1B, CS10);
			SET  (TCCR1B, CS11);
			CLEAR(TCCR1B, CS12);
			break;
		case CLOCK_SOURCE::IO_CLK_OVER_256:
			CLEAR(TCCR1B, CS10);
			CLEAR(TCCR1B, CS11);
			SET  (TCCR1B, CS12);
			break;
		case CLOCK_SOURCE::IO_CLK_OVER_1024:
			SET  (TCCR1B, CS10);
			CLEAR(TCCR1B, CS11);
			SET  (TCCR1B, CS12);
			break;
		case CLOCK_SOURCE::EXTERNAL_ON_FALLING_EDGE:
			CLEAR(TCCR1B, CS10);
			SET  (TCCR1B, CS11);
			SET  (TCCR1B, CS12);
			break;
		case CLOCK_SOURCE::EXTERNAL_ON_RISING_EDGE:
			SET  (TCCR1B, CS10);
			SET  (TCCR1B, CS11);
			SET  (TCCR1B, CS12);
			break;
	}

	output_compare_register_A(init_struct.output_compare_value_A);
	output_compare_register_B(init_struct.output_compare_value_B);

	on_output_compare_match_A = init_struct.on_output_compare_match_A;
	on_output_compare_match_B = init_struct.on_output_compare_match_B;
	on_overflow               = init_struct.on_overflow;

	if (on_output_compare_match_A) enable_output_compare_match_A_interrupt();
	if (on_output_compare_match_B) enable_output_compare_match_B_interrupt();
	if (on_overflow              ) enable_overflow_interrupt();
}

ISR(TIMER1_OVF_vect) { timer1.on_overflow(); }

ISR(TIMER1_COMPA_vect) { timer1.on_output_compare_match_A(); }
ISR(TIMER1_COMPB_vect) { timer1.on_output_compare_match_B(); }

#endif
