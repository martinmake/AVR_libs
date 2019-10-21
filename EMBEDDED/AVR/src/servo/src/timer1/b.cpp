#include "servo/timer1/b.h"

namespace Servo
{
	namespace Timer1
	{
		// METHODS
		void B::init(void)
		{
			Servo::Timer1::Base::init();

			Timer::timer1.on_compare_match_output_B_pin_action(Timer::Timer1::ON_COMPARE_MATCH_OUTPUT_PIN_ACTION::CLEAR);
			Timer::timer1.output_compare_value_B(2 * F_CPU / 8 / 1000);
		}
	}
}
