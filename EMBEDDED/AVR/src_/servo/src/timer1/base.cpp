#include "servo/timer1/base.h"

namespace Servo
{
	namespace Timer1
	{
		void Base::init(void)
		{
			static bool is_initialized = false;

			if (!is_initialized)
			{
				Timer::Timer1::Spec spec;
				spec.mode         = Timer::Timer1::MODE::FAST_PWM;
				spec.clock_source = Timer::Timer1::CLOCK_SOURCE::IO_CLK_OVER_8;
				Timer::timer1.init(spec);

				is_initialized = true;
			}
		}
	}
}
