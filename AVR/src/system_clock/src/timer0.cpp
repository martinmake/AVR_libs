#include <timer/timer0.h>

#include "system_clock/timer0.h"

namespace SystemClock
{
	Timer0::Timer0(void)
	{
	}
	Timer0::~Timer0(void)
	{
	}

	void Timer0::init(void)
	{
		Timer::Timer0::Init timer_timer0_init;
		timer_timer0_init.mode         = Timer::Timer0::MODE::NORMAL;
		timer_timer0_init.clock_source = Timer::Timer0::CLOCK_SOURCE::IO_CLK_OVER_1;
		timer_timer0_init.on_overflow  = [this]() { m_time++; };
		Timer::timer0.init(timer_timer0_init);
	}
}
