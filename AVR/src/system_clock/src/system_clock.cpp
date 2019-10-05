#include "system_clock.h"

SystemClock system_clock;

SystemClock::SystemClock(void)
	: m_time(0)
{
}
SystemClock::~SystemClock(void)
{
}

void SystemClock::sleep(Time delta_time) const
{
	Time end_time = time() + delta_time;
	while (time() < end_time) {}
}

bool SystemClock::timeout(Time max_delta_time, TimeoutActionFunction timeout_action_function) const
{
	Time max_time = time() + max_delta_time;
	while (time() < max_time)
		if (timeout_action_function()) break;

	if (time() < max_time) return false;
	else                   return true;
}
