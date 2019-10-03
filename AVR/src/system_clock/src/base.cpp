#include "system_clock/base.h"

namespace SystemClock
{
	Base::Base(void)
	{
	}
	Base::~Base(void)
	{
	}

	bool Base::timeout(Time max_delta_time, TimeoutActionFunction timeout_action_function) const
	{
		Time max_time = time() + max_delta_time;
		while (time() < max_time)
			timeout_action_function();

		if (time() < max_time) return true;
		else                   return false;
	}
}
