#ifndef _SYSTEM_CLOCK_TIMER0_H_
#define _SYSTEM_CLOCK_TIMER0_H_

#include <util.h>

#include "system_clock/base.h"

namespace SystemClock
{
	class Timer0 : public SystemClock::Base
	{
		public: // CONSTRUCTORS
			Timer0(void);
		public: // DESTRUCTOR
			~Timer0(void);

		public: // GETTERS
			Time time(void) const override;

		public: // METHODS
			void init(void) override;
			bool timeout(Time max_delta_time, TimeoutActionFunction timeout_action_function) const;
	};

	// GETTERS
	inline Timer0::Time Timer0::time(void) const { return m_time; }
}

#endif
