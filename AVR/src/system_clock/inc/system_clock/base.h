#ifndef _SYSTEM_CLOCK_BASE_H_
#define _SYSTEM_CLOCK_BASE_H_

#include <util.h>

namespace SystemClock
{
	class Base
	{
		public: // TYPES
			using TimeoutActionFunction = void (*)(void);
			using Time                  = uint64_t;

		public: // CONSTRUCTORS
			Base(void);
		public: // DESTRUCTOR
			virtual ~Base(void);

		public: // GETTERS
			virtual Time time(void) const = 0;

		public: // METHODS
			virtual void init(void) = 0;
			bool timeout(Time max_delta_time, TimeoutActionFunction timeout_action_function) const;
		protected:
			Time m_time;
	};
}

#endif
