#ifndef _TIMER_BASE_H_
#define _TIMER_BASE_H_

#include <util.h>

namespace Timer
{
	class Base
	{
		public: // CONSTRUCTORS
			Base(void) = default;
		public: // DESTRUCTOR
			virtual ~Base(void) = default;

		public: // METHODS
			virtual void   pause(void) = 0;
			virtual void unpause(void) = 0;
	};
}

#endif
