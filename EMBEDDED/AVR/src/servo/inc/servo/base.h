#ifndef _SERVO_BASE_H_
#define _SERVO_BASE_H_

#include <util.h>

namespace Servo
{
	class Base
	{
		public: // TYPES
			using PulseWidth = uint16_t;

		protected: // CONSTRUCTORS
			Base(void) = default;
		protected: // DESTRUCTOR
			virtual ~Base(void) = default;

		protected: // SETTERS
			virtual void pulse_width(PulseWidth new_pulse_width) = 0;

		protected: // METHODS
			virtual void init(void) = 0;
	};
}

#endif
