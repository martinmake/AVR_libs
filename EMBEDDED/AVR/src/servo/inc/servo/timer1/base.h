#ifndef _SERVO_TIMER1_BASE_H_
#define _SERVO_TIMER1_BASE_H_

#include <util.h>
#include <timer/timer1.h>

#include "servo/base.h"

namespace Servo
{
	namespace Timer1
	{
		class Base : public Servo::Base
		{
			protected: // CONSTRUCTORS
				Base(void) = default;
			protected: // DESTRUCTOR
				virtual ~Base(void) = default;

			protected: // SETTERS
				virtual void pulse_width(PulseWidth new_pulse_width) = 0;

			protected: // METHODS
				virtual void init(void) override;
		};
	}
}

#endif
