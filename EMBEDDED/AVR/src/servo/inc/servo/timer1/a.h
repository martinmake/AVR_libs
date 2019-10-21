#ifndef _SERVO_TIMER1_A_H_
#define _SERVO_TIMER1_A_H_

#include <util.h>

#include "servo/timer1/base.h"

#ifndef F_CPU
#define F_CPU 16000000 // SUPPRESS COMPILER ERROR
#endif

namespace Servo
{
	namespace Timer1
	{
		class A : public Servo::Timer1::Base
		{
			public: // TYPES
				using PulseWidth = uint16_t;

			public: // CONSTRUCTORS
				A(void) = default;
			public: // DESTRUCTOR
				~A(void) = default;

			public: // SETTERS
				void pulse_width(PulseWidth new_pulse_width) override;

			public: // METHODS
				void init(void) override;
		};

		// SETTERS
		inline void A::pulse_width(PulseWidth new_pulse_width)
		{
			Timer::timer1.output_compare_value_B(2 * F_CPU / 8 / new_pulse_width);
		}
	}
}

#endif
