#if defined(__AVR_ATmega328P__)

#include <avr/interrupt.h>

#include <util.h>

#include "timer/timer0.h"

namespace Timer
{
	Timer0 timer0;

	// CONSTRUCTORS
	Timer0::Timer0(const Spec& spec)
	{
		initialize(spec);
	}
}

using namespace Timer;
ISR(TIMER0_COMPA_vect) { timer0.call<Timer0::INTERRUPT::ON_OUTPUT_COMPARE_MATCH_A>(); }
ISR(TIMER0_COMPB_vect) { timer0.call<Timer0::INTERRUPT::ON_OUTPUT_COMPARE_MATCH_B>(); }
ISR(TIMER0_OVF_vect  ) { timer0.call<Timer0::INTERRUPT::ON_OVERFLOW              >(); }

#endif
