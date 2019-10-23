#if defined(__AVR_ATmega328P__)

#include "timer/timer1.h"

namespace Timer
{
	Timer1 timer1;

	// CONSTRUCTORS
	Timer1::Timer1(const Spec& spec)
	{
		initialize(spec);
	}
}

using namespace Timer;
ISR(TIMER1_COMPA_vect) { timer1.call<Timer1::INTERRUPT::ON_OUTPUT_COMPARE_MATCH_A>(); }
ISR(TIMER1_COMPB_vect) { timer1.call<Timer1::INTERRUPT::ON_OUTPUT_COMPARE_MATCH_B>(); }
ISR(TIMER1_CAPT_vect ) { timer1.call<Timer1::INTERRUPT::ON_INPUT_CAPTURE         >(); }
ISR(TIMER1_OVF_vect  ) { timer1.call<Timer1::INTERRUPT::ON_OVERFLOW              >(); }

#endif
