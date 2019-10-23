#if defined(__AVR_ATmega328P__)

#include "core/timer/timer1.h"

namespace Core
{
	Timer::Timer1 timer1;

	namespace Timer
	{
		// CONSTRUCTORS
		Timer1::Timer1(const Spec& spec)
		{
			initialize(spec);
		}
	}
}

using namespace Core;
using namespace Timer;
using namespace TIMER;
ISR(TIMER1_COMPA_vect) { timer1.call<INTERRUPT::ON_OUTPUT_COMPARE_MATCH_A>(); }
ISR(TIMER1_COMPB_vect) { timer1.call<INTERRUPT::ON_OUTPUT_COMPARE_MATCH_B>(); }
ISR(TIMER1_CAPT_vect ) { timer1.call<INTERRUPT::ON_INPUT_CAPTURE         >(); }
ISR(TIMER1_OVF_vect  ) { timer1.call<INTERRUPT::ON_OVERFLOW              >(); }

#endif
