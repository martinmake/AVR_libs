#if defined(__AVR_ATmega328P__)

#include "core/timer/timer2.h"

namespace Core
{
	Timer::Timer2 timer2;

	namespace Timer
	{
		// CONSTRUCTORS
		Timer2::Timer2(const Spec& spec)
		{
			initialize(spec);
		}
	}
}

using namespace Core;
using namespace Timer;
using namespace TIMER;
ISR(TIMER2_OVF_vect  ) { timer2.call<INTERRUPT::ON_OVERFLOW              >(); }
ISR(TIMER2_COMPA_vect) { timer2.call<INTERRUPT::ON_OUTPUT_COMPARE_MATCH_A>(); }
ISR(TIMER2_COMPB_vect) { timer2.call<INTERRUPT::ON_OUTPUT_COMPARE_MATCH_B>(); }

#endif
