#if defined(__AVR_ATmega328P__)

#include "core/timer/timer0.h"

namespace Core
{
	Timer::Timer0 timer0;

	namespace Timer
	{
		// CONSTRUCTORS
		Timer0::Timer0(const Spec& spec)
		{
			initialize(spec);
		}
	}
}


using namespace Core;
using namespace Timer;
using namespace TIMER;
ISR(TIMER0_COMPA_vect) { timer0.call<INTERRUPT::ON_OUTPUT_COMPARE_MATCH_A>(); }
ISR(TIMER0_COMPB_vect) { timer0.call<INTERRUPT::ON_OUTPUT_COMPARE_MATCH_B>(); }
ISR(TIMER0_OVF_vect  ) { timer0.call<INTERRUPT::ON_OVERFLOW              >(); }

#endif
