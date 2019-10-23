#ifndef _SYSTEM_CLOCK_SYSTEM_CLOCK_H_
#define _SYSTEM_CLOCK_SYSTEM_CLOCK_H_

#include <util.h>
#include <timer/all.h>

#define SYSTEM_CLOCK_TIMEOUT(max_time_change) for (system_clock.timeout(max_time_change); !system_clock.has_timed_out(); )

#ifndef F_CPU
#define F_CPU 16000000 // SUPPRESS COMPILER ERROR
#endif

class SystemClock
{
	public: // TYPES
		using TimeoutActionFunction = bool (*)(void);
		using Time                  = uint64_t;
		enum TIMER { TIMER0, TIMER1, TIMER2 };
		struct Spec
		{
			TIMER timer;
		};

	public: // CONSTRUCTORS
		SystemClock(void);
	public: // DESTRUCTOR
		~SystemClock(void);

	public: // GETTERS
		Time time(void) const;
		bool has_timed_out(void) const;

	public: // METHODS
		void init(const Spec& init_struct);
		void sleep(Time delta_time) const;
		bool timeout(
			Time max_delta_time,
			TimeoutActionFunction timeout_action_function) const;
		void timeout(Time max_delta_time);
		void tick(void);

	private:
		Time m_time;
		Time m_end_time;
};

extern SystemClock system_clock;

// GETTERS
inline SystemClock::Time SystemClock::time         (void) const { return m_time; }
inline bool              SystemClock::has_timed_out(void) const { return time() > m_end_time; }

// METHODS
inline void SystemClock::init(const Spec& init_struct)
{
	using namespace Timer;

	switch (init_struct.timer)
	{
		case TIMER::TIMER0:
		{
			Timer0::Spec spec;
			spec.mode                      = Timer0::MODE::CTC;
			spec.clock_source              = Timer0::CLOCK_SOURCE::IO_CLK_OVER_64;
			spec.on_output_compare_match_A = []() { system_clock.tick(); };
			spec.output_compare_value_A    = F_CPU/64/1000;
			timer0.init(spec);
		} break;
		case TIMER::TIMER1: // TODO
		{
		} break;
		case TIMER::TIMER2:
		{
			Timer2::Spec spec;
			spec.mode                      = Timer2::MODE::CTC;
			spec.clock_source              = Timer2::CLOCK_SOURCE::IO_CLK_OVER_64;
			spec.on_output_compare_match_A = []() { system_clock.tick(); };
			spec.output_compare_value_A    = F_CPU/64/1000;
			timer2.init(spec);
		} break;
	}
}
inline void SystemClock::tick(void) { m_time++; }
inline void SystemClock::timeout(Time max_delta_time) { m_end_time = time() + max_delta_time; }

#endif
