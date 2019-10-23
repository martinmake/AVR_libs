#ifndef _EXTERNAL_INTETTUPT_EXTERNAL_INTETTUPT_H_
#define _EXTERNAL_INTETTUPT_EXTERNAL_INTETTUPT_H_

#include <util.h>

class ExternalInterrupt
{
	public: // TYPES
		using on_interrupt_func = void (*)(void);
		enum INTERRUPT : uint8_t
		{
			INTERRUPT0,
			INTERRUPT1,
		};
		enum class SENSE : uint8_t
		{
			LOW_LEVEL,
			LOGICAL_CHANGE,
			FALLING_EDGE,
			RISING_EDGE,
		};
		struct Spec
		{
			INTERRUPT         interrupt;
			SENSE             sense;
			on_interrupt_func on_interrupt;
		};

	public: // CONSTRUCTORS
		ExternalInterrupt(void) = default;
		ExternalInterrupt(const Spec& spec);
	public: // DESTRUCTOR
		~ExternalInterrupt(void) = default;

	public: // STATIC VARIABLES
		static on_interrupt_func on_interrupt0;
		static on_interrupt_func on_interrupt1;

	public: // METHODS
		void init(const Spec& spec);
};

#endif
