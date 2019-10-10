#ifndef _ADC_ADC_H_
#define _ADC_ADC_H_

#include <util.h>
#include <avr/io.h>

class Adc
{
	public: // TYPES
		enum class PRESCALER           : uint8_t { X2, X4, X8, X16, X32, X64, X128 };
		enum class AUTO_TRIGGER_SOURCE : uint8_t { FREE_RUNNING, EXTERNAL_INTERRUPT_REQUEST_0, TIM0_COMPARE_MATCH_A, TIM0_OVERFLOW, TIM1_COMPARE_MATCH_B, TIM1_OVERFLOW, TIM1_CAPTURE_EVENT };
		enum class VREF                : uint8_t { AREF, AVCC, IREF };
		struct Init
		{
			PRESCALER           prescaler           = PRESCALER::X128;
			AUTO_TRIGGER_SOURCE auto_trigger_source = AUTO_TRIGGER_SOURCE::FREE_RUNNING;
			VREF                vref                = VREF::AVCC;
		};
		using on_conversion_func = void (*)(void);

	public: // CONTRUCTORS
		Adc(void);
		Adc(const Init& init_init);

	public: // PUBLIC VARIABLES
		on_conversion_func on_conversion;
		volatile uint16_t value = 0;

	public: // SETTERS
		void channel(uint8_t channel);

	public: // FUNCTIONS
		void init(const Init& init_init);

		void start_sampling(void);
		void  stop_sampling(void);
		uint16_t take_sample(void);

	public: // OPERATORS
		Adc& operator++();
		Adc& operator--();
};

// SETTERS
inline void Adc::channel(uint8_t channel) { ADMUX = (ADMUX & 0b11110000) | channel; }

// FUNCTIONS
inline void Adc::start_sampling() { ADCSRA |=   BIT(ADIE) | BIT(ADATE) | BIT(ADSC); }
inline void Adc:: stop_sampling() { ADCSRA &= ~(BIT(ADIE) | BIT(ADATE));            }
inline uint16_t Adc::take_sample(void)
{
	SET(ADCSRA, ADSC);
	while (IS_SET(ADCSRA, ADSC)) {}
	return ADC;
}

// OPERATORS
inline Adc& Adc::operator++() { ADMUX = (ADMUX & 0b11110000) | ((ADMUX & 0b00001111) == 0b00001000 ? 0b00000000 : ((ADMUX & 0b00001111) + 1)); return *this; }
inline Adc& Adc::operator--() { ADMUX = (ADMUX & 0b11110000) | ((ADMUX & 0b00001111) == 0b00000000 ? 0b00001000 : ((ADMUX & 0b00001111) - 1)); return *this; }

extern Adc adc;

#endif
