#ifndef _ADC_ADC_H_
#define _ADC_ADC_H_

#include <util/util.h>
#include <avr/io.h>

class Adc
{
	public: // TYPES
		enum class PRESCALER_SELECT : uint8_t { X2, X4, X8, X16, X32, X64, X128 };
		enum class VREF             : uint8_t { AREF, AVCC, IREF };
		struct Init
		{
			PRESCALER_SELECT prescaler_select;
			VREF            vref;
		};
		using ISR_callback_func = void (*)(uint16_t result);

	public: // CONTRUCTORS
		Adc(const Init* init);
		Adc();

	public: // PUBLIC VARIABLES
		ISR_callback_func ISR_callback;

	public: // GETTERS
		bool keep_sampling(void) const;
	public: // SETTERS
		void channel(uint8_t channel);

	public: // FUNCTIONS
		void start_conversion(void);
		void start_sampling(void);
		void  stop_sampling(void);
		uint16_t take_sample(void);

	public: // OPERATORS
		Adc& operator++();
		Adc& operator--();

	private:
		bool m_keep_sampling;
};

// GETTERS
inline bool Adc::keep_sampling(void) const { return m_keep_sampling; }
// SETTERS
inline void Adc::channel(uint8_t channel) { ADMUX = (ADMUX & 0b11110000) | channel; }

// FUNCTIONS
inline void Adc::start_conversion() { SET(ADCSRA, ADSC); }
inline void Adc::start_sampling() { m_keep_sampling = true;  SET  (ADCSRA, ADIE); start_conversion(); }
inline void Adc:: stop_sampling() { m_keep_sampling = false; CLEAR(ADCSRA, ADIE);                     }
inline uint16_t Adc::take_sample(void)
{
	start_conversion();
	while (IS_CLEAR(ADCSRA, ADIF)) {}
	return ADC;
}

// OPERATORS
inline Adc& Adc::operator++() { ADMUX = (ADMUX & 0b11110000) |                       ((ADMUX & 0b00001111) + 1) % 0b00010000;  return *this; }
inline Adc& Adc::operator--() { ADMUX = (ADMUX & 0b11110000) | (ADMUX & 0b00001111 ? ((ADMUX & 0b00001111) - 1) : 0b00001111); return *this; }

extern Adc adc;

#endif
