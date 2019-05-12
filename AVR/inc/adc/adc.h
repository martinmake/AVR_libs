#ifndef _ADC_ADC_H_
#define _ADC_ADC_H_

class Adc
{
	public:
		enum class PrescalerSelect : uint8_t {
			S2, S4, S8, S16, S32, S64, S128
		};

		enum class Vref : uint8_t {
			AREF, AVCC, IREF
		};

		struct Init {
			PrescalerSelect prescaler_select;
			Vref            vref;
		};

	public:
		Adc(const Init* init);
		Adc();

		inline void start_conversion() { Bit(ADCSRA, ADSC).set(); }
		inline void select_channel(uint8_t channel) { ADMUX = (ADMUX & 0b11100000) | channel; }
		inline Adc  operator++()    { ADMUX = (ADMUX & 0b11100000) | ((ADMUX & 0b00011111) + 1) % 0b00100000; return *this; };
		inline Adc& operator++(int) { ADMUX = (ADMUX & 0b11100000) | ((ADMUX & 0b00011111) + 1) % 0b00100000; return *this; };
		inline Adc  operator--()    { ADMUX = (ADMUX & 0b11100000) | ((ADMUX & 0b00011111) - 1) % 0b00100000; return *this; };
		inline Adc& operator--(int) { ADMUX = (ADMUX & 0b11100000) | ((ADMUX & 0b00011111) - 1) % 0b00100000; return *this; };
};

#endif
