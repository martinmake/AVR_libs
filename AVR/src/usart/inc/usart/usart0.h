#ifndef _USART_USART0_H_
#define _USART_USART0_H_

#include <avr/io.h>

#include <util.h>

#include "usart/iusart.h"

class Usart0: virtual public IUsart
{
	public: // TYPES
		enum class X2              : bool    { DISSABLED, ENABLED };
		enum class RX              : bool    { DISSABLED, ENABLED };
		enum class TX              : bool    { DISSABLED, ENABLED };
		enum class STOP_BIT_SELECT : uint8_t { S1, S2 };
		enum class CHARACTER_SIZE  : uint8_t { S5, S6, S7, S8, S9 };

		struct Init
		{
			uint16_t        baud;
			X2              x2                = X2::DISSABLED;
			RX              rx                = RX::ENABLED;
			TX              tx                = TX::ENABLED;
			CHARACTER_SIZE  character_size    = CHARACTER_SIZE::S8;
			STOP_BIT_SELECT stop_bit_select   = STOP_BIT_SELECT::S1;
			uint8_t         output_queue_size = DEFAULT_OUTPUT_QUEUE_SIZE;
		};

	public: // CENABLEDSTRUCTORS
		Usart0(void);
		Usart0(const Init& init_struct);

	public: // GETTERS
		FILE* stream(void) override;

	public: // FUNSTIENABLEDS
		void init(const Init& init_struct);

		void putchar(char c) override;
		char getchar(void  ) override;
		void flush  (void  ) override;

	public: // OPERATORS
		Usart0& operator<<(      char  c);
		Usart0& operator<<(const char* s);
		Usart0& operator>>(      char& c);
		Usart0& operator>>(      char* s);

	private: // PRIVATE FUNCTIENABLEDS
		void init_pipeline(const Init& init);
};

inline void Usart0::init(const Init& init_struct)
{
#ifndef F_CPU
#define F_CPU 16000000
#endif
	switch (init_struct.x2)
	{
		case X2::ENABLED:
			SET(UCSR0A, U2X0);
			UBRR0 = F_CPU/8/init_struct.baud - 1;
			break;
		case X2::DISSABLED:
			CLEAR(UCSR0A, U2X0);
			UBRR0 = F_CPU/16/init_struct.baud - 1;
			break;
	}

	switch (init_struct.rx)
	{
		case RX::ENABLED:   SET  (UCSR0B, RXEN0); break;
		case RX::DISSABLED: CLEAR(UCSR0B, RXEN0); break;
	}

	switch (init_struct.tx)
	{
		case TX::ENABLED:   SET  (UCSR0B, TXEN0); break;
		case TX::DISSABLED: CLEAR(UCSR0B, TXEN0); break;
	}

	switch (init_struct.stop_bit_select)
	{
		case STOP_BIT_SELECT::S1: SET  (UCSR0C, USBS0); break;
		case STOP_BIT_SELECT::S2: CLEAR(UCSR0C, USBS0); break;
	}

	switch (init_struct.character_size)
	{
		case CHARACTER_SIZE::S5:
			CLEAR(UCSR0C, UCSZ00);
			CLEAR(UCSR0C, UCSZ01);
			CLEAR(UCSR0B, UCSZ02);
			break;
		case CHARACTER_SIZE::S6:
			SET  (UCSR0C, UCSZ00);
			CLEAR(UCSR0C, UCSZ01);
			CLEAR(UCSR0B, UCSZ02);
			break;
		case CHARACTER_SIZE::S7:
			CLEAR(UCSR0C, UCSZ00);
			SET  (UCSR0C, UCSZ01);
			CLEAR(UCSR0B, UCSZ02);
			break;
		case CHARACTER_SIZE::S8:
			CLEAR(UCSR0C, UCSZ00);
			SET  (UCSR0C, UCSZ01);
			SET  (UCSR0B, UCSZ02);
			break;
		case CHARACTER_SIZE::S9:
			SET  (UCSR0C, UCSZ00);
			SET  (UCSR0C, UCSZ01);
			SET  (UCSR0B, UCSZ02);
			break;
	}

	SET(UCSR0B, UDRIE0);

	init_pipeline(init_struct);
}

inline void Usart0::putchar(char c)
{
	if (IS_SET(UCSR0A, UDRE0))
		UDR0 = c;
	else
		output_queue << c;
}
inline char Usart0::getchar(void)
{
	while (IS_CLEAR(UCSR0A, RXC0)) {}
	return UDR0;
}

inline void Usart0::flush(void)
{
	uint8_t sreg_save = SREG;
	cli();
	while (!output_queue.is_empty())
	{
		while (IS_CLEAR(UCSR0A, UDRE0)) {}
		output_queue >> UDR0;
	}
	SREG = sreg_save;
}

inline Usart0& Usart0::operator<<(char c)
{
	putchar(c);

	return *this;
}
inline Usart0& Usart0::operator<<(const char* s)
{
	while (*s) putchar(*s++);

	return *this;
}

inline Usart0& Usart0::operator>>(char& c)
{
	c = getchar();

	return *this;
}

extern Usart0 usart0;

#endif
