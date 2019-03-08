#ifndef _USART_USART_H_
#define _USART_USART_H_

namespace Usart
{
	typedef enum class nl : uint8_t {
		OFF, ON
	} NL;

	typedef enum class x2 : uint8_t {
		OFF, ON
	} X2;

	typedef enum class rx : uint8_t {
		OFF, ON
	} RX;

	typedef enum class tx : uint8_t {
		OFF, ON
	} TX;

	typedef enum class stop_bit_select : uint8_t {
		ONE, TWO
	} STOP_BIT_SELECT;

	typedef enum class character_size : uint8_t {
		FIVE  = 0b000,
		SIX   = 0b001,
		SEVEN = 0b010,
		EIGHT = 0b011,
		NINE  = 0b111
	} CHARACTER_SIZE;

	typedef struct {
		X2              x2;
		RX              rx;
		TX              tx;
		CHARACTER_SIZE  character_size;
		STOP_BIT_SELECT stop_bit_select;
		uint32_t        baud;
		uint32_t        f_osc;
	} INIT;

	void begin(const INIT* init);
	void begin_simple(uint32_t baud, uint32_t f_osc);
	void send_char(char c);
	void send_bits(uint8_t bits, NL nl);
	void send_str(const char* s);
	void sendf(uint16_t size, const char* format, ...);
}

#endif
