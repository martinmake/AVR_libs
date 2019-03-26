#ifndef _USART_USART_H_
#define _USART_USART_H_

namespace Usart
{
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
		S1, S2
	} STOP_BIT_SELECT;

	typedef enum class character_size : uint8_t {
		S5 = 0b000,
		S6 = 0b001,
		S7 = 0b010,
		S8 = 0b011,
		S9 = 0b111
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
	void begin(uint32_t baud, uint32_t f_osc);
	void sendc(char c);
	void sends(const char* s);
	char recvc();
	void sendf(uint16_t size, const char* format, ...);
}

#endif
