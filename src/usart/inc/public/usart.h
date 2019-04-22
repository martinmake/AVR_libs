#ifndef _USART_USART_H_
#define _USART_USART_H_

class Usart: public UsartInterface
{
	public:
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

	public:
		void begin(const INIT* init);
		void sendc(char c);
		char recvc();
}

#endif
