#include "ssd.h"

namespace Ssd
{
	uint8_t seg1;
	uint8_t seg2;

	void begin()
	{
		LED6DIG1.dd.set();
		LED6DIG2.dd.set();

		SSDBUS_A .dd.set();
		SSDBUS_B .dd.set();
		SSDBUS_C .dd.set();
		SSDBUS_D .dd.set();
		SSDBUS_E .dd.set();
		SSDBUS_F .dd.set();
		SSDBUS_G .dd.set();
		SSDBUS_DP.dd.set();

		TCCR0 = (1 << WGM01);	// CTC
		TCCR0 = (1 << CS01);	// clk/8
		OCR0  = 250;
		TIMSK |= (1 << OCIE0);
	}

	void set_segment(char c)
	{
		switch (c) {
			case ' ': //      ABCDEFGDP
				set_bus(0b00000000); break;
			case '!': //      ABCDEFGDP
				set_bus(0b01000001); break;
			case '"': //      ABCDEFGDP
				set_bus(0b10000100); break;
			case '\''://      ABCDEFGDP
				set_bus(0b01000000); break;
			case '(': //      ABCDEFGDP
				set_bus(0b10011100); break;
			case ')': //      ABCDEFGDP
				set_bus(0b11110000); break;
			case '-': //      ABCDEFGDP
				set_bus(0b00000010); break;
			case '.': //      ABCDEFGDP
				set_bus(0b00000001); break;
			case '0': //      ABCDEFGDP
				set_bus(0b11111100); break;
			case '1': //      ABCDEFGDP
				set_bus(0b01100000); break;
			case '2': //      ABCDEFGDP
				set_bus(0b11011010); break;
			case '3': //      ABCDEFGDP
				set_bus(0b11110010); break;
			case '4': //      ABCDEFGDP
				set_bus(0b01100110); break;
			case '5': //      ABCDEFGDP
				set_bus(0b10110110); break;
			case '6': //      ABCDEFGDP
				set_bus(0b10111110); break;
			case '7': //      ABCDEFGDP
				set_bus(0b11100000); break;
			case '8': //      ABCDEFGDP
				set_bus(0b11111110); break;
			case '9': //      ABCDEFGDP
				set_bus(0b11110110); break;
			case '=': //      ABCDEFGDP
				set_bus(0b00010010); break;
			case '?': //      ABCDEFGDP
				set_bus(0b11001010); break;
			case 'A': //      ABCDEFGDP
				set_bus(0b11101110); break;
			case 'B': //      ABCDEFGDP
				set_bus(0b00111110); break;
			case 'C': //      ABCDEFGDP
				set_bus(0b10011100); break;
			case 'D': //      ABCDEFGDP
				set_bus(0b01111010); break;
			case 'E': //      ABCDEFGDP
				set_bus(0b10011110); break;
			case 'F': //      ABCDEFGDP
				set_bus(0b10001110); break;
			case 'G': //      ABCDEFGDP
				set_bus(0b11110110); break;
			case 'H': //      ABCDEFGDP
				set_bus(0b01101110); break;
			case 'I': //      ABCDEFGDP
				set_bus(0b01100000); break;
			case 'J': //      ABCDEFGDP
				set_bus(0b01111000); break;
			case 'L': //      ABCDEFGDP
				set_bus(0b00011100); break;
			case 'N': //      ABCDEFGDP
				set_bus(0b01010100); break;
			case 'O': //      ABCDEFGDP
				set_bus(0b11111100); break;
			case 'P': //      ABCDEFGDP
				set_bus(0b11001110); break;
			case 'Q': //      ABCDEFGDP
				set_bus(0b11100110); break;
			case 'R': //      ABCDEFGDP
				set_bus(0b00001010); break;
			case 'S': //      ABCDEFGDP
				set_bus(0b10110110); break;
			case 'T': //      ABCDEFGDP
				set_bus(0b00011110); break;
			case 'U': //      ABCDEFGDP
				set_bus(0b01111100); break;
			case 'Y': //      ABCDEFGDP
				set_bus(0b01110110); break;
			case '[': //      ABCDEFGDP
				set_bus(0b10011100); break;
			case ']': //      ABCDEFGDP
				set_bus(0b11110000); break;
			case '_': //      ABCDEFGDP
				set_bus(0b00010000); break;
			case '`': //      ABCDEFGDP
				set_bus(0b11000110); break;
			case 'a': //      ABCDEFGDP
				set_bus(0b11111010); break;
			case 'b': //      ABCDEFGDP
				set_bus(0b00111110); break;
			case 'c': //      ABCDEFGDP
				set_bus(0b00011010); break;
			case 'd': //      ABCDEFGDP
				set_bus(0b01111010); break;
			case 'e': //      ABCDEFGDP
				set_bus(0b11011110); break;
			case 'f': //      ABCDEFGDP
				set_bus(0b10001110); break;
			case 'g': //      ABCDEFGDP
				set_bus(0b11110110); break;
			case 'h': //      ABCDEFGDP
				set_bus(0b00101110); break;
			case 'i': //      ABCDEFGDP
				set_bus(0b01000000); break;
			case 'j': //      ABCDEFGDP
				set_bus(0b01111000); break;
			case 'l': //      ABCDEFGDP
				set_bus(0b01100000); break;
			case 'n': //      ABCDEFGDP
				set_bus(0b00101010); break;
			case 'o': //      ABCDEFGDP
				set_bus(0b00111010); break;
			case 'p': //      ABCDEFGDP
				set_bus(0b11001110); break;
			case 'q': //      ABCDEFGDP
				set_bus(0b11100110); break;
			case 'r': //      ABCDEFGDP
				set_bus(0b00001010); break;
			case 's': //      ABCDEFGDP
				set_bus(0b10110110); break;
			case 't': //      ABCDEFGDP
				set_bus(0b00011110); break;
			case 'u': //      ABCDEFGDP
				set_bus(0b00111000); break;
			case 'y': //      ABCDEFGDP
				set_bus(0b01110110); break;
			case '{': //      ABCDEFGDP
				set_bus(0b10011100); break;
			case '|': //      ABCDEFGDP
				set_bus(0b00001100); break;
			case '}': //      ABCDEFGDP
				set_bus(0b11110000); break;
			default:  //      ABCDEFGDP
				set_bus(0b00000000); break;
		}
	}

	void display_num(uint8_t byte, BASE base)
	{
		if (byte > base*base - 1) {
			Ssd::seg1 = 'E';
			Ssd::seg2 = 'R';
			return;
		}

		uint8_t digit;

		digit   = byte / base;
		Ssd::seg1 = digit < 10 ? '0' + digit : 'A' + digit-10;
		digit   = byte - (digit * base);
		Ssd::seg2 = digit < 10 ? '0' + digit : 'A' + digit-10;
	}

	void display_str(char* s, uint16_t shift_speed /*=500*/)
	{
	for (char *p_s = s; *p_s != '\0'; p_s++)
		{
			Ssd::seg2 = *p_s;
			Ssd::seg1 = p_s == s ? ' ' : *(p_s-1);
			for (uint16_t j = 0; j < shift_speed; j++)
				wait_ms();
		}
	}

	void set_bus(uint8_t mask)
	{
		SSDBUS_A .port.write(!(mask & 0b10000000));
		SSDBUS_B .port.write(!(mask & 0b01000000));
		SSDBUS_C .port.write(!(mask & 0b00100000));
		SSDBUS_D .port.write(!(mask & 0b00010000));
		SSDBUS_E .port.write(!(mask & 0b00001000));
		SSDBUS_F .port.write(!(mask & 0b00000100));
		SSDBUS_G .port.write(!(mask & 0b00000010));
		SSDBUS_DP.port.write(!(mask & 0b00000001));
	}
}
