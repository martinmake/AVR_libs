#ifndef _SPI_H_
#define _SPI_H_

#include <avr/io.h>

#include <standard/standard.h>

class Spi
{
	public:
		enum class ClockRateSelect {
			S2, S4, S8, S16, S32, S64, S128
		};

		struct Init {
			ClockRateSelect clock_rate_select;
			Bit             sck_ddb;
			Bit             miso_ddb;
			Bit             mosi_ddb;
		};

	public:
		Spi(Init* init);
		~Spi();

		inline Spi& operator<<(uint8_t  data) { SPDR = data; while (!(SPSR & (1 << SPIF))) {} return *this; }
		inline Spi& operator>>(uint8_t& data) { data = SPDR; return *this; }
};

#endif
