#ifndef _SPI_H_
#define _SPI_H_

#include <avr/io.h>

#include <standard/standard.h>

namespace Spi
{
	typedef enum class clock_rate_select {
		S2   = 0b100,
		S4   = 0b000,
		S8   = 0b100,
		S16  = 0b001,
		S32  = 0b110,
		S64  = 0b010,
		S128 = 0b011
	} CLOCK_RATE_SELECT;

	typedef struct {
		CLOCK_RATE_SELECT clock_rate_select;
		Bit sck_ddb;
		Bit miso_ddb;
		Bit mosi_ddb;
	} INIT;

	void begin(const INIT* init);
	uint8_t send(uint8_t data);

	class Slave
	{
		private:
			Pin m_ss;

		public:
			Slave(Pin ss);
			~Slave();

			void select();
			void unselect();
	};
}

#endif
