#include <avr/io.h>
#include <avr/interrupt.h>
#include <util/delay.h>

#include <standard/standard.h>
#include <usart/usart.h>

#include "i2c.h"
#include "conditions.h"

namespace I2c
{
	volatile bool transceive_completed;
	volatile bool transceive_failed;

	void begin(const INIT* init)
	{
		uint8_t ps_val = 1;
		for (uint8_t i = 0; i < (uint8_t) init->ps; i++)
			ps_val *= 4;

		TWBR  = (init->f_osc/init->f_scl - 16)/2/ps_val;

		uint8_t ps = (uint8_t) init->ps;
		Bit(TWSR, TWPS0).write(ps & 0b01);
		Bit(TWSR, TWPS1).write(ps & 0b10);

		PORTC |= (1 << PC5) | (1 << PC4);

		TWCR |= (1 << TWEN) | (1 << TWIE) | (1 << TWEA);
	}

	void begin(uint32_t f_scl, uint32_t f_osc)
	{
			INIT init;

			init.x2    = X2::OFF;
			init.ps    = PS::PS64;
			init.f_scl = f_scl;
			init.f_osc = f_osc;

			begin(&init);
	}

	void write(uint8_t _addr, uint8_t _count, uint8_t* _buffer)
	{
		addr   = (_addr << 1) | 0;
		count  = _count;
		buffer = _buffer;
		start();
	}

	void read(uint8_t _addr, uint8_t _count, uint8_t* _buffer)
	{
		addr   = (_addr << 1) | 1;
		count  = _count;
		buffer = _buffer;
		start();
	}
}
