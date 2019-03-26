#include <avr/io.h>
#include <avr/interrupt.h>

#include <standard/standard.h>
#include <usart/usart.h>

#include "i2c.h"

namespace I2c
{
	Bit start_bit     (TWCR, TWSTA);
	Bit stop_bit      (TWCR, TWSTO);
	Bit interrupt_bit (TWCR, TWINT);

	void begin(const INIT* init)
	{
		uint8_t ps_val = 1;
		for (uint8_t i = 0; i < (uint8_t) init->ps; i++)
			ps_val *= 4;

		TWBR  = (init->f_osc/init->f_scl - 16)/2/ps_val;
		TWCR |= (1 << TWEN);

		uint8_t ps = (uint8_t) init->ps;
		Bit(TWSR, TWPS0).write(ps & 0b01);
		Bit(TWSR, TWPS1).write(ps & 0b10);

		PORTC |= (1 << PC5) | (1 << PC4);
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

	bool write(uint8_t addr, uint8_t data)
	{
		Usart::sendf(20, "\naddr: 0x%02x\n", addr);

		if (!_start())
			return false;

		if (!_slaw(addr))
			return false;

		if (!_data(data))
			return false;

		_stop();

		return true;
	}

	bool _start()
	{
		stop_bit.clear();
		start_bit.set();
		interrupt_bit.set();

		while(!interrupt_bit.read())
			;

		if ((TWSR & 0xf8) == 0x08)
			Usart::sends("START!\n");
		else {
			Usart::sends("START ERROR!\n");
			_stop();
			return false;
		}

		return true;
	}

	bool _slaw(uint8_t addr)
	{
		TWDR = addr << 1;
		start_bit.clear();
		interrupt_bit.set();
		while(!interrupt_bit.read())
			;

		if ((TWSR & 0xf8) == 0x18)
			Usart::sends("ACK!\n");
		else if ((TWSR & 0xf8) == 0x20) {
			Usart::sends("NACK!\n");
			_stop();
			return false;
		} else if ((TWSR & 0xf8) == 0x38) {
			Usart::sends("ARBITRATION LOST!\n");
			_stop();
			return false;
		} else if ( ((TWSR & 0xf8) == 0x68) | ((TWSR & 0xf8) == 0x78) | ((TWSR & 0xf8) == 0xb0) ) {
			Usart::sends("ARBITRATION LOST AND ADDRESSED AS SLAVE!\n");
			_stop();
			return false;
		} else {
			Usart::sendf(30, "UNKNOWN ERROR: 0x%02x\n", TWSR & 0xf8);
			_stop();
			return false;
		}

		return true;
	}

	bool _data(uint8_t data)
	{
		TWDR = data;
		interrupt_bit.set();
		while(!interrupt_bit.read())
			;

		if ((TWSR & 0xf8) == 0x28)
			Usart::sends("ACK!\n");
		else if ((TWSR & 0xf8) == 0x30) {
			Usart::sends("NACK!\n");
			_stop();
			return false;
		} else if ((TWSR & 0xf8) == 0x38) {
			Usart::sends("ARBITRATION LOST!\n");
			_stop();
			return false;
		} else {
			Usart::sendf(30, "UNKNOWN ERROR: 0x%02x\n", TWSR & 0xf8);
			_stop();
			return false;
		}

		return true;
	}

	void _stop()
	{
		stop_bit.set();
		interrupt_bit.set();
	}
}
