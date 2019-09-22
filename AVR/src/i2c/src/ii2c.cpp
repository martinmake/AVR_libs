#include "i2c/ii2c.h"

II2c::II2c(void)
{
}

II2c::~II2c(void)
{
}

/*
#include <avr/io.h>
#include <avr/interrupt.h>

#include <standard/standard.h>

#include "i2c.h"
#include "conditions.h"

volatile bool transceive_completed;
volatile bool transceive_failed;

I2c::I2c(const Init* init)
{
	TWBR  = (init->f_osc/init->f_scl - 16)/2/((uint8_t) init->ps);

	switch (init->ps) {
		case Ps::PS1:
			Bit(TWSR, TWPS0) = 0;
			Bit(TWSR, TWPS1) = 0;
			break;
		case Ps::PS4:
			Bit(TWSR, TWPS0) = 1;
			Bit(TWSR, TWPS1) = 0;
			break;
		case Ps::PS16:
			Bit(TWSR, TWPS0) = 0;
			Bit(TWSR, TWPS1) = 1;
			break;
		case Ps::PS64:
			Bit(TWSR, TWPS0) = 1;
			Bit(TWSR, TWPS1) = 1;
			break;

	}

	PORTC |= (1 << PC5) | (1 << PC4);

	TWCR |= (1 << TWEN) | (1 << TWIE) | (1 << TWEA);
}

I2c::I2c(uint32_t f_scl, uint32_t f_osc)
{
		Init init;

		init.x2    = X2::OFF;
		init.ps    = Ps::PS64;
		init.f_scl = f_scl;
		init.f_osc = f_osc;

		*this = I2c(&init);
}

void I2c::write(uint8_t _addr, uint8_t _count, uint8_t* _buffer)
{
	addr   = (_addr << 1) | 0;
	count  = _count;
	buffer = _buffer;
	send_start();
}

void I2c::read(uint8_t _addr, uint8_t _count, uint8_t* _buffer)
{
	addr   = (_addr << 1) | 1;
	count  = _count;
	buffer = _buffer;
	send_start();
}

void I2c::wait_until_transceive_completed()
{
	while (!transceive_completed) {}
}
*/
