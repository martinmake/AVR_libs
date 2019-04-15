#include <avr/io.h>
#include <avr/interrupt.h>
#include <util/delay.h>

#include <standard/standard.h>
#include <usart/usart.h>

#include "i2c.h"
#include "conditions.h"

Bit start_bit     (TWCR, TWSTA);
Bit stop_bit      (TWCR, TWSTO);
Bit interrupt_bit (TWCR, TWINT);

uint8_t  addr   = 0;
uint8_t  count  = 0;
uint8_t* buffer = nullptr;
uint8_t  index  = 0;

void start()
{
	I2c::transceive_failed = false;
	I2c::transceive_completed = false;
	start_bit.set();
	interrupt_bit.set();
}

void slaw(uint8_t addr)
{
	TWDR = addr;
	start_bit.clear();
	interrupt_bit.set();
}

void stop()
{
	stop_bit.set();
	interrupt_bit.set();
	_delay_ms(1);
	stop_bit.clear();
	I2c::transceive_completed = true;
}

ISR(TWI_vect)
{
	STATUS status = static_cast<STATUS>(TWSR & 0xf8);
	switch (status) {
		case STATUS::START:
		case STATUS::RSTART:
			slaw(addr);
			break;
		case STATUS::SLAW_ACK:
		case STATUS::WDATA_ACK:
			if (index < count) {
				TWDR = buffer[index++];
			} else
				stop();
			break;
		case STATUS::SLAR_ACK:
		case STATUS::RDATA_ACK:
			if (index < count) {
				buffer[index++] = TWDR;
			} else
				stop();
			break;
		case STATUS::SLAW_NACK:
		case STATUS::WDATA_NACK:
		case STATUS::SLAR_NACK:
		case STATUS::RDATA_NACK:
		default:
			I2c::transceive_failed = true;
			stop();
			break;
	}

	interrupt_bit.set();
}
