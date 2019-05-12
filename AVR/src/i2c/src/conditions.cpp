#include <avr/io.h>
#include <avr/interrupt.h>
#include <util/delay.h>

#include <standard/standard.h>

#include "i2c.h"
#include "conditions.h"

Bit start_bit     (TWCR, TWSTA);
Bit stop_bit      (TWCR, TWSTO);
Bit interrupt_bit (TWCR, TWINT);

uint8_t  addr   = 0;
uint8_t  count  = 0;
uint8_t* buffer = nullptr;
uint8_t  index  = 0;

void send_start()
{
	i2c.transceive_failed    = false;
	i2c.transceive_completed = false;
	start_bit.set();
	interrupt_bit.set();
}

void send_slaw(uint8_t addr)
{
	TWDR = addr;
	start_bit.clear();
	interrupt_bit.set();
}

void send_stop()
{
	stop_bit.set();
	interrupt_bit.set();
	_delay_ms(1);
	stop_bit.clear();
	i2c.transceive_completed = true;
}

ISR(TWI_vect)
{
	Status status = static_cast<Status>(TWSR & 0xf8);
	switch (status) {
		case Status::START:
		case Status::RSTART:
			send_slaw(addr);
			break;
		case Status::SLAW_ACK:
		case Status::WDATA_ACK:
			if (index < count) {
				TWDR = buffer[index++];
			} else
				send_stop();
			break;
		case Status::SLAR_ACK:
		case Status::RDATA_ACK:
			if (index < count) {
				buffer[index++] = TWDR;
			} else
				send_stop();
			break;
		case Status::SLAW_NACK:
		case Status::WDATA_NACK:
		case Status::SLAR_NACK:
		case Status::RDATA_NACK:
		default:
			i2c.transceive_failed = true;
			send_stop();
			break;
	}

	interrupt_bit.set();
}
