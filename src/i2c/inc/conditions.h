#ifndef _I2C_CONDITIONS_H_
#define _I2C_CONDITIONS_H_

typedef enum class status : uint8_t {
	START      = 0x08,
	RSTART     = 0x10,
	SLAW_ACK   = 0x18,
	SLAW_NACK  = 0x20,
	WDATA_ACK  = 0x28,
	WDATA_NACK = 0x30,
	SLAR_ACK   = 0x40,
	SLAR_NACK  = 0x48,
	RDATA_ACK  = 0x50,
	RDATA_NACK = 0x58
} STATUS;

extern Bit start_bit;
extern Bit stop_bit;
extern Bit interrupt_bit;

extern uint8_t  addr;
extern uint8_t  count;
extern uint8_t* buffer;

extern void start();
extern void slaw(uint8_t addr);
extern void stop();

#endif
