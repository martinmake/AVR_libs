#ifndef _I2C_I2C_H_
#define _I2C_I2C_H_

namespace I2c
{
	typedef enum class x2 : uint8_t {
		OFF, ON
	} X2;

	typedef enum class ps : uint8_t {
		PS1 ,
		PS4 ,
		PS16,
		PS64,
	} PS;

	typedef struct {
		X2       x2;
		PS       ps;
		uint32_t f_scl;
		uint32_t f_osc;
	} INIT;

	extern volatile bool transceive_completed;
	extern volatile bool transceive_failed;

	extern void begin(const INIT* init);
	extern void begin(uint32_t f_scl, uint32_t f_osc);
	extern void write(uint8_t _addr, uint8_t _count, uint8_t* _buffer);
	extern void read(uint8_t _addr, uint8_t _count, uint8_t* _buffer);
}

#endif
