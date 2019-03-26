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

	void begin(const INIT* init);
	void begin(uint32_t f_scl, uint32_t f_osc);
	bool write(uint8_t addr, uint8_t data);
	// uint8_t read(uint8_t addr);

	bool _start();
	bool _slaw(uint8_t addr);
	bool _data(uint8_t data);
	void _stop();
}

#endif
