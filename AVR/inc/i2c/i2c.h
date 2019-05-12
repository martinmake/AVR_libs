#ifndef _I2C_I2C_H_
#define _I2C_I2C_H_

class I2c
{
	public:
		enum class X2 : uint8_t {
			OFF, ON
		};

		enum class Ps : uint8_t {
			PS1  = 1,
			PS4  = 4,
			PS16 = 16,
			PS64 = 64
		};

		struct Init {
			X2       x2;
			Ps       ps;
			uint32_t f_scl;
			uint32_t f_osc;
		};

	public:
		volatile bool transceive_completed;
		volatile bool transceive_failed;

	public:
		I2c(const Init* init);
		I2c(uint32_t f_scl, uint32_t f_osc);

		void write(uint8_t _addr, uint8_t _count, uint8_t* _buffer);
		void read (uint8_t _addr, uint8_t _count, uint8_t* _buffer);
		void wait_until_transceive_completed();
};

extern I2c i2c;

#endif
