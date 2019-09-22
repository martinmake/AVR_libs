#ifndef _I2C_II2C_H_
#define _I2C_II2C_H_

#include <util.h>

class II2c
{
	public: // CONSTRUCTORS
		II2c(void);

	public: // DESTRUCTOR
		virtual ~II2c(void);

	public: // FUNCTIONS
		void write_reg(uint8_t dev_addr, uint8_t reg_addr, uint8_t  data);
		void write_reg(uint8_t dev_addr, uint8_t reg_addr, uint8_t* data, uint16_t count);

		uint8_t read_reg(uint8_t dev_addr, uint8_t reg_addr);
		void    read_reg(uint8_t dev_addr, uint8_t reg_addr, uint8_t* data, uint16_t count);

	public: // VIRTUAL FUNCTIONS
		virtual void init(void) = 0;

		virtual void start(void) = 0;
		virtual void stop (void) = 0;

		virtual bool    write(uint8_t data) = 0;
		virtual uint8_t read (void        ) = 0;

	private:
		static const uint8_t W = 0;
		static const uint8_t R = 1;
};

inline void II2c::write_reg(uint8_t dev_addr, uint8_t reg_addr, uint8_t data)
{
	start();
	write(dev_addr);
	write(reg_addr);
	write(data);
	stop();
}
inline void II2c::write_reg(uint8_t dev_addr, uint8_t reg_addr, uint8_t* data, uint16_t count)
{
	start();
	write((dev_addr << 1) | W);
	write(reg_addr);
	for (uint8_t* p_data = data; p_data < data + count; p_data++)
		write(*data);
	stop();
}

inline uint8_t II2c::read_reg(uint8_t dev_addr, uint8_t reg_addr)
{
	uint8_t data;

	start();
	write(dev_addr);
	write(reg_addr);

	start();
	write((dev_addr << 1) | R);
	data = read();
	stop();

	return data;
}
inline void II2c::read_reg(uint8_t dev_addr, uint8_t reg_addr, uint8_t* data, uint16_t count)
{
	start();
	write(dev_addr);
	write(reg_addr);

	start();
	write(dev_addr | 0x01);
	for (uint8_t* p_data = data; p_data < data + count; p_data++)
		*data = read();
	stop();
}

#endif
