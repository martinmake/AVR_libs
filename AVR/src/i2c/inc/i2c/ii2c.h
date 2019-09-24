#ifndef _I2C_II2C_H_
#define _I2C_II2C_H_

#include <util.h>

class II2c
{
	public: // CONSTRUCTORS
		II2c(void);

	public: // DESTRUCTOR
		virtual ~II2c(void);

	public: // GETTERS
		bool ack(void) const;

	public: // FUNCTIONS
		void write_reg      (uint8_t dev_addr, uint8_t reg_addr, uint8_t  data);
		void write_reg_8bit (uint8_t dev_addr, uint8_t reg_addr, uint8_t  data);
		void write_reg_16bit(uint8_t dev_addr, uint8_t reg_addr, uint16_t data);
		void write_reg      (uint8_t dev_addr, uint8_t reg_addr, uint8_t* data, uint16_t count);
		void write_reg_8bit (uint8_t dev_addr, uint8_t reg_addr, uint8_t* data, uint16_t count);

		uint8_t  read_reg      (uint8_t dev_addr, uint8_t reg_addr);
		uint8_t  read_reg_8bit (uint8_t dev_addr, uint8_t reg_addr);
		uint16_t read_reg_16bit(uint8_t dev_addr, uint8_t reg_addr);
		void     read_reg      (uint8_t dev_addr, uint8_t reg_addr, uint8_t* data, uint16_t count);
		void     read_reg_8bit (uint8_t dev_addr, uint8_t reg_addr, uint8_t* data, uint16_t count);

	public: // VIRTUAL FUNCTIONS
		virtual void init(void) = 0;

		virtual void start(void) = 0;
		virtual void stop (void) = 0;

		virtual bool    write(uint8_t data      ) = 0;
		virtual uint8_t read (bool    ack = NACK) = 0;

	protected:
		static const uint8_t W = 0b00000000;
		static const uint8_t R = 0b00000001;
	protected:
		bool m_ack;
};

inline void II2c::write_reg(uint8_t dev_addr, uint8_t reg_addr, uint8_t data)
{
	write_reg_8bit(dev_addr, reg_addr, data);
}
inline void II2c::write_reg_8bit(uint8_t dev_addr, uint8_t reg_addr, uint8_t data)
{
	start();
	write((dev_addr << 1) | W);
	write(reg_addr);
	write(data);
	stop();
}
inline void II2c::write_reg_16bit(uint8_t dev_addr, uint8_t reg_addr, uint16_t data)
{
	start();
	write((dev_addr << 1) | W);
	write(reg_addr);
	write((data >> 8) & 0xff);
	write((data >> 0) & 0xff);
	stop();
}
inline void II2c::write_reg(uint8_t dev_addr, uint8_t reg_addr, uint8_t* data, uint16_t count)
{
	write_reg_8bit(dev_addr, reg_addr, data, count);
}
inline void II2c::write_reg_8bit(uint8_t dev_addr, uint8_t reg_addr, uint8_t* data, uint16_t count)
{
	start();
	write((dev_addr << 1) | W);
	write(reg_addr);
	for (uint8_t* p_data = data; p_data < data + count; p_data++)
		write(*p_data);
	stop();
}

inline uint8_t II2c::read_reg(uint8_t dev_addr, uint8_t reg_addr)
{
	return read_reg_8bit(dev_addr, reg_addr);
}
inline uint8_t II2c::read_reg_8bit(uint8_t dev_addr, uint8_t reg_addr)
{
	uint8_t data;

	start();
	write((dev_addr << 1) | W);
	write(reg_addr);

	start();
	write((dev_addr << 1) | R);
	data = read();
	stop();

	return data;
}
inline uint16_t II2c::read_reg_16bit(uint8_t dev_addr, uint8_t reg_addr)
{
	uint16_t data = 0x0000;

	start();
	write((dev_addr << 1) | W);
	write(reg_addr);

	start();
	write((dev_addr << 1) | R);
	data |= read( ACK) << 8;
	data |= read(NACK) << 0;
	stop();

	return data;
}
inline void II2c::read_reg(uint8_t dev_addr, uint8_t reg_addr, uint8_t* data, uint16_t count)
{
	read_reg_8bit(dev_addr, reg_addr, data, count);
}
inline void II2c::read_reg_8bit(uint8_t dev_addr, uint8_t reg_addr, uint8_t* data, uint16_t count)
{
	start();
	write((dev_addr << 1) | W);
	write(reg_addr);

	start();
	write((dev_addr << 1) | R);
	for (uint8_t* p_data = data; p_data < data + count - 1; p_data++)
		*p_data = read( ACK);
	data[count - 1] = read(NACK);
	stop();
}

// GETTERS
inline bool II2c::ack(void) const { return m_ack; }

#endif
