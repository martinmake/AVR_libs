#ifndef _I2C_BASE_H_
#define _I2C_BASE_H_

#include <util.h>

namespace I2c
{
	class Base
	{
		public: // CONSTRUCTORS
			Base(void);

		public: // DESTRUCTOR
			virtual ~Base(void);

		public: // GETTERS
			bool ack(void) const;

		public: // FUNCTIONS
			void write_register      (uint8_t device_address, uint8_t register_address, uint8_t  data);
			void write_register_8bit (uint8_t device_address, uint8_t register_address, uint8_t  data);
			void write_register_16bit(uint8_t device_address, uint8_t register_address, uint16_t data);
			void write_register      (uint8_t device_address, uint8_t register_address, uint8_t* data, uint16_t count);
			void write_register_8bit (uint8_t device_address, uint8_t register_address, uint8_t* data, uint16_t count);

			uint8_t  read_register      (uint8_t device_address, uint8_t register_address);
			uint8_t  read_register_8bit (uint8_t device_address, uint8_t register_address);
			uint16_t read_register_16bit(uint8_t device_address, uint8_t register_address);
			void     read_register      (uint8_t device_address, uint8_t register_address, uint8_t* data, uint16_t count);
			void     read_register_8bit (uint8_t device_address, uint8_t register_address, uint8_t* data, uint16_t count);

		public: // VIRTUAL FUNCTIONS
			virtual void init(void) = 0;

			virtual void start(void) = 0;
			virtual void stop (void) = 0;

			virtual void    write(uint8_t data      ) = 0;
			virtual uint8_t read (bool    ack = NACK) = 0;

		protected:
			static const uint8_t W = 0b00000000;
			static const uint8_t R = 0b00000001;
		protected:
			bool m_ack;
	};

	inline void Base::write_register(uint8_t device_address, uint8_t register_address, uint8_t data)
	{
		write_register_8bit(device_address, register_address, data);
	}
	inline void Base::write_register_8bit(uint8_t device_address, uint8_t register_address, uint8_t data)
	{
		start();
		write((device_address << 1) | W);
		write(register_address);
		write(data);
		stop();
	}
	inline void Base::write_register_16bit(uint8_t device_address, uint8_t register_address, uint16_t data)
	{
		start();
		write((device_address << 1) | W);
		write(register_address);
		write((data >> 8) & 0xff);
		write((data >> 0) & 0xff);
		stop();
	}
	inline void Base::write_register(uint8_t device_address, uint8_t register_address, uint8_t* data, uint16_t count)
	{
		write_register_8bit(device_address, register_address, data, count);
	}
	inline void Base::write_register_8bit(uint8_t device_address, uint8_t register_address, uint8_t* data, uint16_t count)
	{
		start();
		write((device_address << 1) | W);
		write(register_address);
		for (uint8_t* p_data = data; p_data < data + count; p_data++)
			write(*p_data);
		stop();
	}

	inline uint8_t Base::read_register(uint8_t device_address, uint8_t register_address)
	{
		return read_register_8bit(device_address, register_address);
	}
	inline uint8_t Base::read_register_8bit(uint8_t device_address, uint8_t register_address)
	{
		uint8_t data;

		start();
		write((device_address << 1) | W);
		write(register_address);

		start();
		write((device_address << 1) | R);
		data = read();
		stop();

		return data;
	}
	inline uint16_t Base::read_register_16bit(uint8_t device_address, uint8_t register_address)
	{
		uint16_t data = 0x0000;

		start();
		write((device_address << 1) | W);
		write(register_address);

		start();
		write((device_address << 1) | R);
		data |= read( ACK) << 8;
		data |= read(NACK) << 0;
		stop();

		return data;
	}
	inline void Base::read_register(uint8_t device_address, uint8_t register_address, uint8_t* data, uint16_t count)
	{
		read_register_8bit(device_address, register_address, data, count);
	}
	inline void Base::read_register_8bit(uint8_t device_address, uint8_t register_address, uint8_t* data, uint16_t count)
	{
		start();
		write((device_address << 1) | W);
		write(register_address);

		start();
		write((device_address << 1) | R);
		for (uint8_t* p_data = data; p_data < data + count - 1; p_data++)
			*p_data = read( ACK);
		data[count - 1] = read(NACK);
		stop();
	}

	// GETTERS
	inline bool Base::ack(void) const { return m_ack; }
}

#endif
