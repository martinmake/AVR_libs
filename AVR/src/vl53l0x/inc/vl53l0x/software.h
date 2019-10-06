#ifndef _VL53L0X_SOFTWATE_H_
#define _VL53L0X_SOFTWATE_H_

#include <gpio.h>
#include <i2c/software.h>

#include "vl53l0x/base.h"

namespace Vl53l0x
{
	template<PORT sda_port, uint8_t sda_pin, PORT scl_port, uint8_t scl_pin, uint16_t signal_length = I2C_SOFTWARE_DEFAULT_DELAY>
	class Software : public Vl53l0x::Base, public I2c::Software<sda_port, sda_pin, scl_port, scl_pin, signal_length>
	{
		public: // CONSTRUCTORS
			Software(uint8_t initial_address = VL53L0X_DEFAULT_ADDRESS);

		public: // DESTRUCTOR
			~Software(void);

		public: // GETTERS
			bool is_connected(void) override;

		public: // METHODS
			void init(void);

			void write_register_8bit (uint8_t register_address, uint8_t  data) override;
			void write_register_16bit(uint8_t register_address, uint16_t data) override;
			void write_register_8bit (uint8_t register_address, uint8_t* data, uint16_t count) override;

			uint8_t  read_register_8bit (uint8_t register_address) override;
			uint16_t read_register_16bit(uint8_t register_address) override;
			void     read_register_8bit (uint8_t register_address, uint8_t* data, uint16_t count) override;
	};

	// CONSTRUCTORS
	template<PORT sda_port, uint8_t sda_pin, PORT scl_port, uint8_t scl_pin,  uint16_t signal_length>
	Software<sda_port, sda_pin, scl_port, scl_pin, signal_length>::Software(uint8_t initial_address)
		: Base(initial_address)
	{
	}

	// DESTRUCTOR
	template<PORT sda_port, uint8_t sda_pin, PORT scl_port, uint8_t scl_pin,  uint16_t signal_length>
	Software<sda_port, sda_pin, scl_port, scl_pin, signal_length>::~Software(void)
	{
	}

	// GETTERS
	template<PORT sda_port, uint8_t sda_pin, PORT scl_port, uint8_t scl_pin,  uint16_t signal_length>
	bool Software<sda_port, sda_pin, scl_port, scl_pin, signal_length>::is_connected(void)
	{
		I2c::Software<sda_port, sda_pin, scl_port, scl_pin, signal_length>::start();
		I2c::Software<sda_port, sda_pin, scl_port, scl_pin, signal_length>::write(m_address << 1);
		I2c::Software<sda_port, sda_pin, scl_port, scl_pin, signal_length>::stop();

		return I2c::Software<sda_port, sda_pin, scl_port, scl_pin, signal_length>::ack();
	}

	// METHODS
	template<PORT sda_port, uint8_t sda_pin, PORT scl_port, uint8_t scl_pin,  uint16_t signal_length>
	void Software<sda_port, sda_pin, scl_port, scl_pin, signal_length>::init(void)
	{
		I2c::Software<sda_port, sda_pin, scl_port, scl_pin, signal_length>::init();
		Base::init();
	}

	template<PORT sda_port, uint8_t sda_pin, PORT scl_port, uint8_t scl_pin,  uint16_t signal_length>
	void Software<sda_port, sda_pin, scl_port, scl_pin, signal_length>::write_register_8bit(uint8_t register_address, uint8_t data)
	{
		I2c::Software<sda_port, sda_pin, scl_port, scl_pin, signal_length>::write_register_8bit(m_address, register_address, data);
	}
	template<PORT sda_port, uint8_t sda_pin, PORT scl_port, uint8_t scl_pin,  uint16_t signal_length>
	void Software<sda_port, sda_pin, scl_port, scl_pin, signal_length>::write_register_16bit(uint8_t register_address, uint16_t data)
	{
		I2c::Software<sda_port, sda_pin, scl_port, scl_pin, signal_length>::write_register_16bit(m_address, register_address, data);
	}
	template<PORT sda_port, uint8_t sda_pin, PORT scl_port, uint8_t scl_pin,  uint16_t signal_length>
	void Software<sda_port, sda_pin, scl_port, scl_pin, signal_length>::write_register_8bit(uint8_t register_address, uint8_t* data, uint16_t count)
	{
		I2c::Software<sda_port, sda_pin, scl_port, scl_pin, signal_length>::write_register_8bit(m_address, register_address, data, count);
	}
	//
	template<PORT sda_port, uint8_t sda_pin, PORT scl_port, uint8_t scl_pin,  uint16_t signal_length>
	uint8_t Software<sda_port, sda_pin, scl_port, scl_pin, signal_length>::read_register_8bit(uint8_t register_address)
	{
		return I2c::Software<sda_port, sda_pin, scl_port, scl_pin, signal_length>::read_register_8bit(m_address, register_address);
	}
	template<PORT sda_port, uint8_t sda_pin, PORT scl_port, uint8_t scl_pin,  uint16_t signal_length>
	uint16_t Software<sda_port, sda_pin, scl_port, scl_pin, signal_length>::read_register_16bit(uint8_t register_address)
	{
		return I2c::Software<sda_port, sda_pin, scl_port, scl_pin, signal_length>::read_register_16bit(m_address, register_address);
	}
	template<PORT sda_port, uint8_t sda_pin, PORT scl_port, uint8_t scl_pin,  uint16_t signal_length>
	void Software<sda_port, sda_pin, scl_port, scl_pin, signal_length>::read_register_8bit(uint8_t register_address, uint8_t* data, uint16_t count)
	{
		return I2c::Software<sda_port, sda_pin, scl_port, scl_pin, signal_length>::read_register_8bit(m_address, register_address, data, count);
	}
}

#endif
