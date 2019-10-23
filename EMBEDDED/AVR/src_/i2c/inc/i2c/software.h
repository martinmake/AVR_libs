#ifndef _I2C_SOFTWATE_H_
#define _I2C_SOFTWATE_H_

#include <gpio.h>

#include "i2c/base.h"

#define I2C_SOFTWARE_DEFAULT_DELAY 5

namespace I2c
{
	template<PORT sda_port, uint8_t sda_pin, PORT scl_port, uint8_t scl_pin, uint16_t signal_length = I2C_SOFTWARE_DEFAULT_DELAY>
	class Software : public Base
	{
		public: // CONSTRUCTORS
			Software(void);

		public: // DESTRUCTOR
			~Software(void);

		public: // METHODS
			void init(void) override;

			void start(void) override;
			void stop (void) override;

			void    write(uint8_t data) override;
			uint8_t read (bool    ack ) override;

			void set_sda(bool state);
			void set_scl(bool state);

			void delay(void);

		private:
			Gpio<sda_port, sda_pin> m_sda;
			Gpio<scl_port, scl_pin> m_scl;
	};

	template<PORT sda_port, uint8_t sda_pin, PORT scl_port, uint8_t scl_pin,  uint16_t signal_length>
	Software<sda_port, sda_pin, scl_port, scl_pin, signal_length>::Software(void)
	{
	}
	template<PORT sda_port, uint8_t sda_pin, PORT scl_port, uint8_t scl_pin,  uint16_t signal_length>
	Software<sda_port, sda_pin, scl_port, scl_pin, signal_length>::~Software(void)
	{
	}

	template<PORT sda_port, uint8_t sda_pin, PORT scl_port, uint8_t scl_pin,  uint16_t signal_length>
	void Software<sda_port, sda_pin, scl_port, scl_pin, signal_length>::init(void)
	{
		set_sda(HIGH);
		set_scl(HIGH);
	}

	template<PORT sda_port, uint8_t sda_pin, PORT scl_port, uint8_t scl_pin,  uint16_t signal_length>
	void Software<sda_port, sda_pin, scl_port, scl_pin, signal_length>::start(void)
	{
		set_sda(HIGH);
		set_scl(HIGH);
		delay();

		set_sda(LOW);
		delay();
	}
	template<PORT sda_port, uint8_t sda_pin, PORT scl_port, uint8_t scl_pin,  uint16_t signal_length>
	void Software<sda_port, sda_pin, scl_port, scl_pin, signal_length>::stop(void)
	{
		set_sda(LOW);
		set_scl(HIGH);
		delay();

		set_sda(HIGH);
		delay();
	}

	template<PORT sda_port, uint8_t sda_pin, PORT scl_port, uint8_t scl_pin,  uint16_t signal_length>
	void Software<sda_port, sda_pin, scl_port, scl_pin, signal_length>::write(uint8_t data)
	{
		for (uint8_t mask = 1 << 7; mask; mask >>= 1)
		{
			set_scl(LOW);
			delay();

			if (data & mask) set_sda(HIGH);
			else             set_sda(LOW);
			delay();

			set_scl(HIGH);
			delay();
		}

		set_scl(LOW);
		delay();
		set_sda(HIGH);
		delay();
		set_scl(HIGH);
		delay();
		m_ack = !m_sda;
		set_scl(LOW);
		delay();
	}
	template<PORT sda_port, uint8_t sda_pin, PORT scl_port, uint8_t scl_pin,  uint16_t signal_length>
	uint8_t Software<sda_port, sda_pin, scl_port, scl_pin, signal_length>::read(bool ack)
	{
		uint8_t data = 0x00;
		for (uint8_t mask = 1 << 7; mask; mask >>= 1)
		{
			set_scl(LOW);
			delay();
			set_scl(HIGH);
			delay();

			if (m_sda) data |= mask;
		}

		set_scl(LOW);
		delay();
		if (ack) set_sda(LOW);
		else     set_sda(HIGH);
		set_scl(HIGH);
		delay();
		set_scl(LOW);
		delay();
		set_sda(HIGH);

		return data;
	}

	template<PORT sda_port, uint8_t sda_pin, PORT scl_port, uint8_t scl_pin,  uint16_t signal_length>
	void Software<sda_port, sda_pin, scl_port, scl_pin, signal_length>::set_sda(bool state)
	{
		switch (state)
		{
			case HIGH: m_sda.make_input (); m_sda.pull_up(); break;
			case LOW:  m_sda.make_output(); m_sda = LOW;     break;
		}
	}
	template<PORT sda_port, uint8_t sda_pin, PORT scl_port, uint8_t scl_pin,  uint16_t signal_length>
	void Software<sda_port, sda_pin, scl_port, scl_pin, signal_length>::set_scl(bool state)
	{
		switch (state)
		{
			case HIGH: m_scl.make_input (); m_scl.pull_up(); while (m_scl.is_low()) {} break;
			case LOW:  m_scl.make_output(); m_scl = LOW; break;
		}
	}

	template<PORT sda_port, uint8_t sda_pin, PORT scl_port, uint8_t scl_pin,  uint16_t signal_length>
	void Software<sda_port, sda_pin, scl_port, scl_pin, signal_length>::delay(void)
	{
		for (uint16_t countdown = signal_length; countdown; countdown--) asm("NOP");
	}
}

#endif
