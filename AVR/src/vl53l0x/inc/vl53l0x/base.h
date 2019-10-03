#ifndef _VL53L0X_BASE_H_
#define _VL53L0X_BASE_H_

#include <util.h>

#define VL53L0X_DEFAULT_ADDRESS 0b0101001

namespace Vl53l0x
{
	class Base
	{
		public: // CONSTRUCTORS
			Base(uint8_t initial_address = VL53L0X_DEFAULT_ADDRESS);

		public: // DESTRUCTOR
			virtual ~Base(void);

		public: // GETTERS
			uint8_t address(void) const;
		public: // SETTERS
			void address(uint8_t new_address);

		public: // METHODS
			void init(void);

		public: // VIRTUAL METHODS
			virtual void write_register_8bit (uint8_t register_address, uint8_t  data) = 0;
			virtual void write_register_16bit(uint8_t register_address, uint16_t data) = 0;

			virtual uint8_t  read_register_8bit (uint8_t register_address) = 0;
			virtual uint16_t read_register_16bit(uint8_t register_address) = 0;

		protected:
			uint8_t m_address;
	};

	// GETTERS
	inline uint8_t Base::address(void) const { return m_address; }
	// SETTERS
	inline void Base::address(uint8_t new_address) { m_address = new_address; }
}

#endif
