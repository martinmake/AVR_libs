#ifndef _VL53L0X_BASE_H_
#define _VL53L0X_BASE_H_

#include <util.h>
#include <i2c/base.h>

#define VL53L0X_DEFAULT_ADDRESS 0b0101001

namespace Vl53l0x
{
	class Base
	{
		public: // CONSTRUCTORS
			Base(uint8_t initial_address = VL53L0X_DEFAULT_ADDRESS);

		public: // DESTRUCTOR
			virtual ~Base(void);

		public: // STATIC VARIABLES
			const uint16_t TIMED_OUT = (uint16_t) -1;

		public: // GETTERS
			        uint8_t  address     (void) const;
			        uint16_t range       (void);
			virtual bool     is_connected(void) = 0;
		public: // SETTERS
			void address(uint8_t new_address);

		public: // METHODS
			void init(void);

		public: // VIRTUAL METHODS
			virtual void write_register_8bit (uint8_t register_address, uint8_t  data) = 0;
			virtual void write_register_16bit(uint8_t register_address, uint16_t data) = 0;
			virtual void write_register_8bit (uint8_t register_address, uint8_t* data, uint16_t count) = 0;

			virtual uint8_t  read_register_8bit (uint8_t register_address) = 0;
			virtual uint16_t read_register_16bit(uint8_t register_address) = 0;
			virtual void     read_register_8bit (uint8_t register_address, uint8_t* data, uint16_t count) = 0;

		protected:
			uint8_t m_address;
		private:
			uint8_t m_stop_variable;

		private:
			enum REGISTER : uint8_t
			{
				SYSRANGE_START                              = 0x00,

				SYSTEM_THRESH_HIGH                          = 0x0C,
				SYSTEM_THRESH_LOW                           = 0x0E,

				STOP_VARIABLE                               = 0x91,

				SYSTEM_SEQUENCE_CONFIG                      = 0x01,
				SYSTEM_RANGE_CONFIG                         = 0x09,
				SYSTEM_INTERMEASUREMENT_PERIOD              = 0x04,

				SYSTEM_INTERRUPT_CONFIG_GPIO                = 0x0A,

				GPIO_HV_MUX_ACTIVE_HIGH                     = 0x84,

				SYSTEM_INTERRUPT_CLEAR                      = 0x0B,

				RESULT_INTERRUPT_STATUS                     = 0x13,
				RESULT_RANGE_STATUS                         = 0x14,

				RESULT_CORE_AMBIENT_WINDOW_EVENTS_RTN       = 0xBC,
				RESULT_CORE_RANGING_TOTAL_EVENTS_RTN        = 0xC0,
				RESULT_CORE_AMBIENT_WINDOW_EVENTS_REF       = 0xD0,
				RESULT_CORE_RANGING_TOTAL_EVENTS_REF        = 0xD4,
				RESULT_PEAK_SIGNAL_RATE_REF                 = 0xB6,

				ALGO_PART_TO_PART_RANGE_OFFSET_MM           = 0x28,

				I2C_SLAVE_DEVICE_ADDRESS                    = 0x8A,

				MSRC_CONFIG_CONTROL                         = 0x60,

				PRE_RANGE_CONFIG_MIN_SNR                    = 0x27,
				PRE_RANGE_CONFIG_VALID_PHASE_LOW            = 0x56,
				PRE_RANGE_CONFIG_VALID_PHASE_HIGH           = 0x57,
				PRE_RANGE_MIN_COUNT_RATE_RTN_LIMIT          = 0x64,

				FINAL_RANGE_CONFIG_MIN_SNR                  = 0x67,
				FINAL_RANGE_CONFIG_VALID_PHASE_LOW          = 0x47,
				FINAL_RANGE_CONFIG_VALID_PHASE_HIGH         = 0x48,
				FINAL_RANGE_CONFIG_MIN_COUNT_RATE_RTN_LIMIT = 0x44,

				PRE_RANGE_CONFIG_SIGMA_THRESH_HI            = 0x61,
				PRE_RANGE_CONFIG_SIGMA_THRESH_LO            = 0x62,

				PRE_RANGE_CONFIG_VCSEL_PERIOD               = 0x50,
				PRE_RANGE_CONFIG_TIMEOUT_MACROP_HI          = 0x51,
				PRE_RANGE_CONFIG_TIMEOUT_MACROP_LO          = 0x52,

				SYSTEM_HISTOGRAM_BIN                        = 0x81,
				HISTOGRAM_CONFIG_INITIAL_PHASE_SELECT       = 0x33,
				HISTOGRAM_CONFIG_READOUT_CTRL               = 0x55,

				FINAL_RANGE_CONFIG_VCSEL_PERIOD             = 0x70,
				FINAL_RANGE_CONFIG_TIMEOUT_MACROP_HI        = 0x71,
				FINAL_RANGE_CONFIG_TIMEOUT_MACROP_LO        = 0x72,
				CROSSTALK_COMPENSATION_PEAK_RATE_MCPS       = 0x20,

				MSRC_CONFIG_TIMEOUT_MACROP                  = 0x46,

				SOFT_RESET_GO2_SOFT_RESET_N                 = 0xBF,
				IDENTIFICATION_MODEL_ID                     = 0xC0,
				IDENTIFICATION_REVISION_ID                  = 0xC2,

				OSC_CALIBRATE_VAL                           = 0xF8,

				GLOBAL_CONFIG_VCSEL_WIDTH                   = 0x32,
				GLOBAL_CONFIG_SPAD_ENABLES_REF_0            = 0xB0,
				GLOBAL_CONFIG_SPAD_ENABLES_REF_1            = 0xB1,
				GLOBAL_CONFIG_SPAD_ENABLES_REF_2            = 0xB2,
				GLOBAL_CONFIG_SPAD_ENABLES_REF_3            = 0xB3,
				GLOBAL_CONFIG_SPAD_ENABLES_REF_4            = 0xB4,
				GLOBAL_CONFIG_SPAD_ENABLES_REF_5            = 0xB5,

				GLOBAL_CONFIG_REF_EN_START_SELECT           = 0xB6,
				DYNAMIC_SPAD_NUM_REQUESTED_REF_SPAD         = 0x4E,
				DYNAMIC_SPAD_REF_EN_START_OFFSET            = 0x4F,
				POWER_MANAGEMENT_GO1_POWER_FORCE            = 0x80,

				VHV_CONFIG_PAD_SCL_SDA__EXTSUP_HV           = 0x89,

				ALGO_PHASECAL_LIM                           = 0x30,
				ALGO_PHASECAL_CONFIG_TIMEOUT                = 0x30,
			};
	};

	// GETTERS
	inline uint8_t Base::address(void) const { return m_address; }
	// SETTERS
	inline void Base::address(uint8_t new_address) { m_address = new_address; } // TODO: FINISH IMPLEMENTATION
}

#endif
