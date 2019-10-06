#include <system_clock.h>

#include "vl53l0x/base.h"

namespace Vl53l0x
{
	// CONSTRUCTORS
	Base::Base(uint8_t initial_address)
		: m_address(initial_address)
	{
	}

	// DESTRUCTOR
	Base::~Base(void)
	{
	}

	// METHODS
	void Base::init(void)
	{
		write_register_8bit(VHV_CONFIG_PAD_SCL_SDA__EXTSUP_HV, read_register_8bit(VHV_CONFIG_PAD_SCL_SDA__EXTSUP_HV | 0x01));

		write_register_8bit(0x88, 0x00); // SET I2C STANDARD MODE

		write_register_8bit(MSRC_CONFIG_CONTROL, read_register_8bit(MSRC_CONFIG_CONTROL) | 0x12);

		write_register_16bit(FINAL_RANGE_CONFIG_MIN_COUNT_RATE_RTN_LIMIT, 0.25 * (1 << 7));

		write_register_8bit(SYSTEM_SEQUENCE_CONFIG, 0xFF);

		uint8_t spad_count;
		bool spad_type_is_aperture;
		uint8_t tmp;

		write_register_8bit(0x80, 0x01);
		write_register_8bit(0xFF, 0x01);
		write_register_8bit(0x00, 0x00);

		write_register_8bit(0xFF, 0x06);
		write_register_8bit(0x83, read_register_8bit(0x83) | 0x04);
		write_register_8bit(0xFF, 0x07);
		write_register_8bit(0x81, 0x01);

		write_register_8bit(0x80, 0x01);

		write_register_8bit(0x94, 0x6b);
		write_register_8bit(0x83, 0x00);
		system_clock.timeout(500);
		while (read_register_8bit(0x83) == 0x00)
			if (system_clock.has_timed_out()) return;
		write_register_8bit(0x83, 0x01);
		tmp = read_register_8bit(0x92);

		spad_count = tmp & 0x7f;
		spad_type_is_aperture = (tmp >> 7) & 0x01;

		write_register_8bit(0x81, 0x00);
		write_register_8bit(0xFF, 0x06);
		write_register_8bit(0x83, read_register_8bit(0x83)  & ~0x04);
		write_register_8bit(0xFF, 0x01);
		write_register_8bit(0x00, 0x01);

		write_register_8bit(0xFF, 0x00);
		write_register_8bit(0x80, 0x00);

		// The SPAD map (RefGoodSpadMap) is read by VL53L0X_get_info_from_device() in
		// the API, but the same data seems to be more easily readable from
		// GLOBAL_CONFIG_SPAD_ENABLES_REF_0 through _6, so read it from there
		uint8_t ref_spad_map[6];
		read_register_8bit(GLOBAL_CONFIG_SPAD_ENABLES_REF_0, ref_spad_map, 6);

		// -- VL53L0X_set_reference_spads() begin (assume NVM values are valid)

		write_register_8bit(0xFF, 0x01);
		write_register_8bit(DYNAMIC_SPAD_REF_EN_START_OFFSET, 0x00);
		write_register_8bit(DYNAMIC_SPAD_NUM_REQUESTED_REF_SPAD, 0x2C);
		write_register_8bit(0xFF, 0x00);
		write_register_8bit(GLOBAL_CONFIG_REF_EN_START_SELECT, 0xB4);

		uint8_t first_spad_to_enable = spad_type_is_aperture ? 12 : 0; // 12 is the first aperture spad
		uint8_t spads_enabled = 0;

		for (uint8_t i = 0; i < 48; i++)
		{
			if (i < first_spad_to_enable || spads_enabled == spad_count)
			{
				// This bit is lower than the first one that should be enabled, or
				// (reference_spad_count) bits have already been enabled, so zero this bit
				ref_spad_map[i / 8] &= ~(1 << (i % 8));
			}
			else if ((ref_spad_map[i / 8] >> (i % 8)) & 0x1)
			{
				spads_enabled++;
			}
		}

		write_register_8bit(GLOBAL_CONFIG_SPAD_ENABLES_REF_0, ref_spad_map, 6);

		write_register_8bit(0xFF, 0x01);
		write_register_8bit(0x00, 0x00);

		write_register_8bit(0xFF, 0x00);
		write_register_8bit(0x09, 0x00);
		write_register_8bit(0x10, 0x00);
		write_register_8bit(0x11, 0x00);

		write_register_8bit(0x24, 0x01);
		write_register_8bit(0x25, 0xFF);
		write_register_8bit(0x75, 0x00);

		write_register_8bit(0xFF, 0x01);
		write_register_8bit(0x4E, 0x2C);
		write_register_8bit(0x48, 0x00);
		write_register_8bit(0x30, 0x20);

		write_register_8bit(0xFF, 0x00);
		write_register_8bit(0x30, 0x09);
		write_register_8bit(0x54, 0x00);
		write_register_8bit(0x31, 0x04);
		write_register_8bit(0x32, 0x03);
		write_register_8bit(0x40, 0x83);
		write_register_8bit(0x46, 0x25);
		write_register_8bit(0x60, 0x00);
		write_register_8bit(0x27, 0x00);
		write_register_8bit(0x50, 0x06);
		write_register_8bit(0x51, 0x00);
		write_register_8bit(0x52, 0x96);
		write_register_8bit(0x56, 0x08);
		write_register_8bit(0x57, 0x30);
		write_register_8bit(0x61, 0x00);
		write_register_8bit(0x62, 0x00);
		write_register_8bit(0x64, 0x00);
		write_register_8bit(0x65, 0x00);
		write_register_8bit(0x66, 0xA0);

		write_register_8bit(0xFF, 0x01);
		write_register_8bit(0x22, 0x32);
		write_register_8bit(0x47, 0x14);
		write_register_8bit(0x49, 0xFF);
		write_register_8bit(0x4A, 0x00);

		write_register_8bit(0xFF, 0x00);
		write_register_8bit(0x7A, 0x0A);
		write_register_8bit(0x7B, 0x00);
		write_register_8bit(0x78, 0x21);

		write_register_8bit(0xFF, 0x01);
		write_register_8bit(0x23, 0x34);
		write_register_8bit(0x42, 0x00);
		write_register_8bit(0x44, 0xFF);
		write_register_8bit(0x45, 0x26);
		write_register_8bit(0x46, 0x05);
		write_register_8bit(0x40, 0x40);
		write_register_8bit(0x0E, 0x06);
		write_register_8bit(0x20, 0x1A);
		write_register_8bit(0x43, 0x40);

		write_register_8bit(0xFF, 0x00);
		write_register_8bit(0x34, 0x03);
		write_register_8bit(0x35, 0x44);

		write_register_8bit(0xFF, 0x01);
		write_register_8bit(0x31, 0x04);
		write_register_8bit(0x4B, 0x09);
		write_register_8bit(0x4C, 0x05);
		write_register_8bit(0x4D, 0x04);

		write_register_8bit(0xFF, 0x00);
		write_register_8bit(0x44, 0x00);
		write_register_8bit(0x45, 0x20);
		write_register_8bit(0x47, 0x08);
		write_register_8bit(0x48, 0x28);
		write_register_8bit(0x67, 0x00);
		write_register_8bit(0x70, 0x04);
		write_register_8bit(0x71, 0x01);
		write_register_8bit(0x72, 0xFE);
		write_register_8bit(0x76, 0x00);
		write_register_8bit(0x77, 0x00);

		write_register_8bit(0xFF, 0x01);
		write_register_8bit(0x0D, 0x01);

		write_register_8bit(0xFF, 0x00);
		write_register_8bit(0x80, 0x01);
		write_register_8bit(0x01, 0xF8);

		write_register_8bit(0xFF, 0x01);
		write_register_8bit(0x8E, 0x01);
		write_register_8bit(0x00, 0x01);
		write_register_8bit(0xFF, 0x00);
		write_register_8bit(0x80, 0x00);

		write_register_8bit(SYSTEM_INTERRUPT_CONFIG_GPIO, 0x04);
		write_register_8bit(GPIO_HV_MUX_ACTIVE_HIGH, read_register_8bit(GPIO_HV_MUX_ACTIVE_HIGH) & ~0x10); // active low
		write_register_8bit(SYSTEM_INTERRUPT_CLEAR, 0x01);

		write_register_8bit(SYSTEM_SEQUENCE_CONFIG, 0xE8);

		write_register_8bit(SYSTEM_SEQUENCE_CONFIG, 0x01);
		write_register_8bit(SYSRANGE_START, 0x01 | 0x40);
		system_clock.timeout(500);
		while ((read_register_8bit(RESULT_INTERRUPT_STATUS) & 0x07) == 0)
			if (system_clock.has_timed_out()) return;
		write_register_8bit(SYSTEM_INTERRUPT_CLEAR, 0x01);
		write_register_8bit(SYSRANGE_START, 0x00);

		write_register_8bit(SYSTEM_SEQUENCE_CONFIG, 0x02);
		write_register_8bit(SYSRANGE_START, 0x01);
		system_clock.timeout(500);
		while ((read_register_8bit(RESULT_INTERRUPT_STATUS) & 0x07) == 0)
			if (system_clock.has_timed_out()) return;
		write_register_8bit(SYSTEM_INTERRUPT_CLEAR, 0x01);
		write_register_8bit(SYSRANGE_START, 0x00);

		write_register_8bit(SYSTEM_SEQUENCE_CONFIG, 0xE8);

		write_register_16bit(FINAL_RANGE_CONFIG_TIMEOUT_MACROP_HI, 660);
	}

	uint16_t Base::range(void)
	{
		uint16_t current_range = 0;
		while (!current_range)
		{
			write_register_8bit(SYSRANGE_START, 0x01);
			system_clock.timeout(700);
			while (read_register_8bit(SYSRANGE_START) & 0x01)
				if (system_clock.has_timed_out()) return TIMED_OUT;

			system_clock.timeout(700);
			while ((read_register_8bit(RESULT_INTERRUPT_STATUS) & 0x07) == 0)
				if (system_clock.has_timed_out()) return TIMED_OUT;

			current_range = read_register_16bit(RESULT_RANGE_STATUS + 10);
			write_register_8bit(SYSTEM_INTERRUPT_CLEAR, 0x01);
		}

		return current_range;
	}
}
