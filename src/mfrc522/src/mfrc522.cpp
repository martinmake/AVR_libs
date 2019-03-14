#include <avr/io.h>
#define F_CPU 16000000
#include <util/delay.h>

#include <standard/standard.h>
#include <usart/usart.h>
#include <spi/spi.h>

#include "mfrc522.h"

Mfrc522::Mfrc522(Spi::Slave slave, Pin rst)
	: m_slave(slave)
{
	reset(rst);
	init();
}

Mfrc522::~Mfrc522()
{
}

void Mfrc522::reset(Pin rst)
{
	rst.dd.clear();
	if (!rst.pin.read()) {
		rst.dd.set();

		rst.port.clear();
		_delay_ms(2);
		rst.port.set();

		_delay_ms(100);
	} else
		Usart::send_str("\nERROR: Soft Reset NOT IMPLEMENTED\n");
}

void Mfrc522::init()
{
	// Default 0x00. Force a 100% ASK modulation independent of the ModGsPReg register setting
	write_reg(Mfrc522::REG::TxASKReg, 0x40);

	// When communicating with a PICC we need a timeout if something goes wrong.
	// f_timer = 13.56 MHz / (2*TPreScaler+1) where TPreScaler = [TPrescaler_Hi:TPrescaler_Lo].
	// TPrescaler_Hi are the four low bits in TModeReg. TPrescaler_Lo is TPrescalerReg.
	write_reg(REG::TModeReg,      0x80);	// TAuto=1; timer starts automatically at the end of the transmission in all communication modes at all speeds
	write_reg(REG::TPrescalerReg, 0xA9);	// TPreScaler = TModeReg[3..0]:TPrescalerReg, ie 0x0A9 = 169 => f_timer=40kHz, ie a timer period of 25μs.
	write_reg(REG::TReloadRegH,   0x03);	// Reload timer with 0x3E8 = 1000, ie 25ms before timeout.
	write_reg(REG::TReloadRegL,   0xE8);

	// Default 0x3F. Set the preset value for the CRC coprocessor for the CalcCRC command to 0x6363 (ISO 14443-3 part 6.2.4)
	write_reg(Mfrc522::REG::ModeReg, 0x3D);

	antenna_on();
}

void Mfrc522::antenna_on()
{
	uint8_t val = read_reg(Mfrc522::REG::TxControlReg);

	if ((val & 0b11) != 0b11)
		write_reg(Mfrc522::REG::TxControlReg, val | 0b11);
}

bool Mfrc522::is_card_present()
{
	uint8_t data_ATQA[2];
	uint8_t len_ATQA;

	reset_baud_rates();

	STATUS_CODE status_code = reqa(data_ATQA, &len_ATQA);
	return (status_code == STATUS_CODE::OK || status_code == STATUS_CODE::COLLISION);
}

Mfrc522::STATUS_CODE Mfrc522::reqa(uint8_t *data_ATQA,	///< The buffer to store the ATQA (Answer to request) in
		                   uint8_t *len_ATQA)	///< Buffer size, at least two bytes. Also number of bytes returned if STATUS_OK.
{
	return reqa_or_wupa(Picc::CMD::REQA, data_ATQA, len_ATQA);
}

Mfrc522::STATUS_CODE Mfrc522::wupa(uint8_t *data_ATQA,	///< The buffer to store the ATQA (Answer to request) in
		                   uint8_t *len_ATQA)	///< Buffer size, at least two bytes. Also number of bytes returned if STATUS_OK.
{
	return reqa_or_wupa(Picc::CMD::WUPA, data_ATQA, len_ATQA);
}

Mfrc522::STATUS_CODE Mfrc522::reqa_or_wupa(Mfrc522::Picc::CMD cmd,	///< The command to send - PICC_CMD_REQA or PICC_CMD_WUPA
		                           uint8_t *data_ATQA,   	///< The buffer to store the ATQA (Answer to request) in
		                           uint8_t *len_ATQA)    	///< Buffer size, at least two bytes. Also number of bytes returned if STATUS_OK.
{
	Mfrc522::STATUS_CODE status_code;
	uint8_t valid_bits = 7;

	// Argument check
	if (data_ATQA == nullptr || *len_ATQA < 2)
		return (STATUS_CODE::NO_ROOM);

	clear_mask_reg(REG::CollReg, (uint8_t) 0x80);

	status_code = transceive((uint8_t*) &cmd, 1, data_ATQA, len_ATQA, &valid_bits, 0, false);
	if (status_code != STATUS_CODE::OK)
		return status_code;
	if (*len_ATQA != 2 || valid_bits != 0)
		return STATUS_CODE::ERROR;
	return STATUS_CODE::OK;
}

Mfrc522::STATUS_CODE Mfrc522::transceive(uint8_t *data_out,	///< Pointer to the data to transfer to the FIFO.
		                         uint8_t  len_out,	///< Number of bytes to transfer to the FIFO.
		                         uint8_t *data_in,	///< nullptr or pointer to buffer if data should be read back after executing the command.
		                         uint8_t *len_in,	///< In: Max number of bytes to write to *backData. Out: The number of bytes returned.
		                         uint8_t *valid_bits,	///< In/Out: The number of valid bits in the last byte. 0 for 8 valid bits. Default nullptr.
		                         uint8_t  rx_align,	///< In: Defines the bit position in backData[0] for the first bit received. Default 0.
		                         bool     check_CRC)	///< In: True => The last two bytes of the response is assumed to be a CRC_A that must be validated.
{
	uint8_t wait_IRg = 0b00110000;
	return communicate_with_picc(Mfrc522::CMD::Transceive, wait_IRg, data_out, len_out, data_in, len_in, valid_bits, rx_align, check_CRC);
}

Mfrc522::STATUS_CODE Mfrc522::communicate_with_picc(Mfrc522::CMD cmd,	///< The command to execute. One of the PCD_Command enums.
                                                    uint8_t  wait_IRq,	///< The bits in the ComIrqReg register that signals successful completion of the command.
                                                    uint8_t *data_out,	///< Pointer to the data to transfer to the FIFO.
                                                    uint8_t  len_out,	///< Number of bytes to transfer to the FIFO.
                                                    uint8_t *data_in,	///< nullptr or pointer to buffer if data should be read back after executing the command.
                                                    uint8_t *len_in,	///< In: Max number of bytes to write to *backData. Out: The number of bytes returned.
                                                    uint8_t *valid_bits,///< In/Out: The number of valid bits in the last byte. 0 for 8 valid bits. Default nullptr.
                                                    uint8_t  rx_align,	///< In: Defines the bit position in backData[0] for the first bit received. Default 0.
                                                    bool     check_CRC)	///< In: True => The last two bytes of the response is assumed to be a CRC_A that must be validated.
{
	// Prepare values for BitFramingReg
	uint8_t tx_last_bits = valid_bits ? *valid_bits : 0;
	uint8_t bit_framing = (rx_align << 4) + tx_last_bits;	// rx_align = BitFramingReg[6..4]. tx_last_bits = BitFramingReg[2..0]

	write_reg(REG::CommandReg, (uint8_t) CMD::Idle);	// Stop any active command.
	write_reg(REG::ComIrqReg, 0x7F);			// Clear all seven interrupt request bits
	write_reg(REG::FIFOLevelReg, 0x80);			// FlushBuffer = 1, FIFO initialization
	write_reg(REG::FIFODataReg, len_out, data_out);		// Write sendData to the FIFO
	write_reg(REG::BitFramingReg, bit_framing);		// Bit adjustments
	write_reg(REG::CommandReg, (uint8_t) cmd);		// Execute the command
	if (cmd == CMD::Transceive)
		set_mask_reg(REG::BitFramingReg, 0x80);		// StartSend=1, transmission of data starts

	// Wait for the command to complete.
	// TODO check/modify for other architectures than Arduino Uno 16bit
	uint16_t i;
	for (i = 2000; i > 0; i--) {
		uint8_t reg = read_reg(REG::ComIrqReg);	// ComIrqReg[7..0] bits are: Set1 TxIRq RxIRq IdleIRq HiAlertIRq LoAlertIRq ErrIRq TimerIRq
		if (reg & wait_IRq)			// One of the interrupts that signal success has been set.
			break;
		if (reg & 0x01)				// Timer interrupt - nothing received in 25ms
			return STATUS_CODE::TIMEOUT;
	}
	// 35.7ms and nothing happend. Communication with the MFRC522 might be down.
	if (i == 0)
		return STATUS_CODE::TIMEOUT;

	// Stop now if any errors except collisions were detected.
	uint8_t error_reg_val = read_reg(REG::ErrorReg);	// ErrorReg[7..0] bits are: WrErr TempErr reserved BufferOvfl CollErr CRCErr ParityErr ProtocolErr
	if (error_reg_val & 0x13)                       	// BufferOvfl ParityErr ProtocolErr
		return STATUS_CODE::ERROR;

	// If the caller wants data back, get it from the MFRC522.
	if (data_in && len_in) {
		uint8_t level = read_reg(REG::FIFOLevelReg);      	// Number of bytes in the FIFO
		if (level > *len_in)
			return STATUS_CODE::NO_ROOM;
		*len_in = level;					// Number of bytes returned
		read_reg(REG::FIFODataReg, level, data_in, rx_align);	// Get received data from FIFO
		if (valid_bits)
			*valid_bits = read_reg(REG::ControlReg) & 0x07;	// RxLastBits[2:0] indicates the number of valid bits in the last received byte. If this value is 000b, the whole byte is valid.
	}

	// Tell about collisions
	if (error_reg_val & 0x08)	// CollErr
		return STATUS_CODE::COLLISION;

	// TODO: Perform CRC_A validation if requested.
//	if (data_in && len_in && check_CRC) {
//		// In this case a MIFARE Classic NAK is not OK.
//		if (*len_in == 1 && *valid_bits == 4)
//			return STATUS_CODE::MIFARE_NACK;
//		// We need at least the CRC_A value and all 8 bits of the last byte must be received.
//		if (*len_in < 2 || *valid_bits != 0)
//			return STATUS_CODE::CRC_WRONG;
//		// Verify CRC_A - do our own calculation and store the control in controlBuffer.
//		uint8_t control_buffer[2];
//		Mfrc522::STATUS_CODE status = calculate_CRC(&data_in[0], *len_in - 2, &control_buffer[0]);
//		if (status != STATUS_CODE::OK)
//			return status;
//		if ((data_in[*len_in - 2] != control_buffer[0]) || (data_in[*len_in - 1] != control_buffer[1]))
//			return STATUS_CODE::CRC_WRONG;
//	}

	return STATUS_CODE::OK;
}

const char* get_status_code_name(Mfrc522::STATUS_CODE status_code)
{
	switch (status_code) {
		case Mfrc522::STATUS_CODE::OK:             return "OK";
		case Mfrc522::STATUS_CODE::ERROR:          return "ERROR";
		case Mfrc522::STATUS_CODE::COLLISION:      return "COLLISION";
		case Mfrc522::STATUS_CODE::TIMEOUT:        return "TIMEOUT";
		case Mfrc522::STATUS_CODE::NO_ROOM:        return "NO_ROOM";
		case Mfrc522::STATUS_CODE::INTERNAL_ERROR: return "INTERNAL_ERROR";
		case Mfrc522::STATUS_CODE::INVALID:        return "INVALID";
		case Mfrc522::STATUS_CODE::CRC_WRONG:      return "CRC_WRONG";
		case Mfrc522::STATUS_CODE::MIFARE_NACK:    return "MIFARE_NACK";
		default:                                   return "UNKNOWN STATUS CODE";
	}
}

void Mfrc522::reset_baud_rates()
{
	// Reset baud rates
	write_reg(REG::TxModeReg, 0x00);
	write_reg(REG::RxModeReg, 0x00);
	// Reset ModWidthReg
	write_reg(REG::ModWidthReg, 0x26);
}

uint8_t Mfrc522::read_reg(REG reg)
{
	uint8_t val;

	m_slave.select();

	Spi::send((1 << 7) | static_cast<uint8_t>(reg));
	val = Spi::send(0);

	m_slave.unselect();

	return val;
}

void Mfrc522::read_reg(Mfrc522::REG reg,   	///< The register to read from. One of the PCD_Register enums.
                       uint8_t  len,    	///< The number of bytes to read
                       uint8_t *data,   	///< Byte array to store the values in.
                       uint8_t  rx_align)	///< Only bit positions rxAlign..7 in values[0] are updated.
{
	if (len == 0)
		return;

	uint8_t addr = 0x80 | (uint8_t) reg;
	uint8_t index = 0;

	len--;	// One read is performed outside of the loop

	m_slave.select();

	Spi::send(addr);	// Tell MFRC522 which address we want to read
	if (rx_align) {		// Only update bit positions rxAlign..7 in values[0]
		// Create bit mask for bit positions rxAlign..7
		uint8_t mask = (0xFF << rx_align) & 0xFF;
		uint8_t val = Spi::send(addr);
		// Apply mask to both current value of values[0] and the new data in value.
		data[0] = val & mask;
		index++;
	}
	while (index < len) {
		data[index] = Spi::send(addr);
		index++;
	}
	data[index] = Spi::send(0);

	m_slave.unselect();
}

/* TODO: Implement data dump
bool Mfrc522::read_card() {
	Mfrc522::STATUS_CODE result = select_picc(&uid);
	return (result == STATUS_CODE::OK);
}
*/

void Mfrc522::write_reg(REG reg, uint8_t data)
{
	m_slave.select();

	Spi::send(static_cast<uint8_t>(reg));
	Spi::send(data);

	m_slave.unselect();
}

void Mfrc522::write_reg(REG reg, uint8_t len, uint8_t* data)
{
	m_slave.select();

	Spi::send(static_cast<uint8_t>(reg));
	for (uint8_t i = 0; i < len; i++) {
		Spi::send(data[i]);
	}

	m_slave.unselect();
}

uint8_t Mfrc522::read_reg(uint8_t reg)
{
	uint8_t val;

	m_slave.select();

	Spi::send((1 << 7) | (reg << 1));
	val = Spi::send(0);

	m_slave.unselect();

	return val;
}

void Mfrc522::write_reg(uint8_t reg, uint8_t data)
{
	m_slave.select();

	Spi::send(reg << 1);
	Spi::send(data);

	m_slave.unselect();
}

void Mfrc522::clear_mask_reg(REG reg, uint8_t mask)
{
	uint8_t val = read_reg(reg);
	write_reg(reg, val & ~mask);
}

void Mfrc522::set_mask_reg(REG reg, uint8_t mask)
{
	uint8_t val = read_reg(reg);
	write_reg(reg, val | mask);
}
