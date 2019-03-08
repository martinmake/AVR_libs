#ifndef _MFRC522_H_
#define _MFRC522_H_

#ifndef _SPI_H_
static_assert(0, "DEPENDENCY NOT MET! (SPI)");
#endif

class Mfrc522
{
	public:
		typedef enum class reg : uint8_t {
			// Page 0: Command and status
			//                 0x00		// reserved for future use
			CommandReg        = 0x01 << 1,	// starts and stops command execution
			ComIEnReg         = 0x02 << 1,	// enable and disable interrupt request control bits
			DivIEnReg         = 0x03 << 1,	// enable and disable interrupt request control bits
			ComIrqReg         = 0x04 << 1,	// interrupt request bits
			DivIrqReg         = 0x05 << 1,	// interrupt request bits
			ErrorReg          = 0x06 << 1,	// error bits showing the error status of the last command executed
			Status1Reg        = 0x07 << 1,	// communication status bits
			Status2Reg        = 0x08 << 1,	// receiver and transmitter status bits
			FIFODataReg       = 0x09 << 1,	// input and output of 64 uint8_t FIFO buffer
			FIFOLevelReg      = 0x0A << 1,	// number of uint8_ts stored in the FIFO buffer
			WaterLevelReg     = 0x0B << 1,	// level for FIFO underflow and overflow warning
			ControlReg        = 0x0C << 1,	// miscellaneous control registers
			BitFramingReg     = 0x0D << 1,	// adjustments for bit-oriented frames
			CollReg           = 0x0E << 1,	// bit position of the first bit-collision detected on the RF interface
			//                  0x0F		// reserved for future use

			// Page 1: Command
			//                  0x10		// reserved for future use
			ModeReg	          = 0x11 << 1,	// defines general modes for transmitting and receiving
			TxModeReg         = 0x12 << 1,	// defines transmission data rate and framing
			RxModeReg         = 0x13 << 1,	// defines reception data rate and framing
			TxControlReg      = 0x14 << 1,	// controls the logical behavior of the antenna driver pins TX1 and TX2
			TxASKReg          = 0x15 << 1,	// controls the setting of the transmission modulation
			TxSelReg          = 0x16 << 1,	// selects the internal sources for the antenna driver
			RxSelReg          = 0x17 << 1,	// selects internal receiver settings
			RxThresholdReg    = 0x18 << 1,	// selects thresholds for the bit decoder
			DemodReg          = 0x19 << 1,	// defines demodulator settings
			//                  0x1A		// reserved for future use
			//                  0x1B		// reserved for future use
			MfTxReg           = 0x1C << 1,	// controls some MIFARE communication transmit parameters
			MfRxReg	          = 0x1D << 1,	// controls some MIFARE communication receive parameters
			//                  0x1E		// reserved for future use
			SerialSpeedReg    = 0x1F << 1,	// selects the speed of the serial UART interface

			// Page 2: Configuration
			//                  0x20	// reserved for future use
			CRCResultRegH     = 0x21 << 1,	// shows the MSB and LSB values of the CRC calculation
			CRCResultRegL     = 0x22 << 1,
			//                  0x23	// reserved for future use
			ModWidthReg       = 0x24 << 1,	// controls the ModWidth setting?
			//                  0x25	// reserved for future use
			RFCfgReg          = 0x26 << 1,	// configures the receiver gain
			GsNReg	          = 0x27 << 1,	// selects the conductance of the antenna driver pins TX1 and TX2 for modulation
			CWGsPReg          = 0x28 << 1,	// defines the conductance of the p-driver output during periods of no modulation
			ModGsPReg         = 0x29 << 1,	// defines the conductance of the p-driver output during periods of modulation
			TModeReg          = 0x2A << 1,	// defines settings for the internal timer
			TPrescalerReg     = 0x2B << 1,	// the lower 8 bits of the TPrescaler value. The 4 high bits are in TModeReg.
			TReloadRegH       = 0x2C << 1,	// defines the 16-bit timer reload value
			TReloadRegL       = 0x2D << 1,
			TCounterValueRegH = 0x2E << 1,	// shows the 16-bit timer value
			TCounterValueRegL = 0x2F << 1,

			// Page 3: Test Registers
			//                  0x30		// reserved for future use
			TestSel1Reg       = 0x31 << 1,	// general test signal configuration
			TestSel2Reg       = 0x32 << 1,	// general test signal configuration
			TestPinEnReg      = 0x33 << 1,	// enables pin output driver on pins D1 to D7
			TestPinValueReg   = 0x34 << 1,	// defines the values for D1 to D7 when it is used as an I/O bus
			TestBusReg        = 0x35 << 1,	// shows the status of the internal test bus
			AutoTestReg       = 0x36 << 1,	// controls the digital self-test
			VersionReg        = 0x37 << 1,	// shows the software version
			AnalogTestReg     = 0x38 << 1,	// controls the pins AUX1 and AUX2
			TestDAC1Reg       = 0x39 << 1,	// defines the test value for TestDAC1
			TestDAC2Reg       = 0x3A << 1,	// defines the test value for TestDAC2
			TestADCReg        = 0x3B << 1	// shows the value of ADC I and Q channels
			//                  0x3C		// reserved for production tests
			//                  0x3D		// reserved for production tests
			//                  0x3E		// reserved for production tests
			//                  0x3F		// reserved for production tests
		} REG;

		// MFRC522 commands. Described in chapter 10 of the datasheet.
		typedef enum class cmd : uint8_t {
			Idle             = 0x00,	// no action, cancels current command execution
			Mem              = 0x01,	// stores 25 uint8_ts into the internal buffer
			GenerateRandomID = 0x02,	// generates a 10-uint8_t random ID number
			CalcCRC          = 0x03,	// activates the CRC coprocessor or performs a self-test
			Transmit         = 0x04,	// transmits data from the FIFO buffer
			NoCmdChange      = 0x07,	// no command change, can be used to modify the CommandReg register bits without affecting the command, for example, the PowerDown bit
			Receive          = 0x08,	// activates the receiver circuits
			Transceive       = 0x0C,	// transmits data from FIFO buffer to antenna and automatically activates the receiver after transmission
			MFAuthent        = 0x0E,	// performs the MIFARE standard authentication as a reader
			SoftReset        = 0x0F		// resets the MFRC522
		} CMD;

		// MFRC522 RxGain[2:0] masks, defines the receiver's signal voltage gain factor (on the PCD).
		// Described in 9.3.3.6 / table 98 of the datasheet at http://www.nxp.com/documents/data_sheet/MFRC522.pdf
		typedef enum class rx_gain : uint8_t {
			RxGain_18dB   = 0x00 << 4,	// 000b - 18 dB, minimum
			RxGain_23dB   = 0x01 << 4,	// 001b - 23 dB
			RxGain_18dB_2 = 0x02 << 4,	// 010b - 18 dB, it seems 010b is a duplicate for 000b
			RxGain_23dB_2 = 0x03 << 4,	// 011b - 23 dB, it seems 011b is a duplicate for 001b
			RxGain_33dB   = 0x04 << 4,	// 100b - 33 dB, average, and typical default
			RxGain_38dB   = 0x05 << 4,	// 101b - 38 dB
			RxGain_43dB   = 0x06 << 4,	// 110b - 43 dB
			RxGain_48dB   = 0x07 << 4,	// 111b - 48 dB, maximum
			RxGain_min    = 0x00 << 4,	// 000b - 18 dB, minimum, convenience for RxGain_18dB
			RxGain_avg    = 0x04 << 4,	// 100b - 33 dB, average, convenience for RxGain_33dB
			RxGain_max    = 0x07 << 4	// 111b - 48 dB, maximum, convenience for RxGain_48dB
		} RX_GAIN;

		// Return codes from the functions in this class. Remember to update get_status_code_name() if you add more.
		// last value set to 0xff, then compiler uses less ram, it seems some optimisations are triggered
		typedef enum class status_code : uint8_t {
			OK             ,		// Success
			ERROR          ,		// Error in communication
			COLLISION      ,		// Collission detected
			TIMEOUT        ,		// Timeout in communication.
			NO_ROOM        ,		// A buffer is not big enough.
			INTERNAL_ERROR ,		// Internal error in the code. Should not happen ;-)
			INVALID        ,		// Invalid argument.
			CRC_WRONG      ,		// The CRC_A does not match
			MIFARE_NACK    = 0xff	// A MIFARE PICC responded with NAK.
		} STATUS_CODE;


		class Picc
		{
			public:
				typedef enum class cmd : uint8_t {
					// The commands used by the PCD to manage communication with several PICCs (ISO 14443-3, Type A, section 6.4)
					REQA       = 0x26,	// REQuest command, Type A. Invites PICCs in state IDLE to go to READY and prepare for anticollision or selection. 7 bit frame.
					WUPA       = 0x52,	// Wake-UP command, Type A. Invites PICCs in state IDLE and HALT to go to READY(*) and prepare for anticollision or selection. 7 bit frame.
					CT         = 0x88,	// Cascade Tag. Not really a command, but used during anti collision.
					SEL_CL1    = 0x93,	// Anti collision/Select, Cascade Level 1
					SEL_CL2    = 0x95,	// Anti collision/Select, Cascade Level 2
					SEL_CL3    = 0x97,	// Anti collision/Select, Cascade Level 3
					HLTA       = 0x50,	// HaLT command, Type A. Instructs an ACTIVE PICC to go to state HALT.
					RATS       = 0xE0,	// Request command for Answer To Reset.
					AUTH_KEY_A = 0x60,	// Perform authentication with Key A
					AUTH_KEY_B = 0x61,	// Perform authentication with Key B
					READ	   = 0x30,	// Reads one 16 byte block from the authenticated sector of the PICC. Also used for MIFARE Ultralight.
					WRITE      = 0xA0,	// Writes one 16 byte block to the authenticated sector of the PICC. Called "COMPATIBILITY WRITE" for MIFARE Ultralight.
					DECREMENT  = 0xC0,	// Decrements the contents of a block and stores the result in the internal data register.
					INCREMENT  = 0xC1,	// Increments the contents of a block and stores the result in the internal data register.
					RESTORE    = 0xC2,	// Reads the contents of a block into the internal data register.
					TRANSFER   = 0xB0,	// Writes the contents of the internal data register to a block.
				} CMD;
		};

	private:
		Spi::Slave m_slave;

	public:
		Mfrc522(Spi::Slave slave, BIT rst_pin, BIT rst_ddb);
		~Mfrc522();

		bool is_card_present();
		const char* get_status_code_name(Mfrc522::STATUS_CODE status_code);
	//	void read_card();
	//	void dump_data();

		uint8_t read_reg(REG reg);
		void write_reg(REG reg, uint8_t data);
		void write_reg(REG reg, uint8_t len, uint8_t* data);
		uint8_t read_reg(uint8_t reg);
		void read_reg(Mfrc522::REG reg,
                              uint8_t  len,
                              uint8_t *data,
                              uint8_t  rx_align);
		void write_reg(uint8_t reg, uint8_t data);
		void clear_mask_reg(REG reg, uint8_t mask);
		void set_mask_reg(REG reg, uint8_t mask);

	private:
		void reset(BIT rst_pin, BIT rst_ddb);
		void init();
		void antenna_on();
		void reset_baud_rates();
		Mfrc522::STATUS_CODE reqa(uint8_t *data_ATQA,
                                          uint8_t *len_ATQA);
		Mfrc522::STATUS_CODE wupa(uint8_t *data_ATQA,
                                          uint8_t *len_ATQA);
		Mfrc522::STATUS_CODE reqa_or_wupa(Mfrc522::Picc::CMD cmd,
					          uint8_t *data_ATQA,
                                                  uint8_t *len_ATQA);
		Mfrc522::STATUS_CODE transceive(uint8_t *data_out,
                                                uint8_t  len_out,
                                                uint8_t *data_in,
                                                uint8_t *len_in,
                                                uint8_t *valid_bits,
                                                uint8_t  rx_align,
                                                bool     check_CRC);
		Mfrc522::STATUS_CODE communicate_with_picc(Mfrc522::CMD command,
				                           uint8_t  wait_IRq,
		                                           uint8_t *data_out,
		                                           uint8_t  len_out,
		                                           uint8_t *data_in,
		                                           uint8_t *len_in,
		                                           uint8_t *valid_bits,
		                                           uint8_t  rx_align,
		                                           bool     check_CRC);

};

#endif
