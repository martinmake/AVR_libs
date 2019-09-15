#ifndef _USART_QUEUE_H_
#define _USART_QUEUE_H_

#include <inttypes.h>

#include <util.h>

class Queue
{
	public:
		 Queue(uint8_t size);
		 Queue(void);
		~Queue(void);

	public: // GETTERS
		bool is_empty(void) const;
		bool is_full (void) const;

	public: // OPERATORS
		Queue& operator<<(char c);
		void   operator>>(volatile uint8_t& c);

	private:
		void    lock  (void);
		void  unlock  (void);
		bool is_locked(void) const;

	private:
		         uint8_t m_size;
		volatile uint8_t m_begin;
		volatile uint8_t m_end;
		         char*   m_buffer;
		volatile bool    m_is_empty;
		volatile bool    m_is_full;
		volatile bool    m_is_locked;
};

inline bool Queue::is_empty(void) const { return m_is_empty; }
inline bool Queue::is_full (void) const { return m_is_full;  }

inline Queue& Queue::operator<<(char c)
{
	if (is_full())
	{
		uint8_t sreg_save = SREG;
		sei();
		PORTD=0xff;
		while (is_full()) {}
		PORTD=0x00;
		SREG = sreg_save;
	}

	m_buffer[m_end] = c;
	m_end = (m_end + 1) % m_size;

			      m_is_empty = false;
	if (m_begin == m_end) m_is_full  = true;

	return *this;
}
inline void Queue::operator>>(volatile uint8_t& c)
{
	c = m_buffer[m_begin];
	m_begin = (m_begin + 1) % m_size;

			      m_is_full  = false;
	if (m_begin == m_end) m_is_empty = true;
}

inline void Queue::   lock  (void)       { m_is_locked = true;  }
inline void Queue:: unlock  (void)       { m_is_locked = false; }
inline bool Queue::is_locked(void) const { return m_is_locked;  }

#endif
