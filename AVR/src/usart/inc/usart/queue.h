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
		         uint8_t m_size;
		volatile uint8_t m_start;
		volatile uint8_t m_end;
		         char*   m_buffer;
		         bool    m_is_empty;
		         bool    m_is_full;
};

inline bool Queue::is_empty(void) const { return m_is_empty; }
inline bool Queue::is_full (void) const { return m_is_full;  }

inline Queue& Queue::operator<<(char c)
{
	while (is_full())
	{
		uint8_t sreg_save = SREG;
		sei();
		while ((m_end + 1) % m_size == m_start) {}
		SREG = sreg_save;
	}

	m_buffer[m_end] = c;
	m_end = (m_end + 1) % m_size;

			      m_is_empty = false;
	if (m_start == m_end) m_is_full  = true;

	return *this;
}
inline void Queue::operator>>(volatile uint8_t& c)
{
	if (!is_empty())
	{
		c = m_buffer[m_start];
		m_start = (m_start + 1) % m_size;

		                      m_is_full  = false;
		if (m_start == m_end) m_is_empty = true;
	}
}

#endif
