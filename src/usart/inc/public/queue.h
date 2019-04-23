#ifndef _USART_QUEUE_H_
#define _USART_QUEUE_H_

#include <inttypes.h>

class Queue
{
	private:
		         uint8_t m_size;
		volatile uint8_t m_start;
		volatile uint8_t m_end;
		         char*   m_buffer;

	public:
		Queue(uint8_t size);
		Queue();
		~Queue();

		inline bool is_empty() { return m_start == m_end; }
		inline Queue& operator<<(char c)
		{
			m_buffer[m_end] = c;
			while ((m_end + 1) % m_size == m_start) {}
			m_end = (m_end + 1) % m_size;
			return *this;
		}
		inline void operator>>(volatile uint8_t& c)
		{
			if (!is_empty()) {
				c = m_buffer[m_start];
				m_start = (m_start + 1) % m_size;
			}
		}
};

#endif
