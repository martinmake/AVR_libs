#include "usart/queue.h"

Queue::Queue(uint8_t size)
	: m_size(size), m_begin(0), m_end(0), m_is_empty(true), m_is_full(false)
{
	m_buffer = new char[m_size];
}

Queue::Queue()
	: m_size(0), m_begin(0), m_end(0), m_is_empty(true), m_is_full(true)
{
}

Queue::~Queue()
{
	delete m_buffer;
}
