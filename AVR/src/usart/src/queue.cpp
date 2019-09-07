#include "usart/queue.h"

Queue::Queue(uint8_t size)
	: m_size(size), m_start(0), m_end(0)
{
	m_buffer = new char[m_size];
}

Queue::Queue()
{
}

Queue::~Queue()
{
}
