#include "gra/index_buffer.h"

namespace Gra
{
	IndexBuffer::IndexBuffer(void)
		: m_renderer_id(0)
	{
	}
	IndexBuffer::IndexBuffer(const unsigned int* data, uint32_t count)
		: m_count(count)
	{
		glCall(glGenBuffers(1, &m_renderer_id));
		bind();
		glCall(glBufferData(GL_ELEMENT_ARRAY_BUFFER, count * sizeof(unsigned int), data, GL_STATIC_DRAW));
	}

	IndexBuffer::~IndexBuffer(void)
	{
		if (!m_renderer_id) return;
		glCall(glDeleteBuffers(1, &m_renderer_id));
	}
}
