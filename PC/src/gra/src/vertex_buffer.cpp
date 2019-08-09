#include "vertex_buffer.h"

namespace Gra
{
	VertexBuffer::VertexBuffer(void)
	{
	}
	VertexBuffer::VertexBuffer(const void* data, uint32_t size)
	{
		glCall(glGenBuffers(1, &m_renderer_id));
		glCall(glBindBuffer(GL_ARRAY_BUFFER, m_renderer_id));
		glCall(glBufferData(GL_ARRAY_BUFFER, size, data, GL_STATIC_DRAW));
	}

	VertexBuffer::~VertexBuffer(void)
	{
		glCall(glDeleteBuffers(1, &m_renderer_id));
	}
}
