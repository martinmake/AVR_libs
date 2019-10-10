#include <utility>

#include "logging.h"

#include "gra/graphics_objects/buffers/base.h"

namespace Gra
{
	namespace GraphicsObject
	{
		namespace Buffer
		{
			Base::Base(GLenum initial_type)
				: m_type(initial_type), m_size(0)
			{
				glCall(glGenBuffers(1, &m_renderer_id)); bind();
				TRACE("GL: BUFFER: GENERATED: {0}", m_renderer_id);
			}
			Base::Base(GLenum initial_type, size_t initial_size)
				: Base(initial_type)
			{
				size(initial_size);
			}
			Base::Base(GLenum initial_type, const void* initial_data, size_t initial_size)
				: Base(initial_type)
			{
				data(initial_data, initial_size);
			}

			Base::~Base(void)
			{
				if (m_renderer_id)
				{
					glCall(glDeleteBuffers(1, &m_renderer_id));
					TRACE("GL: BUFFER: DELETED: {0}", m_renderer_id);
				}
			}

			void Base::copy(const Base& other)
			{
				GraphicsObject::Base::copy(other);

				if (m_size != other.m_size)
					size(other.m_size);

				glCall(glBindBuffer(GL_COPY_READ_BUFFER, other.m_renderer_id));
				glCall(glBindBuffer(GL_COPY_WRITE_BUFFER,      m_renderer_id));
				glCall(glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, m_size));
			}
			void Base::move(Base&& other)
			{
				GraphicsObject::Base::move(std::move(other));

				m_type = other.m_type;
			}
		}
	}
}
