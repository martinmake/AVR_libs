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
				: m_type(initial_type)
			{
				glCall(glGenBuffers(1, &m_renderer_id)); bind();
				TRACE("GL: BUFFER: GENERATED: {0}", m_renderer_id);
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

				m_type = other.m_type;
			}
			void Base::move(Base&& other)
			{
				GraphicsObject::Base::move(std::move(other));

				m_type = other.m_type;
			}
		}
	}
}
