#include <utility>

#include "logging.h"

#include "gra/graphics_objects/textures/base.h"

namespace Gra
{
	namespace GraphicsObject
	{
		namespace Texture
		{
			Base::Base(uint8_t initial_slot)
				: m_slot(initial_slot)
			{
				assert(m_slot < 32);
				glCall(glGenTextures(1, &m_renderer_id));
				TRACE("OPENGL: TEXTURE: GENERATED: {0}", m_renderer_id);
			}
			Base::Base(GLenum initial_type, uint8_t initial_slot)
				: Base(initial_slot)
			{
				m_type = initial_type;
			}

			Base::~Base(void)
			{
				if (m_renderer_id)
				{
					glCall(glDeleteTextures(1, &m_renderer_id));
					TRACE("OPENGL: TEXTURE: DELETED: {0}", m_renderer_id);
				}
			}

			void Base::copy(const Base& other)
			{
				GraphicsObject::Base::copy(other);

				m_slot = other.m_slot;
			}
			void Base::move(Base&& other)
			{
				GraphicsObject::Base::move(std::move(other));

				m_local_buffer = std::exchange(other.m_local_buffer, nullptr);
				m_slot         = std::exchange(other.m_slot,         0      );
			}
		}
	}
}
