#include <inttypes.h>
#include <utility>
#include <vector>

#include <vendor/stb/image.h>

#include "logging.h"

#include "gra/graphics_objects/textures/plane.h"

namespace Gra {
	namespace GraphicsObject
	{
		namespace Texture
		{
			Plane::Plane(unsigned int initial_slot)
				: Base(GL_TEXTURE_2D, initial_slot), m_width(0), m_height(0)
			{
				glCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
				glCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
				glCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
				glCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
				TRACE("TEXTURE: PLANE: CONSTRUCTED: {0}", (void*) this);
			}
			Plane::Plane(const std::string& filepath, unsigned int initial_slot)
				: Plane(initial_slot)
			{
				load(filepath);
			}

			Plane::~Plane(void)
			{
				stbi_image_free(m_local_buffer);
				if (m_local_buffer)
					TRACE("STB: IMAGE: FREED: {0}", m_local_buffer);
				TRACE("TEXTURE: PLANE: DESTRUCTED: {0}", (void*) this);
			}

			void Plane::load(std::string filepath)
			{
				stbi_set_flip_vertically_on_load(true);

				m_local_buffer = stbi_load(filepath.c_str(), &m_width, &m_height, nullptr, 4);
				TRACE("STB: IMAGE: LOADED: {0}", m_local_buffer);

				if (!m_local_buffer)
				{
					std::cout << "[TEXTURE] UNABLE TO LOAD '" << filepath << '`' <<  std::endl;
					return;
				}

				bind();
				glCall(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_width, m_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, m_local_buffer));
			}

			void Plane::copy(const Plane& other)
			{
				Texture::Base::copy(other);

				stbi_image_free(m_local_buffer);
				TRACE("STB: IMAGE: FREED: {0}", m_local_buffer);
				m_local_buffer = stbi_load_from_memory(other.m_local_buffer, 1, &m_width, &m_height, nullptr, 4);
				TRACE("STB: IMAGE: LOADED FROM MEMORY: {0}", m_local_buffer);

				bind();
				glCall(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_width, m_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, m_local_buffer));
			}
			void Plane::move(Plane&& other)
			{
				stbi_image_free(m_local_buffer);
				TRACE("STB: IMAGE: FREED: {0}", m_local_buffer);
				Texture::Base::move(std::move(other));

				m_width  = std::exchange(other.m_width,  0);
				m_height = std::exchange(other.m_height, 0);
			}
		}
	}
}
