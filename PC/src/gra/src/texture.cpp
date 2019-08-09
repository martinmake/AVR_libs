#include <inttypes.h>
#include <vector>

#include <vendor/stb/image.h>

#include "texture.h"

namespace Gra
{
	Texture::Texture(unsigned int slot)
	{
		stbi_set_flip_vertically_on_load(true);

		glCall(glGenTextures(1, &m_renderer_id));
		bind(slot);

		glCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
		glCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
		glCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
		glCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
	}

	Texture::Texture(const std::string& filepath, unsigned int slot)
		: Texture(slot)
	{
		load(filepath);
	}

	Texture::~Texture(void)
	{
		glCall(glDeleteTextures(1, &m_renderer_id));
	}

	bool Texture::load(std::string filepath)
	{
		m_local_buffer = stbi_load(filepath.c_str(), &m_width, &m_height, &m_bpp, 4);

		if (!m_local_buffer)
		{
			std::cout << "[TEXTURE] UNABLE TO LOAD '" << filepath << '`' <<  std::endl;
			return false;
		}

		glCall(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_width, m_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, m_local_buffer));

		stbi_image_free(m_local_buffer);
		m_local_buffer = nullptr;

		return true;
	}
}
