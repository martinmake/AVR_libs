#ifndef _GRA_TEXTURE_H_
#define _GRA_TEXTURE_H_

#include <inttypes.h>

#include "gra/glstd.h"
#include "gra/gldebug.h"

namespace Gra
{
	class Texture
	{
		private:
			unsigned int m_renderer_id;
			uint8_t* m_local_buffer;
			int m_width,
			    m_height;
			int m_bpp;

		public:
			Texture(unsigned int slot = 0);
			Texture(const std::string& filepath, unsigned int slot = 0);
			~Texture(void);

		public:
			bool load(std::string filepath);
			void bind(unsigned int slot = 0) const;
			void unbind(void) const;

		// GETTERS
		public:
			int width(void) const;
			int height(void) const;
	};

	inline void Texture::bind(unsigned int slot) const
	{
		assert(slot < 32);

		glCall(glActiveTexture(GL_TEXTURE0 + slot));
		glCall(glBindTexture(GL_TEXTURE_2D, m_renderer_id));
	}

	inline void Texture::unbind(void) const
	{
		glCall(glBindTexture(GL_TEXTURE_2D, 0));
	}

	// GETTERS
	inline int Texture::width(void) const
	{
		return m_width;
	}
	inline int Texture::height(void) const
	{
		return m_height;
	}
}

#endif
