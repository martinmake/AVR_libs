#ifndef _GRA_VERTEX_BUFFER_H_
#define _GRA_VERTEX_BUFFER_H_

#include <inttypes.h>

#include "gra/glstd.h"
#include "gra/gldebug.h"

namespace Gra
{
	class VertexBuffer
	{
		private:
			unsigned int m_renderer_id;

		public:
			VertexBuffer(void);
			VertexBuffer(const void* initial_data, uint32_t size);
			~VertexBuffer(void);

		public:
			void data(const void* new_data, uint32_t size);

		public:
			void bind(void)   const;
			void unbind(void) const;
	};

	inline void VertexBuffer::bind(void) const
	{
		glCall(glBindBuffer(GL_ARRAY_BUFFER, m_renderer_id));
	}

	inline void VertexBuffer::unbind(void) const
	{
		glCall(glBindBuffer(GL_ARRAY_BUFFER, 0));
	}
}

#endif
