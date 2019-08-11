#ifndef _GRA_VERTEX_ARRAY_H_
#define _GRA_VERTEX_ARRAY_H_

#include <inttypes.h>

#include "gra/glstd.h"
#include "gra/gldebug.h"

#include "gra/vertex_buffer.h"
#include "gra/vertex_buffer_layout.h"

namespace Gra
{
	class VertexArray
	{
		private:
			unsigned int m_renderer_id;

		public:
			VertexArray(void);
			VertexArray(const VertexBuffer& initial_vertex_buffer, const VertexBufferLayout& initial_layout);
			~VertexArray(void);

		public:
			void bind(void)   const;
			void unbind(void) const;

		public: // SETTERS
			void vertex_buffer(const VertexBuffer      & new_vertex_buffer);
			void layout       (const VertexBufferLayout& new_layout       );
	};

	inline void VertexArray::bind(void) const
	{
		glCall(glBindVertexArray(m_renderer_id));
	}

	inline void VertexArray::unbind(void) const
	{
		glCall(glBindVertexArray(0));
	}
}

#endif
