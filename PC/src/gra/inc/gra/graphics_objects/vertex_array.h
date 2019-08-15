#ifndef _GRA_GRAPHICS_OBJECT_VERTEX_ARRAY_H_
#define _GRA_GRAPHICS_OBJECT_VERTEX_ARRAY_H_

#include <inttypes.h>
#include <vector>

#include "gra/glstd.h"
#include "gra/gldebug.h"

#include "gra/graphics_objects/buffers/vertex.h"

namespace Gra
{
	namespace GraphicsObject
	{
		class VertexArray : public GraphicsObject::Base
		{
			public:
				struct Layout
				{
					struct Element
					{
						unsigned int  type;
						unsigned int  count;
						unsigned char normalized;
					};

					std::vector<Element> elements;
					unsigned int         stride;

					template <typename T>
					void push(uint32_t count);
				};

			private:
				Buffer::Vertex m_vertex_buffer;
				Layout         m_layout;

			public:
				VertexArray(void);
				VertexArray(const Buffer::Vertex& initial_vertex_buffer, const Layout& initial_layout);

				VertexArray(const VertexArray&  other);
				VertexArray(      VertexArray&& other);

				~VertexArray(void);

			public:
				void bind(void)   const override;
				void unbind(void) const override;

			public: // SETTERS
				void vertex_buffer(const Buffer::Vertex& new_vertex_buffer);
				void layout       (const         Layout& new_layout       );

			private:
				void copy(const Base&  other);
				void move(      Base&& other);
			public:
				VertexArray& operator=(const VertexArray&  rhs);
				VertexArray& operator=(      VertexArray&& rhs);
		};

		inline void VertexArray::  bind(void) const { glCall(glBindVertexArray(m_renderer_id)); }
		inline void VertexArray::unbind(void) const { glCall(glBindVertexArray(            0)); }

		inline VertexArray::VertexArray(const VertexArray&  other) : Base() { copy(          other ); }
		inline VertexArray::VertexArray(      VertexArray&& other) : Base() { move(std::move(other)); }

		inline VertexArray& VertexArray::operator=(const VertexArray&  rhs) { copy(          rhs ); return *this; }
		inline VertexArray& VertexArray::operator=(      VertexArray&& rhs) { move(std::move(rhs)); return *this; }

		template <typename T> void VertexArray::Layout::push(uint32_t count)
		{
			(void) count;
			static_assert(sizeof(T) && false, "[VERTEX ARRAY LAYOUT] TYPE UNAVAIBLE!");
		}
		template <> void VertexArray::Layout::push<         float>(uint32_t count);
		template <> void VertexArray::Layout::push<unsigned int  >(uint32_t count);
		template <> void VertexArray::Layout::push<unsigned char >(uint32_t count);
	}
}

#endif
