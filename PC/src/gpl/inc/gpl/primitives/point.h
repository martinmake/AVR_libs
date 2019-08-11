#ifndef _GPL_PRIMITIVE_POINT_H_
#define _GPL_PRIMITIVE_POINT_H_

#include "gpl/primitives/base.h"
#include "gpl/gpl.h"

namespace Gpl
{
	namespace Primitive
	{
		class Point : public Base
		{
			private:
				glm::vec3 m_position;
				float m_size;
			public:
				static Gra::VertexBufferLayout s_vertex_buffer_layout;
				static Gra::IndexBuffer        s_index_buffer;

			public:
				Point(const glm::vec3& initial_position, const glm::vec4& initial_color, float initial_size);
				virtual ~Point(void);

			public:
				void draw(const Gra::Renderer& renderer, const glm::mat4& mvp) const override;

			public: // GETTERS
				const glm::vec3& position(void) const;
				      float     size     (void) const;
			public: // SETTERS
				void position(const glm::vec3& new_position);
				void size    (      float     new_size    );
		};

		// GETTERS
		inline const glm::vec3& Point::position(void) const { return m_position; }
		inline       float      Point::size    (void) const { return m_size;     }
		// SETTERS
		inline void Point::size    (      float      new_size    ) { m_size     = new_size;     }
	}
}

#endif
